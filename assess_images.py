from diffusers import DPMSolverMultistepScheduler
from Logger import *
from models.guided_diffusion import GuidedDiffusionPipeline
from models.inversable_stable_diffusion import InversableStableDiffusionPipeline
from scores import clip, lpips, pytorch_fid
from utils import read_json
from models import open_clip
from watermarks import TreeRingWm
from datasets import ImgLoader
from tqdm import tqdm
from torchvision import transforms
from sklearn import metrics

import argparse
import pandas as pd

def setup_logger(args):
    logger = Logger(name=args.run_name)
    logger.config(args)
    table_specs = {'adv_score': float, 'no_w_score': float, 'prompt': str}

    if args.model_id != '512x512_diffusion':
        table_specs['clip_diff'] = float
        table_specs['lpips_diff'] = float

    logger.create_table(table_specs)
    return logger

def get_model(args):
    if args.model_id == 'stabilityai/stable-diffusion-2-1-base':
        scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder = 'scheduler')
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            args.model_id,
            scheduler = scheduler,
            torch_dtype = torch.float16,
            revision = 'fp16'
        )
        pipe.to(args.device)
    elif args.model_id == '512x512_diffusion':
        model_params = read_json(f'{args.model_id}.json')
        model_params['timestep_respacing'] = f'ddim{args.num_inference_steps}'
        pipe = GuidedDiffusionPipeline(
            model_params, 
            num_images = args.num_images, 
            device = args.device 
        )
    return pipe

def setup_scoring_funcs(args):
    clip_score = None
    lpips_score = None 
    fid_score = None
    ### CLIP Score
    if args.model_id != '512x512_diffusion':
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
            args.reference_model, 
            pretrained=args.reference_model_pretrain, 
            device=args.device
        )
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

        clip_score = clip.CLIPScore(
            ref_model, 
            ref_clip_preprocess, 
            ref_tokenizer, 
            args.device
        )
    
    ### LPIPS Score
    if args.model_id != '512x512_diffusion':
        lpips_model = lpips.load_model(mode='alex', device=args.device)
        lpips_score = (lpips_model,lpips.compute_lpips)

    ### FID Score
    fid_score = pytorch_fid.calculate_fid_given_paths

    return clip_score, lpips_score, fid_score

def extract_prompt(prompt):
    if isinstance(prompt,str):
        return prompt
    
    prompt = prompt['y'].squeeze(0).item()
    return prompt

def main(args):
    logger = setup_logger(args)
    clip_score,lpips_score,fid_score = setup_scoring_funcs(args)
    
    pipe = get_model(args)
    preprocess_func = None

    if args.model_id != '512x512_diffusion':
        preprocess_func = pipe.get_image_latents
        tester_prompt = ''
        text_embeddings = pipe.get_text_embedding(tester_prompt)

    wm_injector = TreeRingWm(
        device = args.device,
        pipe = pipe
    )
    wm_injector.load( args.watermark_path )

    table = pd.read_csv(args.table_path)
    wm_imgs = ImgLoader(args.original_images_path, mapping_table=table, key='wm_img', device=args.device)
    adv_imgs = ImgLoader(args.adv_images_path, mapping_table=table, key='wm_img', device=args.device)
    ref_imgs = ImgLoader(args.imagenet_path)

    for i in tqdm(range(len(wm_imgs))):
        wm_img, wm_fp, wm_prompt = wm_imgs[i]
        adv_img, adv_fp, adv_prompt = adv_imgs[i]
        ref_img, _ = ref_imgs[i]

        wm_img = wm_img.unsqueeze(0).half().to(args.device)
        adv_img = adv_img.unsqueeze(0).half().to(args.device)
        ref_img = ref_img.unsqueeze(0).half().to(args.device)

        adv_latent = preprocess_func(adv_img/255, sample=False) if preprocess_func is not None else adv_img
        init_adv_latent = pipe.forward_diffusion(
            latents=adv_latent,
            text_embeddings=adv_prompt if args.model_id == '512x512_diffusion' else text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps
        )

        ref_latent = preprocess_func(ref_img/255, sample=False) if preprocess_func is not None else ref_img
        init_ref_latent = pipe.forward_diffusion(
            latents=ref_latent,
            text_embeddings=adv_prompt if args.model_id == '512x512_diffusion' else text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps
        )

        adv_score = wm_injector.eval(init_adv_latent)
        ref_score = wm_injector.eval(init_ref_latent)
        entry = [-adv_score, -ref_score, extract_prompt(wm_prompt)]

        if clip_score is not None:
            img1 = transforms.functional.to_pil_image(wm_img[0].to(torch.uint8))
            img2 = transforms.functional.to_pil_image(adv_img[0].to(torch.uint8))
            sims = clip_score([img1,img2],wm_prompt)
            clip_diff = abs((sims[0] - sims[1]))
            entry.append(clip_diff.cpu().item())
        
        if lpips_score is not None:
            lpips_w = lpips_score[1](ref_img, wm_img, lpips_score[0], args.device)
            lpips_adv = lpips_score[1](ref_img, adv_img, lpips_score[0], args.device)
            lpips_diff = abs(lpips_w-lpips_adv)
            entry.append(lpips_diff)
        
        logger.add_data(entry)
    
    preds = logger.get_data('adv_score') + logger.get_data('no_w_score')
    t_labels = [1] * (len(preds)//2) + [0] * (len(preds)//2)

    # Accuracy measures
    fpr, tpr, thresholds = metrics.roc_curve(t_labels,preds,pos_label=1)
    auroc = metrics.auc(fpr, tpr)
    acc = np.max(1-(fpr + (1-tpr))/2)
    low = tpr[np.where(fpr<.01)[0][-1]]

    prec, reca, thresholds = metrics.precision_recall_curve(y_true=t_labels,y_score=preds,pos_label=1)
    auprc = metrics.auc(reca,prec)

    columns = ['thresholds','precision','recall']
    prc_table = list(zip(thresholds,prec,reca))
    prc_table = pd.DataFrame(prc_table,columns=columns)
    table_dir = os.path.split(logger.table_dir)[0]
    prc_table.to_csv(os.path.join(table_dir,'PRC_Table.csv'))

    print(f'auroc: {auroc}, auprc: {auprc}, acc: {acc}, TPR@1%FPR: {low}')
    # FID
    fid_value_w = fid_score(
        [args.imagenet_path, args.original_images_path],
        batch_size = 32,
        device = args.device,
        dims = 2048,
        num_workers = 0,
    )
    
    fid_value_adv = fid_score(
        [args.imagenet_path, args.adv_images_path],
        batch_size = 32,
        device = args.device,
        dims = 2048,
        num_workers = 0,
    )

    final_log = {
        'auroc': auroc, 'auprc': auprc, 'acc': acc, 'TPR@1%FPR': low,
        'fid_adv': abs(fid_value_w - fid_value_adv),
    }

    print(f'fid_adv: {abs(fid_value_w - fid_value_adv)}')

    if clip_score is not None:
        clip_diff_data = logger.get_data('clip_diff')
        clip_diff_mean = np.mean(clip_diff_data)
        clip_diff_std = np.std(clip_diff_data)
        final_log['clip_diff_mean'] = clip_diff_mean
        final_log['clip_diff_std'] = clip_diff_std
        print(f'clip_diff_mean: {clip_diff_mean}')
    if lpips_score is not None:
        lpips_diff_data = logger.get_data('lpips_diff')
        lpips_diff_mean = np.mean(lpips_diff_data)
        lpips_diff_std = np.std(lpips_diff_data)
        final_log['lpips_diff_mean'] = lpips_diff_mean
        final_log['lpips_diff_std'] = lpips_diff_std
        print(f'lpips_diff_mean: {lpips_diff_mean}')
    logger.log(final_log)
    logger.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='Assess Images')
    parser.add_argument('--original_images_path', default=None)
    parser.add_argument('--adv_images_path', default=None)
    parser.add_argument('--table_path', default=None)
    parser.add_argument('--imagenet_path', default=None)
    parser.add_argument('--watermark_path', default=None)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')

    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--test_num_inference_steps', default=50, type=int)
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=999999, type=int)
    args = parser.parse_args()
    main(args)
