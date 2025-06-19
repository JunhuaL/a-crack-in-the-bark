from diffusers.models import AutoencoderKL
from torchvision.io import read_image
from tqdm import tqdm
import argparse
import os
import re
import torch

def load_vae(args):
    if args.model_id == "stabilityai/stable-diffusion-2-1-base":
        vae = AutoencoderKL.from_pretrained(args.model_id,subfolder='vae')
    elif args.model_id == "stabilityai/sdxl-vae":
        vae = AutoencoderKL.from_pretrained(args.model_id)
    elif args.model_id == "ostris/vae-kl-f8-d16":
        vae = AutoencoderKL.from_pretrained(args.model_id)
    vae.eval()
    vae.to(args.device)
    return vae

@torch.inference_mode()
def to_latents(vae: AutoencoderKL, img: torch.Tensor, device: str):
    img = (img/255).unsqueeze(0).to(device)
    encoding_dist = vae.encode(img).latent_dist
    encoding = encoding_dist.mode()
    latents = encoding * 0.18215
    return latents

def main(args):
    vae = load_vae(args)

    img_files = os.listdir(args.img_folder)

    imgs = []
    idxs = []
    for img_fn in img_files:
        idxs.append(int(re.findall('\d+', img_fn)[-1]))
        img = read_image(os.path.join(args.img_folder, img_fn)).type(torch.float32)
        imgs.append(img)

    latents = []
    for img in tqdm(imgs):
        latent = to_latents(vae, img, args.device)[0].cpu()
        latents.append(latent)
    
    latents = torch.stack(latents)
    if not os.path.exists(args.latents_folder):
        os.makedirs(args.latents_folder, exist_ok=True)

    torch.save((idxs, latents), os.path.join(args.latents_folder, args.latents_file + '.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent conversion")
    parser.add_argument('img_folder', type=str)
    parser.add_argument('latents_folder', type=str)
    parser.add_argument('latents_file', type=str)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()
    main(args)
