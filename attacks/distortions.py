import torch
import numpy as np

from torchvision import transforms
from PIL import Image

def circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    return ((x - x0)**2 + (y-y0)**2)<= r**2

def overlay(img1, img2, seed, args):
    overlay_pth = args.mark_path
    atk = Image.open(overlay_pth)
    atk_ratio = args.mark_ratio
    atk = atk.resize((int(atk.size[0]*atk_ratio),int(atk.size[0]*atk_ratio)))
    atk_center = (atk.size[0]//2,atk.size[1]//2)
    #overlay over img1
    img_center = (img1.size[0]//2,img1.size[1]//2)
    img_offset = (img_center[0]-atk_center[0],img_center[1]-atk_center[1])
    img1.paste(atk,img_offset,atk)

    #overlay over img2
    img_center = (img2.size[0]//2,img2.size[1]//2)
    img_offset = (img_center[0]-atk_center[0],img_center[1]-atk_center[1])
    img2.paste(atk,img_offset,atk)

    return img1, img2

def low_pass_filter_img(img,args):
    img_t = transforms.functional.pil_to_tensor(img)
    fft_img = torch.fft.fftshift(torch.fft.fft2(img_t), dim=(-1,-2))

    radius = args.fft_filter_radius
    circle_mask_img = np.array([circle_mask(512,r=radius) for _ in range(3)])
    low_pass = torch.tensor(circle_mask_img).type(torch.float32)

    filtered_img = (fft_img * low_pass)
    ifft_img = torch.fft.ifft2(torch.fft.ifftshift(filtered_img, dim=(-1, -2))).real.clip(0,255).type(torch.uint8)
    img = transforms.functional.to_pil_image(ifft_img)
    return img

def high_pass_filter_img(img,args):
    img_t = transforms.functional.pil_to_tensor(img)
    fft_img = torch.fft.fftshift(torch.fft.fft2(img_t), dim=(-1,-2))

    radius = args.fft_filter_radius
    circle_mask_img = np.array([circle_mask(512,r=radius) for _ in range(3)])
    high_pass = 1 - torch.tensor(circle_mask_img).type(torch.float32)

    filtered_img = (fft_img * high_pass)
    ifft_img = torch.fft.ifft2(torch.fft.ifftshift(filtered_img, dim=(-1, -2))).real.clip(0,255).type(torch.uint8)
    img = transforms.functional.to_pil_image(ifft_img)
    return img

def low_pass(img1, img2, seed, args):
    img1 = low_pass_filter_img(img1,args)
    img2 = low_pass_filter_img(img2,args)
    return img1, img2

def high_pass(img1, img2, seed, args):
    img1 = high_pass_filter_img(img1,args)
    img2 = high_pass_filter_img(img2,args)
    return img1, img2

def spectral_mask(img, args):
    img_t = transforms.functional.pil_to_tensor(img)
    fft_img = torch.fft.fftshift(torch.fft.fft2(img_t), dim=(-1,-2))

    mask_threshold = args.spectral_mask_threshold
    spec_img = torch.log(torch.abs(fft_img))
    spec_img = spec_img/spec_img.max()
    spec_mask = (spec_img > mask_threshold).type(torch.uint8)

    filtered_img = (fft_img * spec_mask)
    ifft_img = torch.fft.ifft2(torch.fft.ifftshift(filtered_img, dim=(-1,-2))).real.clip(0,255).type(torch.uint8)
    img = transforms.functional.to_pil_image(ifft_img)
    return img

def spectrum_thresholding(img1, img2, seed, args):
    img1 = spectral_mask(img1,args)
    img2 = spectral_mask(img2,args)
    return img1, img2

def direct_watermark_removal(img1, img2, seed, args, gt_patch, w_mask):
    img2_t=transforms.functional.pil_to_tensor(img2)
    fft_img2 = torch.fft.fftshift(torch.fft.fft2(img2_t), dim=(-1,-2))

    img2_w_mask = circle_mask(size=fft_img2.shape[-1],r=args.w_radius)
    fft_img2[:,img2_w_mask] = fft_img2[:,img2_w_mask] - gt_patch[0,:3,w_mask[0,args.w_channel].detach().cpu()].detach().cpu()
    ifft_img2 = torch.fft.ifft2(torch.fft.ifftshift(fft_img2, dim=(-1,-2))).real.clip(0,255).type(torch.uint8)
    img2 = transforms.functional.to_pil_image(ifft_img2)
    return img1, img2

def high_freq_operations(img, args):
    img_t = transforms.functional.pil_to_tensor(img)
    fft_img = torch.fft.fftshift(torch.fft.fft2(img_t), dim=(-1,-2))

    radius = args.fft_filter_radius
    circle_mask_ = np.array([circle_mask(img_t.shape[-1],r=radius) for i in range(3)])
    circle_mask_ = torch.tensor(circle_mask_).type(torch.float32)
    high_freq_shift = circle_mask_ * args.fft_filter_intensity

    high_freq_patch = fft_img * high_freq_shift
    filtered_img = (fft_img - high_freq_patch)
    ifft_img = torch.fft.ifft2(torch.fft.ifftshift(filtered_img, dim=(-1, -2))).real.clip(0,255).type(torch.uint8)
    img = transforms.functional.to_pil_image(ifft_img)
    return img

def low_freq_operations(img, args):
    img_t = transforms.functional.pil_to_tensor(img)
    fft_img = torch.fft.fftshift(torch.fft.fft2(img_t), dim=(-1,-2))

    radius = args.fft_filter_radius
    circle_mask_ = np.array([circle_mask(img_t.shape[-1],r=radius) for i in range(3)])
    circle_mask_ = 1 - torch.tensor(circle_mask_).type(torch.float32)
    low_freq_shift = circle_mask_ * args.fft_filter_intensity

    low_freq_patch = fft_img * low_freq_shift
    filtered_img = (fft_img - low_freq_patch)
    ifft_img = torch.fft.ifft2(torch.fft.ifftshift(filtered_img, dim=(-1, -2))).real.clip(0,255).type(torch.uint8)
    img = transforms.functional.to_pil_image(ifft_img)
    return img

def high_freq_shift(img1, img2, seed, args):
    img1 = high_freq_operations(img1, args)
    img2 = high_freq_operations(img2, args)
    return img1, img2

def low_freq_shift(img1, img2, seed, args):
    img1 = low_freq_operations(img1, args)
    img2 = low_freq_operations(img2, args)
    return img1, img2

def color_threshold(img, args):
    img_t = transforms.functional.pil_to_tensor(img)

    for j in range(img_t.shape[0]):
        channel_ratio = img_t[j]/ img_t[j].max()
        mask = channel_ratio < args.color_threshold
        img_t[j,mask] = 0

    img = transforms.functional.to_pil_image(img_t)
    return img

def color_threshold_distortion(img1, img2, seed, args):
    img1 = color_threshold(img1,args)
    img2 = color_threshold(img2,args)
    return img1, img2

def image_shift(tgt_img,src_img,args):
    tgt_t = transforms.functional.pil_to_tensor(tgt_img)
    src_t = transforms.functional.pil_to_tensor(src_img)

    tgt_fft = torch.fft.fftshift(torch.fft.fft2(tgt_t),dim=(-1,-2))
    src_fft = torch.fft.fftshift(torch.fft.fft2(src_t),dim=(-1,-2))

    src2tgt = (tgt_fft - src_fft)
    dist = (src2tgt**2).sum().sqrt()
    e_src2tgt = src2tgt/dist
    src_img = src_fft + (e_src2tgt * (dist * args.shift_ratio))
    src_ifft = torch.fft.ifft2(torch.fft.ifftshift(src_img,dim=(-1,-2))).real.clip(0,255).type(torch.uint8)
    img = transforms.functional.to_pil_image(src_ifft)
    return img

def image_shifting(img1, img2, seed, args):
    ref_img = Image.open(args.ref_img).convert('RGB')
    img1 = image_shift(ref_img,img1,args)
    img2 = image_shift(ref_img,img2,args)
    return img1, img2