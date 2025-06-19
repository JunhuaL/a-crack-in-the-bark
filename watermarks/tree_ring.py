from models.inversable_stable_diffusion import InversableStableDiffusionPipeline
from typing_utils import *

import copy
import numpy as np
import os
import torch

### Adapted from https://github.com/YuxinWenRick/tree-ring-watermark
class TreeRingWm:
    def __init__(self, 
            w_pattern: str = 'ring', w_radius: int = 10, w_mask_shape: str = 'circle', w_channel: int = 3, w_injection: str = 'complex',
            device: str = 'cuda', w_pattern_const: int = 0, img_shape: tuple = (512,512),
            pipe: InversableStableDiffusionPipeline = None,
        ):
        self.w_pattern = w_pattern
        self.w_radius = w_radius 
        self.w_mask_shape = w_mask_shape
        self.w_channel = w_channel
        self.w_injection = w_injection
        self.device = device
        self.w_pattern_const = w_pattern_const
        self.img_shape = img_shape
        self.pipe = pipe

        self.watermark = None
        self.wm_mask = None

    def setup(self):
        self.watermark = self.get_watermarking_pattern()
        self.wm_mask = self.get_watermarking_mask()
        return

    def circle_mask(self, size, radius, x_offset=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
        x0 = y0 = size // 2
        x0 += x_offset
        y0 += y_offset
        y, x = np.ogrid[:size, :size]
        y = y[::-1]
        return ((x - x0)**2 + (y-y0)**2)<= self.w_radius**2
    
    def get_watermarking_pattern(self):
        if self.pipe is not None:
            gt_init = self.pipe.get_random_latents(height = self.img_shape[0], width = self.img_shape[1])
        else:
            gt_init = torch.randn((1, 3, self.img_shape[0], self.img_shape[1]), device=self.device)

        if 'seed_ring' in self.w_pattern:
            gt_patch = gt_init

            gt_patch_tmp = copy.deepcopy(gt_patch)
            for i in range(self.w_radius, 0, -1):
                tmp_mask = self.circle_mask(gt_init.shape[-1], i)
                tmp_mask = torch.tensor(tmp_mask).to(self.device)
                
                for j in range(gt_patch.shape[1]):
                    gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
        elif 'seed_zeros' in self.w_pattern:
            gt_patch = gt_init * 0
        elif 'seed_rand' in self.w_pattern:
            gt_patch = gt_init
        elif 'rand' in self.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
            gt_patch[:] = gt_patch[0]
        elif 'zeros' in self.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        elif 'const' in self.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
            gt_patch += self.w_pattern_const
        elif 'ring' in self.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
            gt_patch_tmp = copy.deepcopy(gt_patch)
            for i in range(self.w_radius, 0, -1):
                tmp_mask = self.circle_mask(gt_init.shape[-1], i)
                tmp_mask = torch.tensor(tmp_mask).to(self.device)
                
                for j in range(gt_patch.shape[1]):
                    gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

        return gt_patch
    
    def get_watermarking_mask(self):
        watermarking_mask = torch.zeros(self.watermark.shape, dtype=torch.bool).to(self.device)
        if self.w_mask_shape == 'circle':
            np_mask = self.circle_mask(self.watermark.shape[-1], self.w_radius)
            torch_mask = torch.tensor(np_mask).to(self.device)

            if self.w_channel == -1:
                # all channels
                watermarking_mask[:, :] = torch_mask
            else:
                watermarking_mask[:, self.w_channel] = torch_mask
        elif self.w_mask_shape == 'square':
            anchor_p = self.watermark.shape[-1] // 2
            if self.w_channel == -1:
                # all channels
                watermarking_mask[:, :, anchor_p-self.w_radius:anchor_p+self.w_radius, anchor_p-self.w_radius:anchor_p+self.w_radius] = True
            else:
                watermarking_mask[:, self.w_channel, anchor_p-self.w_radius:anchor_p+self.w_radius, anchor_p-self.w_radius:anchor_p+self.w_radius] = True
        elif self.w_mask_shape == 'no':
            pass
        else:
            raise NotImplementedError(f'w_mask_shape: {self.w_mask_shape}')
        
        return watermarking_mask
    
    def inject(self, init_latents):
        if self.w_injection == 'complex':
            init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim = (-1, -2))
            init_latents_w_fft[self.wm_mask] = self.watermark[self.wm_mask].clone()
            init_latents = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim = (-1, -2))).real
        elif self.w_injection == 'seed':
            init_latents[self.wm_mask] = self.watermark[self.wm_mask].clone()
        else:
            NotImplementedError(f'w_injection: {self.w_injection}')
        
        return init_latents
    
    def eval(self, latent):
        if self.w_injection == 'complex':
            pred_init_latent = torch.fft.fftshift(torch.fft.fft2(latent), dim = (-1, -2))
        elif self.w_injection == 'seed':
            pred_init_latent = latent
        else:
            NotImplementedError(f'w_injection: {self.w_injection}')
        
        score = torch.abs(pred_init_latent[self.wm_mask] - self.watermark[self.wm_mask]).mean().item()
        return score
    
    def export(self):
        tr_params = dict()
        tr_params['w_pattern'] = self.w_pattern
        tr_params['w_radius'] = self.w_radius
        tr_params['w_mask_shape'] = self.w_mask_shape
        tr_params['w_channel'] = self.w_channel
        tr_params['w_injection'] = self.w_injection
        tr_params['w_pattern_const'] = self.w_pattern_const
        tr_params['img_shape'] = self.img_shape
        
        tmp_watermark = self.watermark.clone().cpu()
        watermark = {"real": tmp_watermark.real,
                     "imag": tmp_watermark.imag
        }
        tr_params['watermark'] = watermark
        tr_params['wm_mask'] = self.wm_mask.clone().cpu()
        return tr_params

    def save(self, output_dir, filename='tr_params.pth'):
        params = self.export()
        save_path = os.path.join(output_dir, filename)
        torch.save(params, save_path)

    def load(self, filepath):
        params = torch.load(filepath)
        self.w_pattern = params['w_pattern']
        self.w_radius = params['w_radius']
        self.w_mask_shape = params['w_mask_shape']
        self.w_channel = params['w_channel']
        self.w_injection = params['w_injection']
        self.w_pattern_const = params['w_pattern_const']
        self.img_shape = params['img_shape']

        tmp_watermark = params['watermark']
        watermark = tmp_watermark['real'].to(torch.cfloat)
        watermark.imag = tmp_watermark['imag']
        self.watermark = watermark.to(self.device)
        self.wm_mask = params['wm_mask'].to(self.device)
