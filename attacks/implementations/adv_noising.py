from attacks.attack import Attack
from diffusers import DPMSolverMultistepScheduler
from models.inversable_stable_diffusion_with_grad import InversableStableDiffusionPipeline
from torch.nn.functional import l1_loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing_utils import *
import numpy as np
import torch

class AdversarialNoising(Attack):
    def __init__(
            self, eps: float, alpha: float, n_steps: int, batch_size: int,
            surrogate_diff_model: str, inference_steps: int, tr_params: dict, 
            device: str
        ):
        super().__init__(eps, alpha, n_steps, batch_size)
        scheduler = DPMSolverMultistepScheduler.from_pretrained(surrogate_diff_model, subfolder='scheduler')
        self.pipe = InversableStableDiffusionPipeline.from_pretrained(surrogate_diff_model, scheduler=scheduler)
        self.pipe.requires_safety_checker = False
        self.pipe.vae.enable_gradient_checkpointing()
        self.pipe.unet.enable_gradient_checkpointing()
        self.pipe.vae.train()
        self.pipe.unet.train()
        self.pipe.to(device)
        self.text_embedding = self.pipe.get_text_embedding('')

        self.tr_params = tr_params
        self.inference_steps = inference_steps
        self.device = device

    def setup(self, dataset: DataLoader):
        ### Adversarial noising has no special setup
        pass

    def verify_key_with_grad(self, pred, true):
        mask = self.get_watermarking_mask(pred)
        distance = - torch.linalg.norm(pred[mask] + true[mask])
        return distance

    def extract_with_grad(self, x: torch.Tensor):
        text_embedding = self.text_embedding.clone().to(self.device)
        img_latents = self.pipe.get_image_latents(x, sample=False)

        reversed_latents = self.pipe.forward_diffusion(
            latents=img_latents,
            text_embeddings=text_embedding,
            guidance_scale=1,
            num_inference_steps=self.inference_steps
        )
        reversed_latents_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents), dim=(-1, -2))
        return reversed_latents_fft

    @torch.no_grad()
    def extract(self, x: torch.Tensor):
        text_embedding = self.text_embedding.clone().to(self.device)
        img_latents = self.pipe.get_image_latents(x, sample=False)

        reversed_latents = self.pipe.forward_diffusion(
            latents=img_latents,
            text_embeddings=text_embedding,
            guidance_scale=1,
            num_inference_steps=self.inference_steps
        )

        if 'complex' in self.tr_params['w_injection']:
            reversed_latents_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents),dim=(-1,-2))
        elif 'seed' in self.tr_params['w_injection']:
            reversed_latents_fft = reversed_latents
        
        return reversed_latents_fft

    def noise_image(self, x: torch.Tensor):
        noise = torch.zeros_like(x, device=self.device)
        noise.requires_grad = True
        y_true = None

        opt = Adam([noise],lr=self.alpha)
        for _ in range(self.n_steps):
            opt.zero_grad()

            if y_true is None:
                with torch.no_grad():
                    y_true = self.extract(x)
            
            y_pred = self.extract_with_grad(torch.clamp(x + noise, 0, 1))

            loss = self.verify_key_with_grad(y_pred, y_true)

            loss.backward()
            opt.step()

            noise.data = torch.clamp(noise.data, -self.eps, self.eps)

        return torch.clamp(x + noise, 0, 1).detach()

    def attack(self, dataset: DataLoader):
        data_out = []
        filenames = []
        for data, fps in dataset:
            data = data.to(self.device) / 255

            out = self.noise_image(data) * 255

            data_out.append(out)
            filenames += fps
        data_out = torch.concat(data_out)
        return data_out, filenames
    
    def get_watermarking_mask(self, x: torch.Tensor):
        watermarking_mask = torch.zeros(x.shape, dtype=torch.bool).to(self.device)
        w_channel = self.tr_params['w_channel']
        w_radius = self.tr_params['w_radius']
        if self.tr_params['w_mask_shape'] == 'circle':
            np_mask = self.circle_mask(x.shape[-1], w_radius)
            torch_mask = torch.tensor(np_mask).to(self.device)

            if w_channel == -1:
                # all channels
                watermarking_mask[:, :] = torch_mask
            else:
                watermarking_mask[:, w_channel] = torch_mask
        elif self.tr_params['w_mask_shape'] == 'square':
            anchor_p = x.shape[-1] // 2
            if w_channel == -1:
                # all channels
                watermarking_mask[:,
                                  :,
                                  (anchor_p - w_radius):(anchor_p + w_radius),
                                  (anchor_p - w_radius):(anchor_p + w_radius)
                                 ] = True
            else:
                watermarking_mask[:,
                                  w_channel,
                                  (anchor_p - w_radius):(anchor_p + w_radius),
                                  (anchor_p - w_radius):(anchor_p + w_radius)
                                 ] = True
        elif self.tr_params['w_mask_shape'] == 'no':
            pass
        else:
            raise NotImplementedError(f"w_mask_shape: {self.tr_params['w_mask_shape']}")
        
        return watermarking_mask

    def circle_mask(self, size, radius, x_offset=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
        x0 = y0 = size // 2
        x0 += x_offset
        y0 += y_offset
        y, x = np.ogrid[:size, :size]
        y = y[::-1]
        return ((x - x0)**2 + (y-y0)**2)<= radius**2

    def save_images(self):
        return