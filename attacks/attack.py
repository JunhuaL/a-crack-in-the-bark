from torch import Tensor
from typing_utils import *
import os
import torch
import torchvision.transforms as transforms

class Attack:
    def __init__(self, eps: float, alpha: float, n_steps: int, batch_size: int):
        self.eps = eps
        self.alpha = alpha
        self.n_steps = n_steps
        self.batch_size = batch_size

    def setup(self):
        raise NotImplementedError()

    def attack(self) -> Tensor:
        raise NotImplementedError()
    
    def save_images(self, data: Tensor, filenames: list, out_dir: str) -> bool:
        print(f"Saving images to out dir: {out_dir}")
        for image, filename in zip(data, filenames):
            image = transforms.functional.to_pil_image((image * 255).clamp(0, 255).to(torch.uint8))
            image.save(os.path.join(out_dir, f'{filename}.png'))
        print("Images Saved!")