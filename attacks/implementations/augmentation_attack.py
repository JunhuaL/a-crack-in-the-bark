from attack import Attack
from distortions import *

class AugmentationAttack(Attack):
    def __init__(self, eps: float, alpha: float, n_steps: int, batch_size: int, augmentation: str):
        super().__init__(self, eps, alpha, n_steps, batch_size)
        self.augmentation = augmentation

    def setup(self):
        return

    def attack(self):
        return
    
    def save_images(self):
        return