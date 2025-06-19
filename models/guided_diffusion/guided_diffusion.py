from typing import Callable, Any
from .script_util import *
import torch

class OutputWrapper:
    def __init__(self, outputs):
        self.images = outputs

class GuidedDiffusionPipeline:
    def __init__(self, model_params: dict, num_images: int, device: str):
        self.model_params = model_and_diffusion_defaults()
        self.model_params.update(model_params)

        temp_params = dict([(key,self.model_params[key]) for key in model_and_diffusion_defaults().keys()])
        self.model, self.diffusion = create_model_and_diffusion(**temp_params)

        self.model.load_state_dict(torch.load(self.model_params['model_path']))
        self.model.to(device)

        if self.model_params['use_fp16']:
            self.model.convert_to_fp16()

        self.model.eval()
        self.shape = (num_images, 3, self.model_params['image_size'], self.model_params['image_size'])
        
        self.num_images = num_images
        self.device = device

    ### 
    # Duplicating Stable Diffusion API to allow same script structure.
    # A lot of the additonal variable will not be used however its 
    # useful for keeping the call structure the same.
    ###
    def __call__(
        self,
        prompt: str,
        height: int = None,
        width: int = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: str = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: torch.Generator = None,
        latents: torch.FloatTensor = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Callable[[int, int, torch.FloatTensor], None] = None,
        callback_steps: int = 1,
        watermarking_gamma: float = None,
        watermarking_delta: float = None,
        watermarking_mask: torch.BoolTensor = None,
    ):
        outputs = self.diffusion.ddim_sample_loop(
                    model=self.model,
                    shape=self.shape,
                    noise=latents,
                    model_kwargs=prompt,
                    device=self.device,
                    return_image=True,
                )
        outputs = OutputWrapper(outputs)
        return outputs
    
    def get_random_latents(self, height=None, width=None):
        return torch.randn(*self.shape, device=self.device)
    
    def forward_diffusion(
        self,
        latents: torch.FloatTensor = None,
        text_embeddings: Any = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
    ):
        outputs = self.diffusion.ddim_reverse_sample_loop(
                    model=self.model,
                    shape=self.shape,
                    image=latents,
                    model_kwargs=text_embeddings,
                    device=self.device
        )

        return outputs