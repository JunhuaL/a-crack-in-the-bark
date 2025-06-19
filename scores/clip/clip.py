import torch
from models.open_clip.model import CLIP
from models.open_clip.tokenizer import HFTokenizer
from torchvision.transforms import Compose

class CLIPScore:
    def __init__(
        self,
        model: CLIP,
        preprocessor: Compose,
        tokenizer: HFTokenizer,
        device: str 
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, images, prompt):
        with torch.no_grad():
            img_batch = [self.preprocessor(i).unsqueeze(0) for i in images]
            img_batch = torch.concatenate(img_batch).to(self.device)
            image_features = self.model.encode_image(img_batch)

            text = self.tokenizer([prompt]).to(self.device)
            text_features = self.model.encode_text(text)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            return (image_features @ text_features.T).mean(-1)