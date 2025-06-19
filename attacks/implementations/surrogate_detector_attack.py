from attacks.attack import Attack
from diffusers import AutoencoderKL
from models.surrogate_models.resnet import ResNet_18
from torch import dtype
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

class SurrogateDetectorAttack(Attack):
    def __init__(self, surrogate: ResNet_18, 
                 eps: float, alpha: float, n_steps: int, batch_size: int, init_steps: int,
                 apply_fft: bool, w_channel: int, target_label: int, device: str, vae: AutoencoderKL = None
        ):
        super().__init__(eps, alpha, n_steps, batch_size)
        self.init_steps = init_steps
        self.model = surrogate
        self.apply_fft = apply_fft
        self.w_channel = w_channel
        self.target_label = target_label
        self.vae = vae
        self.device = device
        self.cost_func = CrossEntropyLoss().to(device)
        self.init_delta = None
        
    def setup(self, dataset: DataLoader):
        if( self.init_steps > 0 ):
            deltas = []
            for i, (data, fps) in enumerate(dataset):
                data = data.to(self.device)
                target_labels = torch.ones(data.size(0), dtype=torch.long).to(self.device)

                adv_data = self.get_adv_data(data, target_labels, None)

                deltas.append((adv_data - data).mean(dim=0))

                if i >= self.init_steps:
                    break
            self.init_delta = torch.stack(deltas).mean(dim=0)
        else:
            self.init_delta = None

    def attack(self, dataset: DataLoader):
        data_out = []
        filenames = []
        for (data, fps) in tqdm(dataset):
            data = data.to(self.device)
            target_labels = torch.tensor([self.target_label]*len(data), dtype=torch.long).to(self.device)

            adv_data = self.get_adv_data(data, target_labels, self.init_delta)
            
            if self.vae:
                adv_data = self.decode_image(adv_data)
                fps = [f'{fp}.png' for fp in fps]
            data_out.append(adv_data)
            filenames += fps
        data_out = torch.concat(data_out)
        return data_out, filenames
    
    def get_adv_data(self, data, target_labels, init_delta = None):
        self.model.eval()
        init_data = data.clone().detach().to(self.device)

        if init_delta is None:
            adv_data = init_data + torch.empty_like(init_data).uniform_(-self.eps, self.eps)
        else:
            adv_data = init_data + torch.clamp(init_delta, -self.eps, self.eps)

        for step in range(self.n_steps):
            self.model.zero_grad()
            adv_data.requires_grad = True

            if self.apply_fft:
                adv_data = torch.fft.fftshift(torch.fft.fft2(adv_data),dim=(-1,-2))

            pred_labels = self.model(adv_data)

            if self.apply_fft:
                pred_labels = pred_labels.real

            error = self.cost_func(pred_labels, target_labels)

            grad = torch.autograd.grad(
                error, adv_data, retain_graph=False, create_graph=False
                )[0]
            
            pgd_grad = self.alpha * grad.sgn()
            adv_data = adv_data.detach() - pgd_grad

            if self.apply_fft:
                adv_data = torch.fft.ifft2(torch.fft.ifftshift(adv_data),dim=(-1,-2)).real

            adv_data = adv_data.detach()

        return adv_data

    @torch.inference_mode()
    def decode_image(self, data):
        scaled_data = 1 / 0.18215 * data
        images = [
            self.vae.decode(scaled_data[i: i + 1]).sample for i in range(len(data))
        ]
        images = torch.cat(images, dim=0).clamp(0,1)
        images = (images * 255)
        return images