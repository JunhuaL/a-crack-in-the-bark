import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from models.surrogate_models import ResNet_18

def set_random_seed(seed):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)

def read_json(filename):
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)
    
def fft_transform(data: torch.Tensor):
    data = torch.fft.fftshift(torch.fft.fft2(data), dim=(-1,-2))
    return data

def setup_surrogate(args):
    ### Only two classes watermarked vs non-watermarked
    num_classes = 2
    data_type = None

    if args.w_channel is not None:
        model_in_dim = 1
    elif args.vae in ['none']:
        model_in_dim = 3
    elif args.vae in ['stabilityai/stable-diffusion-2-1-base','stabilityai/sdxl-vae']:
        model_in_dim = 4
    elif args.vae in ['ostris/vae-kl-f8-d16']:
        model_in_dim = 16
    else:
        raise f"Model type for {args.vae} not implemented."
    
    if args.apply_fft:
        data_type = torch.cfloat

    model = ResNet_18(model_in_dim, num_classes, dtype=data_type).to(args.device)
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)

    return model, criterion, optimizer

def load_surrogate(args):
    model,_,_ = setup_surrogate(args)
    model.load_state_dict(torch.load(args.model_save_path))
    model.to(args.device)
    model.eval()
    
    return model