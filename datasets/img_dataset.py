import os
import re
import torch
import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision.io import read_image

def file_key(f_name):
    return [int(x) for x in re.findall('\d+',f_name)]

class PairedDataset(Dataset):
    def __init__(self, img_dataset, wm_dataset):
        self.data1 = img_dataset
        self.data2 = wm_dataset

        if len(img_dataset) != len(wm_dataset):
            print(
                "Lengths of dataset1 and dataset 2 are not equal"
            )
            exit()
        
    def __getitem__(self, index):
        return self.data2[index][0], self.data1[index][0], self.data2[index][1]
    
    def __len__(self):
        return len(self.data1)

class WmLoader(Dataset):
    def __init__(self,data_dir,split=None):
        data = torch.load(data_dir)
        self.wm_mask = data[0]
        _,reals,imags = data[1]
        reals = reals.to(torch.cfloat)
        reals.imag = imags

        if split is not None:
            self.data = reals[split]
        else:
            self.data = reals

    def __getitem__(self, index):
        return self.wm_mask,self.data[index]
    
    def __len__(self):
        return len(self.data)

class ImgLoader(Dataset):
    def __init__(self, data_dir, mapping_table = None, key = None, split=None, device='cpu'):
        img_fps = np.array(os.listdir(data_dir))
        img_fps = img_fps[split] if split is not None else img_fps
        self.img_fps = img_fps
        self.data = [] 
        for img_fp in img_fps:
            img = read_image(os.path.join(data_dir,img_fp)).type(torch.float32)
            self.data.append(img)
        self.data = torch.stack(self.data) 
        self.key = key
        self.table = mapping_table
        self.device = device

    def __getitem__(self, index):
        if self.table is not None:
            img_fp = int(self.img_fps[index].split('.')[0])
            prompt = self.table.loc[self.table[self.key] == img_fp]['prompt'].values[0]
            if not isinstance(prompt, str):
                model_kwargs = {"y":torch.tensor(prompt, device=self.device).unsqueeze(0)}
                prompt = model_kwargs
            return self.data[index], self.img_fps[index], prompt
        else:
            return self.data[index], self.img_fps[index]
    
    def __len__(self):
        return len(self.data)
    
class LatentLoader(Dataset):
    def __init__(self,data_dir,split=None):
        latent_path = data_dir
        idxs, data = torch.load(latent_path)
        idxs = np.array(idxs)

        self.latent_max_val = data.abs().max().detach().cpu().item()
        self.data = data[split] if split is not None else data     
        self.idxs = idxs[split] if split is not None else idxs

    def __getitem__(self, index):
        return self.data[index], self.idxs[index]
    
    def __len__(self):
        return len(self.data)
    
class LatentDataset(Dataset):
    def __init__(self, data_dirs: str, clss: list, split: list = None):
        self.data = []
        self.labels = []

        for dir,cls in zip(data_dirs, clss):
            idxs,latents = torch.load(dir)
            labels = torch.tensor([cls]*len(latents))
            self.data.append(latents)
            self.labels.append(labels)
            
        self.data = torch.concat(self.data)
        self.data = self.data[split] if split is not None else self.data
        self.labels = torch.concat(self.labels)
        self.labels = self.labels[split] if split is not None else self.labels

    def __getitem__(self, index):
        return self.data[index],self.labels[index]

    def __len__(self):
        return len(self.labels)

class ImgDirDataset(Dataset):
    def __init__(self, data_dirs: str, clss: list, split: list = None):
        self.data = []
        self.labels = []

        dir_2_clss = dict(zip(data_dirs, clss))
        img_paths = []
        for dir in data_dirs:
            paths = [ os.path.join(dir, img_name) for img_name in os.listdir(dir)]
            img_paths.extend(paths)
        
        split_imgs = np.array(img_paths)
        split_imgs = split_imgs[split] if split is not None else split_imgs

        for img_path in split_imgs:
            img = read_image(img_path).type(torch.float32)
            self.data.append(img)
            for key in dir_2_clss:
                if img_path.startswith(key):
                    cls = dir_2_clss[key]
            self.labels.append(cls)

        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels)

    def __getitem__(self, index):
        return self.data[index],self.labels[index]
    
    def __len__(self):
        return len(self.labels)
