from datasets import ImgDirDataset, LatentDataset
from models.surrogate_models import ResNet_18
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import fft_transform, setup_surrogate

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

def setup_dataset(args):
    if args.mode == 'rawpix':
        total_data = len(os.listdir(args.wm_img_folder) + os.listdir(args.unwm_img_folder))
    else:
        wm_data_count = len(torch.load(args.wm_img_folder)[0])
        unwm_data_count = len(torch.load(args.unwm_img_folder)[0])
        total_data = wm_data_count + unwm_data_count
    
    idxs = list(range(total_data))
    train_idxs, test_idxs = train_test_split(idxs, test_size=0.3, random_state=args.seed)

    if args.mode == 'rawpix':
        train_split = ImgDirDataset([args.wm_img_folder,args.unwm_img_folder],[0,1], train_idxs)
        test_split = ImgDirDataset([args.wm_img_folder,args.unwm_img_folder],[0,1], test_idxs)
    elif args.mode == 'latent':
        train_split = LatentDataset([args.wm_img_folder,args.unwm_img_folder],[0,1], train_idxs)
        test_split = LatentDataset([args.wm_img_folder,args.unwm_img_folder],[0,1], test_idxs)
    elif args.mode == 't49latent':
        train_split = LatentDataset([args.wm_img_folder,args.unwm_img_folder],[0,1], train_idxs)
        test_split = LatentDataset([args.wm_img_folder,args.unwm_img_folder],[0,1], test_idxs)
    else:
        raise NotImplementedError(f"using unknwon data representation: {args.mode}")
    
    train_data_len = len(train_split)
    test_data_len = len(test_split)

    train_loader = DataLoader(train_split, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_split, batch_size=args.batch_size, shuffle=True)

    return train_loader, test_loader, train_data_len, test_data_len

### Add transforms in the order you want to apply them in
def setup_transforms(args):
    transforms = []
    if args.apply_fft:
        transforms.append(fft_transform)
    return transforms

def inference_loop(model: nn.Module, data: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, train: bool, device: str, data_transforms: list = []):
    total_loss = 0
    correct = 0
    for imgs, labels in tqdm(data):
        imgs = imgs.to(device)
        
        if args.w_channel is not None:
            imgs = imgs[:, args.w_channel, :, :].unsqueeze(1)

        for transform in data_transforms:
            imgs = transform(imgs)

        labels = labels.long().to(device)
        preds = model(imgs).real
        loss = criterion(preds, labels)
        if train:
            model.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pred_labels = torch.max(preds, 1)[1]
        correct += (pred_labels==labels).sum().item()

    return total_loss, correct

def main(args):
    print("Loading dataset...")
    train_data, test_data, train_len, test_len = setup_dataset(args)
    print("Surrogate model setup...")
    model, criterion, optimizer = setup_surrogate(args)
    transforms = setup_transforms(args)
    
    best_test_acc = 0.0
    best_model_state = None
    print("Training stated!")
    for epoch in range(args.epochs):
        model.train()
        total_loss, correct = inference_loop(model, train_data, criterion, optimizer, True, args.device, transforms)
        train_loss = total_loss / train_len
        train_acc = correct / train_len

        model.eval()
        ### precautionary context manager
        with torch.no_grad():
            total_loss, correct = inference_loop(model, test_data, criterion, optimizer, False, args.device, transforms)
            test_loss = total_loss / test_len
            test_acc = correct / test_len
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict().copy()
            if args.v:
                print(
                    f"New best model found at epoch {epoch + 1} with test accuracy: {test_acc:.4f}%"
                )
        
        if args.v:
            print(
                f"Epoch [{epoch+1}/{args.epochs}], Trn Loss: {train_loss:.4f}, Trn Acc: {train_acc:.4f}, Tst Loss: {test_loss:.4f}, Tst Acc: {test_acc:.4f}"
            )
    
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path, exist_ok=True)
    model_pth = os.path.join(args.model_save_path, args.model_save_name + ".pth")
    if best_model_state is not None:
        torch.save(best_model_state, model_pth)
    else:
        torch.save(model.state_dict(), model_pth)
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Adversarial surrogate")
    parser.add_argument('wm_img_folder', type=str)
    parser.add_argument('unwm_img_folder', type=str)
    parser.add_argument('model_save_path', type=str)
    parser.add_argument('model_save_name', type=str)

    parser.add_argument('--mode', default = 'rawpix', choices = ['rawpix','latent','t49latent'], type=str)
    parser.add_argument('--vae', default = 'stabilityai/stable-diffusion-2-1-base',
                        choices = ['none','stabilityai/stable-diffusion-2-1-base','stabilityai/sdxl-vae','ostris/vae-kl-f8-d16'],
                        type = str
    )

    parser.add_argument('--apply_fft', action='store_true')
    parser.add_argument('--w_channel', default=None, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--seed', default=999999, type=int)

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--v', action="store_true")
    args = parser.parse_args()
    main(args)
