from torch.utils.data import Dataset
import os
import pandas as pd

### Intended for csv files, specifically produced by filtering through 
class PromptDataset(Dataset):
    def __init__(self, dataset_pth):
        if not os.path.exists(dataset_pth):
            RuntimeError(f"file {dataset_pth} not found.")
        
        self.dataset = pd.read_csv(dataset_pth)

    def __getitem__(self, index):
        sample = self.dataset.iloc[index]['names']
        prompt = f'A photo of a {sample}'
        return prompt
    
    def __len__(self):
        return len(self.dataset)