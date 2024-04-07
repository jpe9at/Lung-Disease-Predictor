import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class Data(Dataset):
    def __init__(self, X, y):
        #needed to transform input tensor to float.
        X = np.array([arr.astype(float) for arr in X.values], dtype=float)
        self.X = torch.tensor(X, dtype = torch.float32)
        #prediction
        self.y = torch.tensor(y, dtype = torch.long)
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len

class DataModule: 
    def __init__(self, X,y):
        self.dataset = Data(X,y)

    def get_dataloader(self, batch_size, num_workers=4):
        return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=None, pin_memory=False)

    def train_dataloader(self, batch_size):
        return self.get_dataloader(batch_size)

    def val_dataloader(self, batch_size):
        return self.get_dataloader(batch_size)




