import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class Data(Dataset):
    def __init__(self, X, y):
        X = np.array([arr.astype(float) for arr in X.values], dtype=float)
        self.X = torch.tensor(X, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = torch.long)
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len

'''
#How to deal with other types of data? 
'''

class DataModule: 
    """The base class of data."""
    def __init__(self, X,y):
        self.dataset = Data(X,y)

    def get_dataloader(self, batch_size, num_workers=4):
        return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=None, pin_memory=False)

    def train_dataloader(self, batch_size):
        return self.get_dataloader(batch_size)

    def val_dataloader(self, batch_size):
        #here say something about gradients not required ? 
        return self.get_dataloader(batch_size)

#data = self.train if train else self.val
#    return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,num_workers=self.num_workers)

#
#dataframe = pd.read_csv('~/Datasets/Deli Climate/DailyDelhiClimateTrain.csv')
#time_series_data = dataset_for_time_series(dataframe.iloc[:,1:],4) 
#data = DataModule(time_series_data[0],time_series_data[1])


#num of workers for allowing of multi processing
    #num_workers=0 means that it’s the main process that does the data loading when needed.
    #num_workers=1 means you only have a single worker, so it might be slow.


device = "cuda" if torch.cuda.is_available() else "cpu"
kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}



