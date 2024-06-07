import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os 
import cv2

def read_image(filepath, target_size=(160, 160)):
    # Read the image from the filepath
    img = cv2.imread(filepath)
    # Resize the image to the target size
    img = cv2.resize(img, target_size)
    # Convert image to RGB (OpenCV reads images in BGR format by default)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert the image to a NumPy array and normalize the pixel values
    img_np = np.array(img, dtype = np.float32) / 255
    # Rearrange the dimensions to (channels, height, width)
    img_np = np.transpose(img_np, (2,0,1))
    return img_np

def create_dataframe_with_image_data(directory):
    # Initialize empty lists to store image data and labels
    images = []
    labels = []
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".jpeg") or filename.endswith(".png"):
            filepath = os.path.join(directory, filename)
            image = read_image(filepath)
            images.append(image)
            if 'bacteria'  in filename:
                labels.append(1.0)
            elif 'virus'  in filename: 
                labels.append(0.0)
            else: 
                labels.append(2.0)
    # Create a pandas DataFrame from the lists
    df = pd.DataFrame({'image': images, 'label': labels})
    return df

def merge_dfs(df_1,df_2):
    merged_df = pd.concat([df_1, df_2], axis=0, ignore_index=True)
    shuffled_df = merged_df.sample(frac=1).reset_index(drop=True)
    return shuffled_df



class Data(Dataset):
    def __init__(self, X, y, transform=None):
        X = np.array([arr.astype(float) for arr in X.values], dtype=float)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)
        self.len = self.X.shape[0]
        self.transform = transform
       
    def __getitem__(self, index):
        image = self.X[index].numpy().transpose(1, 2, 0)  # Convert to HWC format
        if self.transform:
            image = self.transform(image)
        return image, self.y[index]
   
    def __len__(self):
        return self.len


class DataModule: 
    def __init__(self, X, y, transform=None):
        self.dataset = Data(X, y, transform=transform)

    def get_dataloader(self, batch_size, num_workers=0):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    def train_dataloader(self, batch_size):
        return self.get_dataloader(batch_size)

    def val_dataloader(self, batch_size):
        return self.get_dataloader(batch_size)




