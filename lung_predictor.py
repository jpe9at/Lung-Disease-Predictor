from CustomDataClass import DataModule, Data, merge_dfs, create_dataframe_with_image_data 
from Module import CNNLungs
import Trainer 
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import shap

import os
import cv2

import argparse
import json

######################################
#Set an option whether Hyperparameter 
#Optimization schould be included
######################################

parser = argparse.ArgumentParser(description="Description of your program")
parser.add_argument("--hyperparameter_optimization", help="Use Hyperparameter optimization", default=None)

args = parser.parse_args()
    
hyperparameter_optimization = args.hyperparameter_optimization
if hyperparameter_optimization == True:
  print("Hyperparameter optimization:", hyperparameter_optimization)


######################################
#Load the Data and create Data Class
######################################

#define the data augmentation

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(160, scale=(0.8, 1.0)),
    transforms.ToTensor()
])

val_test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])


print('Load training data')

home_directory = os.path.expanduser('~')

directory1 = home_directory + '/Datasets/chest_xray/train/NORMAL'
directory2 = home_directory + '/Datasets/chest_xray/train/PNEUMONIA'

df_train_neg = create_dataframe_with_image_data(directory1)
df_train_pos = create_dataframe_with_image_data(directory2)
df_train = merge_dfs(df_train_neg, df_train_pos)

print('Load validation data')

directory1 = home_directory + '/Datasets/chest_xray/val/NORMAL'
directory2 = home_directory + '/Datasets/chest_xray/val/PNEUMONIA'

df_val_neg = create_dataframe_with_image_data(directory1)
df_val_pos = create_dataframe_with_image_data(directory2)
df_val = merge_dfs(df_val_neg, df_val_pos)

print('Load test data')

directory1 = home_directory + '/Datasets/chest_xray/test/NORMAL'
directory2 = home_directory + '/Datasets/chest_xray/test/PNEUMONIA'

df_test_neg = create_dataframe_with_image_data(directory1)
df_test_pos = create_dataframe_with_image_data(directory2)
df_test = merge_dfs(df_test_neg, df_test_pos)


print('Creating Data Module')

X_train = df_train['image']
y_train = df_train['label']
data_train = DataModule(X_train, y_train, transform=train_transforms)

X_val = df_val['image']
y_val = df_val['label']
data_val = DataModule(X_val, y_val, transform=val_test_transforms)

X_test = df_test['image']
y_test = df_test['label']
data_test = DataModule(X_test, y_test, transform=val_test_transforms)


######################################
#Initialise and train CNN
######################################

if hyperparameter_optimization: 
    print('starting hyperparameter training loop')
    best_params, best_accuracy = Trainer.Trainer.hyperparameter_optimization(data_train, data_test)
  
    cnn_model= CNNLungs(best_params['hidden_size'], 3, optimizer = best_params['optimizer'], learning_rate = best_params['learning_rate'], l2 = best_params['l2_rate'],  param_initialisation = best_params['weight_initialisation']) 
    trainer = Trainer.Trainer(50, best_params['batch_size'], early_stopping_patience = 20)
    trainer.fit(cnn_model,data_train,data_val)

else:
    cnn_model= CNNLungs(32, 3, optimizer = 'Adam', learning_rate = 8.998e-05, param_initialisation = (None), scheduler = 'OnPlateau', l1 = 0.0, clip_val = 1.5, l2 = 0.1 ) 
    trainer = Trainer.Trainer(1, 8, early_stopping_patience = 10)
    trainer.fit(cnn_model,data_train,data_val)


######################################
#Visualize Results
######################################


test_loss = trainer.test(cnn_model,data_test)
print(f'Test Loss is {test_loss:.4f}')

n_epochs = range(trainer.max_epochs)
train_loss = trainer.train_loss_values
nan_values = np.full(trainer.max_epochs - len(train_loss), np.nan)
train_loss = np.concatenate([train_loss,nan_values])

val_loss = trainer.val_loss_values
nan_values = np.full(trainer.max_epochs - len(val_loss), np.nan)
val_loss = np.concatenate([val_loss,nan_values])

plt.figure()
plt.plot(n_epochs, train_loss, color='blue', label='train_loss' , linestyle='-')
plt.plot(n_epochs, val_loss, color='orange', label='val_loss' , linestyle='-')
plt.legend()


######################################
#Use the DeepSHAP explainer
######################################

examples = data_test.dataset.X[-70:-5]
test = data_test.dataset.X[-5:-2]

with torch.no_grad():
    predicted_probs = cnn_model(test)
    predicted_labels = predicted_probs.argmax(dim=1)

# Convert predicted_labels tensor to numpy array
predicted_labels = predicted_labels.numpy() 

correct_labels = data_test.dataset.y[-5:-2]

explainer = shap.DeepExplainer(cnn_model, examples)

# Compute SHAP values

shap_values = explainer.shap_values(test, check_additivity=False)

# Loop through the available SHAP values
for i in range(len(shap_values)):
    shap.image_plot(shap_values[i], test[i].numpy(), show=False)
    plt.title(f"Predicted: {predicted_labels[i]}, Correct: {correct_labels[i]}")

plt.show()


