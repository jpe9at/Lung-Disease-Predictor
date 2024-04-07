from CustomDataClass import DataModule, Data, merge_dfs, create_dataframe_with_image_data 
from Module import CNNLungs
import Trainer 
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

import shap

import os
import cv2



######################################
#Load the Data and create Data Class
######################################

print('Load training data')

home_directory = os.path.expanduser('~')

directory1 = home_directory + '/Datasets/chest_xray/train/NORMAL'
directory2 = home_directory + '/Datasets/chest_xray/train/PNEUMONIA'

df_train_neg = create_dataframe_with_image_data(directory2)
df_train_pos = create_dataframe_with_image_data(directory2)
df_train = merge_dfs(df_train_neg, df_train_pos)

print('Load validation data')

directory1 = home_directory + '/Datasets/chest_xray/val/NORMAL'
directory2 = home_directory + '/Datasets/chest_xray/val/PNEUMONIA'

df_val_neg = create_dataframe_with_image_data(directory2)
df_val_pos = create_dataframe_with_image_data(directory2)
df_val = merge_dfs(df_val_neg, df_val_pos)

print('Load test data')

directory1 = home_directory + '/Datasets/chest_xray/test/NORMAL'
directory2 = home_directory + '/Datasets/chest_xray/test/PNEUMONIA'

df_test_neg = create_dataframe_with_image_data(directory2)
df_test_pos = create_dataframe_with_image_data(directory2)
df_test = merge_dfs(df_test_neg, df_test_pos)


print('Creating Data Module')

X_train =df_test.iloc[:,0]
y_train =df_test.iloc[:,1]
data_train = DataModule(X_train,y_train)

X_val =df_val.iloc[:,0]
y_val =df_val.iloc[:,1]
data_val = DataModule(X_val,y_val)

X_test =df_test.iloc[:,0]
y_test =df_test.iloc[:,1]
data_test = DataModule(X_test,y_test)


######################################
#Initialise and train CNN
######################################

if hyperparameter_optimization == True: 
  
  trainer.fit(cnn_model,data_train,data_val)
  best_params, best_accuracy = Trainer.Trainer.hyperparameter_optimization(data_train, data_test)
  
  cnn_model= CNNLungs(best_params['hidden_size'], 3, optimizer = best_params['optimizer'], learning_rate = best_params['learning_rate'], l2 = best_params  param_initialisation = best_params['weight_initialisation']) 
  trainer = Trainer.Trainer(50, best_params['batch_size'], early_stopping_patience = 100)
  trainer.fit(cnn_model,data_train,data_val)

else: 
  cnn_model= CNNLungs(32, 3, optimizer = 'Adam', learning_rate = 0.00007, param_initialisation = (None,'He'), scheduler = 'OnPlateau', l1 = 0.0, clip_val = 1.5, l2 = 0.1 ) 
  trainer = Trainer.Trainer(100, early_stopping_patience = 10)
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

examples = data_test.dataset.X[:-3]
test = data_test.dataset.X[-3:]

explainer = shap.DeepExplainer(cnn_model, examples)

# Compute SHAP values

shap_values = explainer.shap_values(test)

# Create a single figure to display all images with overlaid SHAP values
fig, axs = plt.subplots(nrows=7, ncols=1, figsize=(8, 2*7))

# Visualize SHAP values overlaid on images
for i in range(7):
    shap.image_plot(shap_values[i], examples[i].numpy(), show=False, ax=axs[i])

plt.figure()
plt.tight_layout()
plt.show()


