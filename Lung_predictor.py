from CustomDataClass import DataModule, Data 
from Module import CNNLungs
import Trainer 
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

import shap

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
                labels.append(-1.0)
    # Create a pandas DataFrame from the lists
    df = pd.DataFrame({'image': images, 'label': labels})
    return df

def merge_dfs(df_1,df_2):
    merged_df = pd.concat([df_1, df_2], axis=0, ignore_index=True)
    shuffled_df = merged_df.sample(frac=1).reset_index(drop=True)
    return shuffled_df

home_directory = os.path.expanduser('~')

#Training Data
directory1 = home_directory + '/Datasets/chest_xray/train/NORMAL'
directory2 = home_directory + '/Datasets/chest_xray/train/PNEUMONIA'

print('load td')
df_train_neg = create_dataframe_with_image_data(directory2)
df_train_pos = create_dataframe_with_image_data(directory2)
df_train = merge_dfs(df_train_neg, df_train_pos)

#Validation Data
directory1 = home_directory + '/Datasets/chest_xray/val/NORMAL'
directory2 = home_directory + '/Datasets/chest_xray/val/PNEUMONIA'

print('load val data')
df_val_neg = create_dataframe_with_image_data(directory2)
df_val_pos = create_dataframe_with_image_data(directory2)
df_val = merge_dfs(df_val_neg, df_val_pos)

#Testing Data
directory1 = home_directory + '/Datasets/chest_xray/test/NORMAL'
directory2 = home_directory + '/Datasets/chest_xray/test/PNEUMONIA'

print('load testd')
df_test_neg = create_dataframe_with_image_data(directory2)
df_test_pos = create_dataframe_with_image_data(directory2)
df_test = merge_dfs(df_test_neg, df_test_pos)

print('data Module')
X_train =df_test.iloc[:,0]
y_train =df_test.iloc[:,1]
data_train = DataModule(X_train,y_train)

X_val =df_val.iloc[:,0]
y_val =df_val.iloc[:,1]
data_val = DataModule(X_val,y_val)

X_test =df_test.iloc[:,0]
y_test =df_test.iloc[:,1]
data_test = DataModule(X_test,y_test)

print('init cnn')
#cnn_model= CNNLungs(64, 3, learning_rate = 0.0001)
                     #, best_params['optimizer'], best_params['learning_rate'],  param_initialisation = best_params['weight_initialisation']) 
#trainer = Trainer.Trainer(50, 8, early_stopping_patience = 10)

print('start fitting')

'''
trainer.fit(cnn_model,data_train,data_val)
best_params, best_accuracy = Trainer.Trainer.hyperparameter_optimization(data_train, data_test)

cnn_model= CNNLungs(best_params['hidden_size'], 3, optimizer = best_params['optimizer'], learning_rate = best_params['learning_rate'], l2 = best_params  param_initialisation = best_params['weight_initialisation']) 
trainer = Trainer.Trainer(50, best_params['batch_size'], early_stopping_patience = 100)
trainer.fit(cnn_model,data_train,data_val)
'''

cnn_model= CNNLungs(32, 3, optimizer = 'Adam', learning_rate = 0.00007, param_initialisation = (None,'He'), scheduler = 'OnPlateau', l1 = 0.0, clip_val = 1.5, l2 = 0.1 ) 
trainer = Trainer.Trainer(100, early_stopping_patience = 10)
trainer.fit(cnn_model,data_train,data_val)

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

plt.tight_layout()
plt.show()


'''
y_hat, y = trainer.test(lstm_model,data_test)

y_0 = y[:,0]
y_1 = y[:,1]
y_2 = y[:,2]
y_hat_0 = y_hat[:,0]
y_hat_1 = y_hat[:,1]
y_hat_2 = y_hat[:,2]

x_values = range(len(y_0))


plt.figure()
plt.plot(x_values, y_1, color='blue', label='test_data' , linestyle='-')
plt.plot(x_values, y_hat_1, color='green', label='prediction' , linestyle='-')
plt.legend()

plt.figure()
plt.plot(x_values, y_2, color='red', label='test_data' , linestyle='-')
plt.plot(x_values, y_hat_2, color='green', label='prediction' , linestyle='-')
plt.legend()

plt.figure()
plt.plot(x_values, y_0, color='orange', label='test_data' , linestyle='-')
plt.plot(x_values, y_hat_0, color='green', label='prediction' , linestyle='-')
plt.legend()
plt.show()

'''
