import random
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import u_net
import utils

from seg_metrics import seg_metrics as sg
# import seg_metrics.seg_metrics as sg

# to ensure reproducible training/validation split
random.seed(42)

# directorys with data and to stored training checkpoints
DATA_DIR = Path.cwd() / "TrainingData"

# this is my best epoch - what is yours?
BEST_EPOCH = 46 #without synthetic data
# BEST_EPOCH = 199 #with synthetic data
CHECKPOINTS_DIR = Path.cwd() / "Trained_models" / "Unet_001" / "segmentation_model_weights_001" / f"u_net_{BEST_EPOCH}.pth" #without synthetic data
# CHECKPOINTS_DIR = Path.cwd() / "segmentation_model_weights_synthetic_large" / f"u_net_{BEST_EPOCH}.pth" #with synthetic data
# CHECKPOINTS_DIR = Path.cwd() / "segmentation_model_weights_lr5" / f"u_net_{BEST_EPOCH}.pth" #with augmented data, no synthetic data

# hyperparameters
NO_VALIDATION_PATIENTS = 2
IMAGE_SIZE = [64, 64]

# find patient folders in training directory
# excluding hidden folders (start with .)
patients = [
    path
    for path in DATA_DIR.glob("*")
    if not any(part.startswith(".") for part in path.parts)
]
random.shuffle(patients)

# split in training/validation after shuffling
partition = {
    "train": patients[:-NO_VALIDATION_PATIENTS],
    "validation": patients[-NO_VALIDATION_PATIENTS:],
}

# load validation data
valid_dataset = utils.ProstateMRDataset(partition["validation"], IMAGE_SIZE)

unet_model = u_net.UNet(num_classes=1)
unet_model.load_state_dict(torch.load(CHECKPOINTS_DIR))
unet_model.eval()

#%% Calculate performance
def scores(end_index, csv_path, plot_images = False):
    labels = [0,1]
    csv_file = csv_path
    
    for predict_index in range(end_index):
        (input, target) = valid_dataset[predict_index]
        output = torch.sigmoid(unet_model(input[np.newaxis, ...]))
        gdth = target[0].numpy()
        pred = torch.round(output).detach().numpy()[0,0]
        # print(np.shape(gdth))
        print((pred))
    
        metrics = sg.write_metrics(labels=labels[1:],  # exclude background
                  gdth_img=gdth,
                  pred_img=pred,
                  csv_file=csv_file,
                  metrics=['dice','jaccard'])
        print(metrics)
        
        if plot_images:
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(input[0], cmap="gray")
            ax[0].set_title("Input")
            ax[0].axis("off")

            ax[1].imshow(target[0])
            ax[1].set_title("Ground-truth")
            ax[1].axis("off")

            ax[2].imshow(pred[0, 0].detach().numpy())
            ax[2].set_title("Prediction")
            ax[2].axis("off")
            fig.suptitle('Slice {}'.format(predict_index))
            plt.show()
    return metrics
 
nr_slices_per_patient = 86
N_slices_validation = NO_VALIDATION_PATIENTS * nr_slices_per_patient - 1 #-1 because python starts indexing from 0

metrics = scores(N_slices_validation, 'metrics.csv', plot_images=False) #SPECIFY different path for each model


#%% Calculate performance

from sklearn.metrics import confusion_matrix
import pandas as pd
def dice_scores(end_index, start_index=0, plot_images = False):
    cols = ['Index','Sensitivity','Specificity','Accuracy','Dice_score','Jaccard']
    df_results = pd.DataFrame(columns=cols)
    for predict_index in range(end_index):
        (input, target) = valid_dataset[predict_index]
        output = torch.sigmoid(unet_model(input[np.newaxis, ...]))
        ground_truth = target[0]
        prediction = torch.round(output)
        
        mcm = confusion_matrix(np.ndarray.flatten(ground_truth.detach().numpy()),np.ndarray.flatten(prediction.detach().numpy()),labels=[False, True])
        tn = mcm[0, 0]
        fp = mcm[0, 1]
        fn = mcm[1, 0]
        tp = mcm[1, 1]
        
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        accuracy = (tp + tn) / (tp + tn + fn + fp)
        dice_score = 2*tp / (2*tp + fp + fn)
        Jaccard = dice_score / (2-dice_score)
        
        df_results = df_results.append(pd.Series([predict_index, sensitivity, specificity, accuracy, dice_score, Jaccard],index=df_results.columns), ignore_index=True)
        
        if plot_images:
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(input[0], cmap="gray")
            ax[0].set_title("Input")
            ax[0].axis("off")

            ax[1].imshow(target[0])
            ax[1].set_title("Ground-truth")
            ax[1].axis("off")

            ax[2].imshow(prediction[0, 0].detach().numpy())
            ax[2].set_title("Prediction")
            ax[2].axis("off")
            fig.suptitle('Slice {}'.format(predict_index))
            plt.show()
            
    return df_results

nr_slices_per_patient = 86
N_slices_validation = NO_VALIDATION_PATIENTS * nr_slices_per_patient - 1 #-1 because python starts indexing from 0

df_results = dice_scores(N_slices_validation, plot_images=False)
#df_results = dice_scores(N_slices_validation) plot_images=True will plot the image for every slice!!!

mean_sensitivity = df_results['Sensitivity'].mean()
mean_specificity = df_results['Specificity'].mean()
mean_accuracy = df_results['Accuracy'].mean()
mean_Dice = df_results['Dice_score'].mean()
mean_IoU = df_results['Jaccard'].mean()
