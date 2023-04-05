import random
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import utils
import vae
import u_net
import os
import SimpleITK as sitk

# to ensure reproducible training/validation split
random.seed(42)

# directorys with data and to stored training checkpoints
DATA_DIR = Path.cwd() / "TrainingData"

# this is my best epoch - what is yours?
BEST_EPOCH_VAE = 98
CHECKPOINTS_DIR_VAE = Path.cwd() / "vae_model_weights" / f"vae_{BEST_EPOCH_VAE}.pth"
BEST_EPOCH_UNET = 46
CHECKPOINTS_DIR_UNET = Path.cwd() / "segmentation_model_weights" / f"u_net_{BEST_EPOCH_UNET}.pth"

# dimension of VAE latent space
Z_DIM = 256

# %% Generate synthetic data
"""
As the input to the decoder follows a known distribution (i.e. Gaussian), we 
can sample from a Gaussian distribution and pass the values to the decoder to 
obtain new synthetic data.
"""
vae_model = vae.VAE()
vae_model.load_state_dict(torch.load(CHECKPOINTS_DIR_VAE))
vae_model.eval()

n_samples = 400  # define how many new data
noise_vector = vae.get_noise(n_samples, Z_DIM)
pred_mr = vae_model.generator(noise_vector)
synth_img = vae_model.head(pred_mr)
img_array = synth_img.detach().numpy()

# plot one image to see outcome
plt.imshow(img_array[0, 0, :, :], cmap="gray")
plt.show()

# %% Generate masks for synthetic data using U-net

# Obtain segmentation masks for synthetic data
unet_model = u_net.UNet(num_classes=1)
unet_model.load_state_dict(torch.load(CHECKPOINTS_DIR_UNET))
unet_model.eval()

synthetic_image_DIR = Path.cwd() / "SyntheticData"
os.mkdir(synthetic_image_DIR)

# save segmentation masks in list for later training
augmented_data = []
with torch.no_grad():
    for i in range(0, np.shape(synth_img)[0]):
        input_img = synth_img[i]
        output = torch.sigmoid(unet_model(input_img[np.newaxis, ...]))
        pred_mask = torch.round(output)
        augmented_data.append([input_img, pred_mask])

        sNAME = 's' + str(i)
        sNAME_DIR = os.path.join(synthetic_image_DIR, sNAME)
        os.mkdir(sNAME_DIR)
        # sNAME_DIR.mkdir(parents=True, exist_ok=True)
        # print(sNAME_DIR)

        input_image_sitk = sitk.GetImageFromArray(input_img)
        pred_image_sitk = sitk.GetImageFromArray(pred_mask)
        sitk.WriteImage(input_image_sitk,
                        os.path.join(sNAME_DIR, 'mr_bffe.mhd'))  # save synthetic MRI in the same way as real data
        sitk.WriteImage(pred_image_sitk, os.path.join(sNAME_DIR,
                                                      'prostaat.mhd'))  # save predictions of mask in the same way as real masks

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(input_img[0], cmap="gray")
        ax[0].set_title("Synthetic image")
        ax[0].axis("off")

        ax[1].imshow(pred_mask[0, 0])
        ax[1].set_title("Prediction mask")
        ax[1].axis("off")
        plt.show()