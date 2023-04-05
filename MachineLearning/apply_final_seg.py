import random
from pathlib import Path
import numpy as np
import torch
from matplotlib import pyplot as plt
import SimpleITK as sitk
import u_net
import utils
import os
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
# to ensure reproducible training/validation split
random.seed(42)

# directorys with data and to stored training checkpoints
DATA_DIR = Path.cwd() / "ValidatieData"

# this is my best epoch - what is yours?
BEST_EPOCH = 52  # without synthetic data
CHECKPOINTS_DIR = Path.cwd() / "segmentation_model_weights_synthetic" / f"u_net_{BEST_EPOCH}.pth"

# hyperparameters
IMAGE_SIZE = [64, 64]

# find patient folders in training directory
# excluding hidden folders (start with .)
patients = [
    path
    for path in DATA_DIR.glob("*")
    if not any(part.startswith(".") for part in path.parts)
]

# load validation data
valid_dataset = utils.ValidationMRDataset(patients, IMAGE_SIZE)

unet_model = u_net.UNet(num_classes=1)
unet_model.load_state_dict(torch.load(CHECKPOINTS_DIR))
unet_model.eval()

# set model to evaluation mode
unet_model.eval()


def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size - img.shape[0]
    delta_height = desired_size - img.shape[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def resize_with_padding(img, expected_size):
    # img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.shape[0]
    delta_height = expected_size[1] - img.shape[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


test_3Dimage_DIR = Path.cwd() / "ResultsML"
if not os.path.exists(test_3Dimage_DIR):
    os.mkdir(test_3Dimage_DIR)

with torch.no_grad():
    result_patients = []
    idx_slice = 0
    for i in range(0, np.shape(valid_dataset.mr_image_list)[0]):

        sNAME = 'p' + str(i+1)
        sNAME_DIR = os.path.join(test_3Dimage_DIR, sNAME)

        if not os.path.exists(sNAME_DIR):
            os.mkdir(sNAME_DIR)

        pred_3d = np.zeros((86, IMAGE_SIZE[0], IMAGE_SIZE[1]))
        mr_3d = np.zeros((86, IMAGE_SIZE[0], IMAGE_SIZE[1]))

        for pred_slice in range(0, 86):
            input_img = valid_dataset[idx_slice]
            output = torch.sigmoid(unet_model(input_img[np.newaxis, ...]))
            pred_mask = torch.round(output)

            # resize_func = transforms.Resize((333, 271), interpolation=InterpolationMode.NEAREST)
            # pred_resized = resize_func(pred_mask)

            pred_3d[pred_slice] = pred_mask[0, 0]
            mr_3d[pred_slice] = input_img[0]

            idx_slice += 1

            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(input_img[0], cmap="gray")
            # ax[0].set_title("Input")
            # ax[0].axis("off")
            #
            # ax[1].imshow(pred_mask[0, 0])
            # ax[1].set_title("Prediction")
            # ax[1].axis("off")
            # plt.show()

        pred_image_sitk = sitk.GetImageFromArray(pred_3d)
        sitk.WriteImage(pred_image_sitk, os.path.join(sNAME_DIR, f'prostaat.mhd'))

        input_image_sitk = sitk.GetImageFromArray(mr_3d)
        sitk.WriteImage(input_image_sitk, os.path.join(sNAME_DIR, 'mr_bffe.mhd'))
