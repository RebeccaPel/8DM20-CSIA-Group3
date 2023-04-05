import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import PIL


class ProstateMRDataset(torch.utils.data.Dataset):
    """Dataset containing prostate MR images.
    Parameters
    ----------
    paths : list[Path]
        paths to the patient data
    img_size : list[int]
        size of images to be interpolated to
    """

    def __init__(self, paths, img_size, augment=False):
        self.mr_image_list = []
        self.mask_list = []
        self.augment = augment
        # load images
        for path in paths:
            self.mr_image_list.append(
                sitk.GetArrayFromImage(sitk.ReadImage(path / "mr_bffe.mhd"))
            )
            self.mask_list.append(
                sitk.GetArrayFromImage(sitk.ReadImage(path / "prostaat.mhd"))
            )

        # number of patients and slices in the dataset
        self.no_patients = len(self.mr_image_list)
        self.no_slices = self.mr_image_list[0].shape[0]

        # transforms to resize images
        self.img_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.CenterCrop(256),
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]
        )

        aug_list = [transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                    # size of crop is between 0.7 and 1% of the original size
                    transforms.RandomRotation(10),
                    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1))]  # blurs the image

        self.img_augment = transforms.RandomChoice(aug_list, p=[1] * len(aug_list))

        # standardise intensities based on mean and std deviation
        self.train_data_mean = np.mean(self.mr_image_list)
        self.train_data_std = np.std(self.mr_image_list)
        self.norm_transform = transforms.Normalize(
            self.train_data_mean, self.train_data_std
        )

    def __len__(self):
        """Returns length of dataset"""
        return self.no_patients * self.no_slices

    def __getitem__(self, index):
        """Returns the preprocessing MR image and corresponding segementation
        for a given index.
        Parameters
        ----------
        index : int
            index of the image/segmentation in dataset
        """

        # compute which slice an index corresponds to
        patient = index // self.no_slices
        the_slice = index - (patient * self.no_slices)

        if self.augment:
            return (self.norm_transform(
                self.img_augment(
                    self.img_transform(
                        self.mr_image_list[patient][the_slice, ...].astype(np.float32)
                    )
                ), ),
                    self.img_augment(
                        self.img_transform(
                            (self.mask_list[patient][the_slice, ...] > 0).astype(np.int32)
                        ), )
            )

        else:
            return (
                self.norm_transform(
                    self.img_transform(
                        self.mr_image_list[patient][the_slice, ...].astype(np.float32)
                    )
                ),
                self.img_transform(
                    (self.mask_list[patient][the_slice, ...] > 0).astype(np.int32)
                ),
            )


class SyntheticDataset(torch.utils.data.Dataset):
    """Dataset containing synthetic prostate MR images.
    Parameters
    ----------
    paths : list[Path]
        paths to the patient data
    img_size : list[int]
        size of images to be interpolated to
    """

    def __init__(self, paths, img_size):
        self.mr_image_list = []
        self.mask_list = []
        self.img_size = img_size
        # load images
        for path in paths:
            self.mr_image_list.append(
                sitk.GetArrayFromImage(sitk.ReadImage(path / "mr_bffe.mhd"))
            )
            self.mask_list.append(
                sitk.GetArrayFromImage(sitk.ReadImage(path / "prostaat.mhd"))
            )

        # number of patients and slices in the dataset
        self.no_patients = len(self.mr_image_list)
        self.no_slices = self.mr_image_list[0].shape[0]

        # transforms to tensor images
        self.img_transform = transforms.Compose(
            [
                # transforms.ToPILImage(),
                # transforms.Resize(img_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        """Returns length of dataset"""
        return self.no_patients * self.no_slices

    def __getitem__(self, index):
        """Returns the preprocessing MR image and corresponding segementation
        for a given index.
        Parameters
        ----------
        index : int
            index of the image/segmentation in dataset
        """

        # compute which slice an index corresponds to
        patient = index // self.no_slices
        the_slice = index - (patient * self.no_slices)

        # print(np.shape(self.mr_image_list[patient]))
        # print(self.mr_image_list[patient].astype(np.uint8))
        # print(np.unique(self.mr_image_list[patient]))

        return (
            self.img_transform(
                np.reshape(self.mr_image_list[patient][the_slice, ...].astype(np.int32), self.img_size)
            ),
            self.img_transform(
                np.reshape((self.mask_list[patient][the_slice, ...] > 0).astype(np.int32), self.img_size)
            ),
        )


class ValidationMRDataset(torch.utils.data.Dataset):
    """Dataset containing prostate MR images.
    Parameters
    ----------
    paths : list[Path]
        paths to the patient data
    img_size : list[int]
        size of images to be interpolated to
    """

    def __init__(self, paths, img_size):
        self.mr_image_list = []
        # load images
        for path in paths:
            self.mr_image_list.append(
                sitk.GetArrayFromImage(sitk.ReadImage(path / "mr_bffe.mhd"))
            )

        # number of patients and slices in the dataset
        self.no_patients = len(self.mr_image_list)
        self.no_slices = self.mr_image_list[0].shape[0]

        # transforms to resize images
        self.img_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.CenterCrop(256),
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]
        )

        # standardise intensities based on mean and std deviation
        self.val_data_mean = np.mean(self.mr_image_list)
        self.val_data_std = np.std(self.mr_image_list)
        self.norm_transform = transforms.Normalize(
            self.val_data_mean, self.val_data_std
        )

    def __len__(self):
        """Returns length of dataset"""
        return self.no_patients * self.no_slices

    def __getitem__(self, index):
        """Returns the preprocessing MR image and corresponding segementation
        for a given index.
        Parameters
        ----------
        index : int
            index of the image/segmentation in dataset
        """

        # compute which slice an index corresponds to
        patient = index // self.no_slices
        the_slice = index - (patient * self.no_slices)

        return (
            self.norm_transform(
                self.img_transform(
                    self.mr_image_list[patient][the_slice, ...].astype(np.float32)
                )
            )
        )


class DiceBCELoss(nn.Module):
    """Loss function, computed as the sum of Dice score and binary cross-entropy.
    Notes
    -----
    This loss assumes that the inputs are logits (i.e., the outputs of a linear layer),
    and that the targets are integer values that represent the correct class labels.
    """

    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, outputs, targets, smooth=1):
        """Calculates segmentation loss for training
        Parameters
        ----------
        outputs : torch.Tensor
            predictions of segmentation model
        targets : torch.Tensor
            ground-truth labels
        smooth : float
            smooth parameter for dice score avoids division by zero, by default 1
        Returns
        -------
        float
            the sum of the dice loss and binary cross-entropy
        """
        outputs = torch.sigmoid(outputs)

        # flatten label and prediction tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # compute Dice
        intersection = (outputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
                outputs.sum() + targets.sum() + smooth
        )
        BCE = nn.functional.binary_cross_entropy(outputs, targets, reduction="mean")

        return BCE + dice_loss