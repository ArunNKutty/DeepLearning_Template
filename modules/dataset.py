"""This file contains functions to download and transform the CIFAR10 dataset"""
# Needed for image transformations
import albumentations as A

# # Needed for padding issues in albumentations
# import cv2
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
from torchvision import datasets

# Use precomputed values for mean and standard deviation of the dataset
CIFAR_MEAN = (0.4915, 0.4823, 0.4468)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CUTOUT_SIZE = 16

# Create class labels and convert to tuple
CIFAR_CLASSES = tuple(
    c.capitalize()
    for c in [
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
)



class CIFAR10Transforms(Dataset):
    """Apply albumentations augmentations to CIFAR10 dataset"""

    # Given a dataset and transformations,
    # apply the transformations and return the dataset
    def __init__(self, dataset, transforms):
        self.transforms = transforms
        self.dataset = dataset

    def __getitem__(self, idx):
        # Get the image and label from the dataset
        image, label = self.dataset[idx]

        # Apply transformations on the image
        image = self.transforms(image=np.array(image))["image"]

        return image, label

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return (
            f"CIFAR10Transforms(dataset={self.dataset}, transforms={self.transforms})"
        )

    def __str__(self):
        return (
            f"CIFAR10Transforms(dataset={self.dataset}, transforms={self.transforms})"
        )


def apply_cifar_image_transformations(mean,std,cutout_size):
  """ Function to apply image transformations to the dataset """
  train_transforms = A.Compose(
      A.Compose(
        [
           
            A.PadIfNeeded(4),
            #Random Crop 32,32
            A.RandomCrop(32, 32),
            # FlipLR
            A.HorizontalFlip(),
            # A.Cutout(num_holes=1, max_h_size=cutout_size, max_w_size=cutout_size, fill_value=list(mean), always_apply=True, p=0.50),
            # A.ShiftScaleRotate(
            #     shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.5
            # ),
            A.CoarseDropout(
                max_holes=1,
                max_height=cutout_size,
                max_width=cutout_size,
                min_holes=1,
                min_height=cutout_size,
                min_width=cutout_size,
                fill_value=list(mean),
                p=1.0,
                mask_fill_value=None,
            ),
            A.Normalize(mean=list(mean), std=list(std)),
            ToTensorV2(),
        ]
    ) 
  )

  # Test data transformations
  test_transforms = A.Compose(
        [
            A.Normalize(mean=list(mean), std=list(std)),
            ToTensorV2(),
        ]
    )
  
  return train_transforms,test_transforms


def split_cifar_data(data_path, train_transforms, test_transforms):
    """
    Split the data to train and test
    """
    print("Downloading CIFAR10 dataset\n")
    # Download MNIST dataset
    train_data = datasets.CIFAR10(data_path, train=True, download=True)
    test_data = datasets.CIFAR10(data_path, train=False, download=True)

    # Calculate and print the mean and standard deviation of the dataset
    mean, std = calculate_mean_std(train_data)
    print(f"\nMean: {mean}")
    print(f"Std: {std}\n")

    # Apply transforms on the dataset
    # Use the above class to apply transforms on the dataset using albumentations
    train_data = CIFAR10Transforms(train_data, train_transforms)
    test_data = CIFAR10Transforms(test_data, test_transforms)

    print("Transforms applied on the dataset\n")

    return train_data, test_data


def calculate_mean_std(dataset):
    """Function to calculate the mean and standard deviation of CIFAR dataset"""
    data = dataset.data.astype(np.float32) / 255.0
    mean = np.mean(data, axis=(0, 1, 2))
    std = np.std(data, axis=(0, 1, 2))
    return mean, std


def get_cifar_dataloaders(data_path, batch_size, num_workers, seed):
    """Get the final train and test data loaders"""

    ## Data Transformations
    train_transforms, test_transforms = apply_cifar_image_transformations(
        mean=CIFAR_MEAN, std=CIFAR_STD, cutout_size=CUTOUT_SIZE
    )

    # print(f"Train and test data path: {data_path}")

    # Train and Test data
    # print("Splitting the dataset into train and test\n")
    train_data, test_data = split_cifar_data(
        data_path, train_transforms, test_transforms
    )

    # To be passed to dataloader
    def _init_fn(worker_id):
        np.random.seed(int(seed))

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_init_fn,
    )

    # print(f"Dataloader arguments: {dataloader_args}\n")
    # print("Creating train and test dataloaders\n")
    # train dataloader
    train_loader = DataLoader(train_data, **dataloader_args)

    # test dataloader
    test_loader = DataLoader(test_data, **dataloader_args)

    return train_loader, test_loader
