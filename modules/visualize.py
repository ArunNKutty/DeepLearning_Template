import matplotlib.pyplot as plt
import numpy as np
# Overlay gradcam on top of numpy image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import torch
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchmetrics import ConfusionMatrix
from torchvision import transforms


def convert_back_image(image):
    """Using mean and std deviation convert image back to normal"""
    cifar10_mean = (0.4914, 0.4822, 0.4471)
    cifar10_std = (0.2469, 0.2433, 0.2615)
    image = image.numpy().astype(dtype=np.float32)

    for i in range(image.shape[0]):
        image[i] = (image[i] * cifar10_std[i]) + cifar10_mean[i]

    # To stop throwing a warning that image pixels exceeds bounds
    image = image.clip(0, 1)

    return np.transpose(image, (1, 2, 0))


def plot_sample_training_images(batch_data, batch_label, class_label, num_images=30):
    """Function to plot sample images from the training data."""
    images, labels = batch_data, batch_label

    # Calculate the number of images to plot
    num_images = min(num_images, len(images))
    # calculate the number of rows and columns to plot
    num_cols = 5
    num_rows = int(np.ceil(num_images / num_cols))

    # Initialize a subplot with the required number of rows and columns
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))

    # Iterate through the images and plot them in the grid along with class labels

    for img_index in range(1, num_images + 1):
        plt.subplot(num_rows, num_cols, img_index)
        plt.tight_layout()
        plt.axis("off")
        plt.imshow(convert_back_image(images[img_index - 1]))
        plt.title(class_label[labels[img_index - 1].item()])
        plt.xticks([])
        plt.yticks([])

    return fig, axs


def plot_train_test_metrics(results):
    """
    Function to plot the training and test metrics.
    """
    # Extract train_losses, train_acc, test_losses, test_acc from results
    train_losses = results["train_loss"]
    train_acc = results["train_acc"]
    test_losses = results["test_loss"]
    test_acc = results["test_acc"]

    # Plot the graphs in a 1x2 grid showing the training and test metrics
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Loss plot
    axs[0].plot(train_losses, label="Train")
    axs[0].plot(test_losses, label="Test")
    axs[0].set_title("Loss")
    axs[0].legend(loc="upper right")

    # Accuracy plot
    axs[1].plot(train_acc, label="Train")
    axs[1].plot(test_acc, label="Test")
    axs[1].set_title("Accuracy")
    axs[1].legend(loc="upper right")

    return fig, axs


def plot_misclassified_images(data, class_label, num_images=10):
    """Plot the misclassified images from the test dataset."""
    # Calculate the number of images to plot
    num_images = min(num_images, len(data["ground_truths"]))
    # calculate the number of rows and columns to plot
    num_cols = 5
    num_rows = int(np.ceil(num_images / num_cols))

    # Initialize a subplot with the required number of rows and columns
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))

    # Iterate through the images and plot them in the grid along with class labels

    for img_index in range(1, num_images + 1):
        # Get the ground truth and predicted labels for the image
        label = data["ground_truths"][img_index - 1].cpu().item()
        pred = data["predicted_vals"][img_index - 1].cpu().item()
        # Get the image
        image = data["images"][img_index - 1].cpu()
        # Plot the image
        plt.subplot(num_rows, num_cols, img_index)
        plt.tight_layout()
        plt.axis("off")
        plt.imshow(convert_back_image(image))
        plt.title(f"""ACT: {class_label[label]} \nPRED: {class_label[pred]}""")
        plt.xticks([])
        plt.yticks([])

    return fig, axs


# Function to plot gradcam for misclassified images using pytorch_grad_cam
def plot_gradcam_images(
    model,
    data,
    class_label,
    target_layers,
    device,
    targets=None,
    num_images=10,
    image_weight=0.25,
):
    """Show gradcam for misclassified images"""

    # Flag to enable cuda
    device == "cuda"

    # Calculate the number of images to plot
    num_images = min(num_images, len(data["ground_truths"]))
    # calculate the number of rows and columns to plot
    num_cols = 5
    num_rows = int(np.ceil(num_images / num_cols))

    # Initialize a subplot with the required number of rows and columns
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))

    # Initialize the GradCAM object
    # https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/grad_cam.py
    # https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/base_cam.py
    cam = GradCAM(model=model, target_layers=target_layers)

    # Iterate through the images and plot them in the grid along with class labels
    for img_index in range(1, num_images + 1):
        # Extract elements from the data dictionary
        # Get the ground truth and predicted labels for the image
        label = data["ground_truths"][img_index - 1].cpu().item()
        pred = data["predicted_vals"][img_index - 1].cpu().item()
        # Get the image
        image = data["images"][img_index - 1].cpu()

        # Get the GradCAM output
        # https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/utils/model_targets.py
        grad_cam_output = cam(
            input_tensor=image.unsqueeze(0),
            targets=targets
            # aug_smooth=True,
            # eigen_smooth=True,
        )
        grad_cam_output = grad_cam_output[0, :]

        
        overlayed_image = show_cam_on_image(
            convert_back_image(image),
            grad_cam_output,
            use_rgb=True,
            image_weight=image_weight,
        )

        # Plot the image
        plt.subplot(num_rows, num_cols, img_index)
        plt.tight_layout()
        plt.axis("off")
        plt.imshow(overlayed_image)
        plt.title(f"""ACT: {class_label[label]} \nPRED: {class_label[pred]}""")
        plt.xticks([])
        plt.yticks([])
    return fig, axs


def plot_grad_cam_different_targets(model, loader, classes, device):
    target_layers = [model.layer3[-1]]
    _, (input_tensor, target) = next(enumerate(loader))
    # Get the first image from the batch
    imput_tensor = input_tensor[0].unsqueeze(0).to(device)

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers)

    fig = plt.figure(figsize=(10, 5))
    plt.suptitle(f"GradCAM ID | Target Class : {classes[target[0]]}")
    for i in range(10):
        plt.subplot(2, 5, i + 1, aspect="auto")

        # Get the CAM
        grayscale_cam = cam(
            input_tensor=imput_tensor, targets=[ClassifierOutputTarget(i)]
        )

        # Get the first image from the batch
        grayscale_cam = grayscale_cam[0, :]

        unnormalized = transforms.Normalize(
            (-1.98947368, -1.98436214, -1.71072797), (4.048583, 4.11522634, 3.83141762)
        )(imput_tensor[0, :])
        visualization = show_cam_on_image(
            unnormalized.permute(1, 2, 0).cpu().detach().numpy(),
            grayscale_cam,
            use_rgb=True,
            image_weight=0.6,
        )
        plt.imshow(transforms.ToPILImage()(visualization))
        plt.title(
            f"GradCAM {i} | {classes[i]}",
        )
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()


def plot_grad_cam_misclassified(model, incorrect, classes, device):
    target_layers = [model.layer3[-1]]

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers)
    fig = plt.figure(figsize=(20, 10))
    for i in range(10):
        misclassified_tuple = incorrect[i]
        input_tensor = misclassified_tuple[0].unsqueeze(0).to(device)
        target_label = misclassified_tuple[1].item()
        predicted_label = misclassified_tuple[2].item()

        unnormalized = transforms.Normalize(
            (-1.98947368, -1.98436214, -1.71072797), (4.048583, 4.11522634, 3.83141762)
        )(input_tensor[0, :])

        plt.subplot(4, 5, i * 2 + 1, aspect="auto")
        # Get the CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)

        # Get the first image from the batch
        grayscale_cam = grayscale_cam[0, :]

        visualization = show_cam_on_image(
            unnormalized.permute(1, 2, 0).cpu().detach().numpy(),
            grayscale_cam,
            use_rgb=True,
            image_weight=0.6,
        )
        plt.imshow(transforms.ToPILImage()(visualization))
        plt.title(
            f"P {classes[predicted_label]} | T {classes[target_label]} | Pred CAM ",
        )
        plt.xticks([])
        plt.yticks([])

        plt.subplot(4, 5, i * 2 + 2, aspect="auto")
        # Get the CAM
        grayscale_cam = cam(
            input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_label)]
        )

        # Get the first image from the batch
        grayscale_cam = grayscale_cam[0, :]

        visualization = show_cam_on_image(
            unnormalized.permute(1, 2, 0).cpu().detach().numpy(),
            grayscale_cam,
            use_rgb=True,
            image_weight=0.6,
        )
        plt.imshow(transforms.ToPILImage()(visualization))
        plt.title(
            f"P {classes[predicted_label]} | T {classes[target_label]} | Target CAM ",
        )
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
