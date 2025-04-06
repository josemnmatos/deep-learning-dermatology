import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from medmnist import INFO, Evaluator
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import medmnist


# Get data from MedMNIST dataset, (Script from given notebook)
def get_data_loaders(data_augmentation=False):
    print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")
    data_flag = "dermamnist"
    download = True
    BATCH_SIZE = 128

    info = INFO[data_flag]
    task = info["task"]
    n_channels = info["n_channels"]
    n_classes = len(info["label"])

    DataClass = getattr(medmnist, info["python_class"])

    # preprocessing
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
        ]
    )

    basic_trasform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_data_transform = train_transform if data_augmentation else basic_trasform

    # load the data
    train_dataset = DataClass(
        split="train", transform=train_data_transform, download=download
    )

    test_dataset = DataClass(split="test", transform=basic_trasform, download=download)

    validation_dataset = DataClass(
        split="val", transform=basic_trasform, download=download
    )

    pil_dataset = DataClass(split="train", download=download)

    # encapsulate data into dataloader form
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        pin_memory_device="cuda",
    )
    train_loader_at_eval = data.DataLoader(
        dataset=train_dataset,
        batch_size=2 * BATCH_SIZE,
        shuffle=False,
    )
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=2 * BATCH_SIZE,
        shuffle=False,
    )

    validation_loader = data.DataLoader(
        dataset=validation_dataset,
        batch_size=2 * BATCH_SIZE,
        shuffle=False,
    )

    if data_augmentation:
        print("-" * 50)
        print("Applied Training Augmentations:")
        for t in train_transform.transforms:
            print(f"- {t}")
        print("-" * 50)

    return (
        train_loader,
        train_loader_at_eval,
        test_loader,
        validation_loader,
        n_channels,
        n_classes,
        task,
        pil_dataset,
    )


# Done with the help of Claude Sonnet 3.7
def plot_avg_losses(t_losses, v_losses):
    """Plot average losses with 95% CI, handling early stopping"""

    # Find the maximum length across all arrays
    max_length = max(len(losses) for losses in t_losses)

    # Create dataframes with named indices to avoid duplicates
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()

    # Process each run's data
    for run_idx, (t_loss, v_loss) in enumerate(zip(t_losses, v_losses)):
        # Pad shorter arrays with NaN values
        t_padded = t_loss + [float("nan")] * (max_length - len(t_loss))
        v_padded = v_loss + [float("nan")] * (max_length - len(v_loss))

        # Add to dataframes
        train_df[f"run_{run_idx}"] = t_padded
        val_df[f"run_{run_idx}"] = v_padded

    # Set epoch as index
    train_df.index = range(max_length)
    val_df.index = range(max_length)

    # Create long-form dataframes for seaborn
    train_long = train_df.stack().reset_index()
    train_long.columns = ["epoch", "run", "loss"]
    train_long["type"] = "Train"

    val_long = val_df.stack().reset_index()
    val_long.columns = ["epoch", "run", "loss"]
    val_long["type"] = "Validation"

    # Combine data
    combined_df = pd.concat([train_long, val_long])

    # Plot with confidence intervals
    ax = sns.lineplot(
        data=combined_df,
        x="epoch",
        y="loss",
        hue="type",
        errorbar=("ci", 95),
        palette={"Train": "blue", "Validation": "orange"},
    )

    # Add transparency to the confidence intervals
    for collection in ax.collections:
        collection.set_alpha(0.1)

    plt.title(f"Train and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def get_class_weights(train_loader, n_classes):
    """Calculate class weights for imbalanced dataset"""
    # Get the number of samples in each class
    class_counts = np.zeros(n_classes)
    for _, labels in train_loader:
        for label in labels:
            class_counts[label.item()] += 1

    # Calculate the weights
    total_samples = sum(class_counts)
    class_weights = total_samples / (n_classes * class_counts)

    # Ensure weights tensor has correct shape and is of expected type
    return torch.tensor(class_weights, dtype=torch.float).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
