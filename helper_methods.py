import matplotlib.pyplot as plt
import seaborn as sns
from medmnist import INFO, Evaluator
import torch.utils.data as data
import torchvision.transforms as transforms
import medmnist


# Get data from MedMNIST dataset, (Script from given notebook)
def get_data_loaders():
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
    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # load the data
    train_dataset = DataClass(
        split="train", transform=data_transform, download=download
    )
    test_dataset = DataClass(split="test", transform=data_transform, download=download)

    # ?
    validation_dataset = DataClass(
        split="val", transform=data_transform, download=download
    )

    pil_dataset = DataClass(split="train", download=download)

    # encapsulate data into dataloader form
    train_loader = data.DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    train_loader_at_eval = data.DataLoader(
        dataset=train_dataset, batch_size=2 * BATCH_SIZE, shuffle=False
    )
    test_loader = data.DataLoader(
        dataset=test_dataset, batch_size=2 * BATCH_SIZE, shuffle=False
    )

    validation_loader = data.DataLoader(
        dataset=validation_dataset, batch_size=2 * BATCH_SIZE, shuffle=False
    )

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


def plot_losses(losses, title):
    # losses should be a dict {"train": [list of train losses], "validation": [list of val losses]}
    plt.plot(losses["train"], label="Training Loss")
    plt.plot(losses["validation"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.show()


def print_7_class_confusion_matrix(conf_matrix):
    print("Confusion Matrix:")
    print("True\Pred\n", end="")
    for i in range(7):
        print(f"\t{i}", end="")
    print()
    for i in range(7):
        print(f"{i}", end="")
        for j in range(7):
            print(f"\t{conf_matrix[i][j]}", end="")
        print()


def plot_7_class_confusion_matrix(conf_matrix, title):
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()


def print_eval(evals_dict):
    for key, value in evals_dict.items():
        print(f"{key}: {value}")
    print()
