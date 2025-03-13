import matplotlib.pyplot as plt
import seaborn as sns


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
