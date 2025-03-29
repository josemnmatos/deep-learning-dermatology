from custom_models.mlp import MLP
from helper_methods import get_data_loaders
from pipeline import CustomModelPipeline
from torch import optim
import torch.nn as nn

# Experiment Parameters
N_REPEATS = 10
N_EPOCHS = 10
LR = 0.005


def init_pipeline(
    train_loader,
    validation_loader,
    train_loader_at_eval,
    test_loader,
    input_size,
    num_classes,
):
    """Initialize pipeline with MLP model"""
    N_LAYERS = 3
    HIDDEN_LAYERS = ((input_size + num_classes) // 2,) * N_LAYERS

    # Initialize MLP with proper parameters
    model = MLP(
        input_size=input_size, hidden_sizes=HIDDEN_LAYERS, num_classes=num_classes
    )

    pipeline = CustomModelPipeline(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(model.parameters(), lr=LR),
        n_epochs=N_EPOCHS,
        training_data=train_loader,
        validation_data=validation_loader,
        training_eval_data=train_loader_at_eval,
        test_data=test_loader,
    )
    return pipeline


def main():
    """Main function to run the experiment"""
    (
        train_loader,
        train_loader_at_eval,
        test_loader,
        validation_loader,
        n_channels,
        n_classes,
        task,
        pil_dataset,
    ) = get_data_loaders()

    # first sample
    sample = next(iter(train_loader))
    print(f"Sample Shape: {sample[0].shape}")

    # No of Neurons in Input Layer
    # Output is flattened before feeding to MLP
    input_size = n_channels * sample[0].shape[2] * sample[0].shape[3]

    # No of Neurons in Output Layer
    num_classes = n_classes

    print(f"Input size: {input_size}, Number of classes: {num_classes}")

    accuracies = []
    f1_scores = []
    t_losses = []
    v_losses = []

    # Create and execute pipelines for each repeat
    for i in range(N_REPEATS):
        print(f"Repeat {i + 1}/{N_REPEATS}")
        pipeline = init_pipeline(
            train_loader,
            validation_loader,
            train_loader_at_eval,
            test_loader,
            input_size,
            num_classes,
        )
        pipeline.execute()
        # get scores on test data
        scores = pipeline.get_scores()
        accuracies.append(scores["accuracy"])
        f1_scores.append(scores["f1"])

    # Calculate average and std of accuracies and f1 scores
    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    std_accuracy = (
        sum([(x - avg_accuracy) ** 2 for x in accuracies]) / len(accuracies)
    ) ** 0.5
    std_f1 = (sum([(x - avg_f1) ** 2 for x in f1_scores]) / len(f1_scores)) ** 0.5

    print(f"Average accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Average F1 score: {avg_f1:.4f} ± {std_f1:.4f}")


if __name__ == "__main__":
    main()
