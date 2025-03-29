from helper_methods import *
from custom_models.methods import *
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def benchmark_model(
    model_constructor,
    training_data,
    training_data_at_eval,
    validation_data,
    test_data,
    n_epochs,
    n_repeats,
    optimizer_class,
    learning_rate,
    criterion_class,
    flatten_input,
    to_device=True,
    device="cuda",
):
    training_losses = []
    validation_losses = []
    training_evaluations = []
    test_evaluations = []

    for repeat_number in range(n_repeats):
        # Reset the model --------------------------
        model = model_constructor()

        criterion = criterion_class()

        optimizer = optimizer_class(model.parameters(), lr=learning_rate)

        # Training ----------------------------------
        model_loss, model = fit(
            training_data=training_data,
            validation_data=validation_data,
            nn=model,
            optimizer=optimizer,
            criterion=criterion,
            n_epochs=n_epochs,
            flatten_input=flatten_input,
            to_device=to_device,
            device=device,
        )

        training_losses.append(model_loss["train"])
        validation_losses.append(model_loss["validation"])

        # Evaluation ---------------------------------

        print("Evaluating model...")

        # Training data
        evals_training = evaluate_network(
            nn=model,
            test_data=training_data_at_eval,
            to_device=to_device,
            device=device,
            flatten_input=flatten_input,
        )

        # Test data
        evals_test = evaluate_network(
            nn=model,
            test_data=test_data,
            to_device=to_device,
            device=device,
            flatten_input=flatten_input,
        )

        training_evaluations.append(evals_training)
        test_evaluations.append(evals_test)

        print(f"Repetition no. {repeat_number + 1}/{n_repeats} completed.")

    return {
        "training_losses": training_losses,
        "validation_losses": validation_losses,
        "training_evaluations": training_evaluations,
        "test_evaluations": test_evaluations,
    }


def plot_average_losses(training_losses, validation_losses, title, ylim=None):
    avg_training_losses = np.mean(training_losses, axis=0)
    avg_validation_losses = np.mean(validation_losses, axis=0)

    std_training_losses = np.std(training_losses, axis=0)
    std_validation_losses = np.std(validation_losses, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(avg_training_losses, label="Training loss")
    plt.plot(avg_validation_losses, label="Validation loss")
    plt.fill_between(
        range(len(avg_training_losses)),
        avg_training_losses - std_training_losses,
        avg_training_losses + std_training_losses,
        alpha=0.3,
    )

    plt.fill_between(
        range(len(avg_validation_losses)),
        avg_validation_losses - std_validation_losses,
        avg_validation_losses + std_validation_losses,
        alpha=0.3,
    )

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if ylim:
        plt.ylim(ylim)
    plt.legend()
    plt.show()
