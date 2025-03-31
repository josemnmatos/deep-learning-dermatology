# %%
# Import necessary libraries and modules
# Make sure 'pipeline_corrected.py' contains the CustomModelPipelineCorrected class
from pipeline import CustomModelPipelineCorrected
from custom_models.mlp import MLP  # Assuming this is your MLP model definition
from helper_methods import get_data_loaders  # Assuming this loads your data
from torch import optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import torch
import pandas as pd  # Added for storing results

# set seaborn style
sns.set_theme(
    context="notebook", style="whitegrid", palette="bright", color_codes=True, rc=None
)

# %%
# --- Configuration Section ---

# Parameters for the search
N_EPOCHS_SEARCH = (
    100  # Max epochs for hyperparameter search runs (early stopping applies)
)
N_EPOCHS_FINAL = 200  # Max epochs for the final model training
ES_PATIENCE = 15  # Early stopping patience

# Define Hyperparameter Search Space
# Experiment 1: Hidden Layer Configurations (Size and Depth combined for simplicity here)
# You can structure this differently if you want strict separation like before
layer_configs = {
    "HL_256_64": (256, 64),
    "HL_512_128": (512, 128),
    "HL_1024_256": (1024, 256),
    "HL_2048_512": (2048, 512),
    "HL_4096_1024": (4096, 1024),
    "HL_1024": (1024,),  # From Depth Exp
    "HL_1024_512_256": (1024, 512, 256),  # From Depth Exp
    # Add more configurations if needed
}

# Experiment 2: Optimizers and Learning Rates
optimizers_config = {
    "Adam_0.005": (optim.Adam, 0.005),
    "Adam_0.001": (optim.Adam, 0.001),
    "RMSprop_0.005": (optim.RMSprop, 0.005),
    "RMSprop_0.001": (optim.RMSprop, 0.001),
}

# Experiment 3: Loss Functions (Optional - if you want to search this)
# criterion_config = {
#     "CrossEntropy": nn.CrossEntropyLoss,
#     "MultiMargin": nn.MultiMarginLoss,
# }
# For now, let's stick to CrossEntropyLoss as indicated by your previous findings
criterion_to_use = nn.CrossEntropyLoss()

# Experiment 4: Regularization (Optional - add weight decay to search)
weight_decays = [0, 0.0001]  # Example: Test with and without L2


# --- Data Loading ---
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

# Calculate input size
sample = next(iter(train_loader))
input_size = n_channels * sample[0].shape[2] * sample[0].shape[3]
num_classes = n_classes
print(f"Input size: {input_size}, Number of classes: {num_classes}")


# --- Helper Function to Create Pipeline ---
def create_pipeline_for_config(layer_config, opt_class, lr, wd, criterion):
    """Initialize pipeline with MLP model for a specific configuration"""
    # Initialize MLP with proper parameters
    model = MLP(
        input_size=input_size, hidden_sizes=layer_config, num_classes=num_classes
    )

    # Initialize optimizer for this specific model instance
    optimizer = opt_class(model.parameters(), lr=lr, weight_decay=wd)

    pipeline = CustomModelPipelineCorrected(
        model=model,
        criterion=criterion(),  # Instantiate the criterion
        optimizer=optimizer,  # Pass the optimizer instance
        n_epochs=N_EPOCHS_SEARCH,  # Use search epochs
        training_data=train_loader,
        validation_data=validation_loader,
        test_data=test_loader,  # Needed for the final .test() call later
        patience=ES_PATIENCE,
        min_delta=0.001,
    )
    return pipeline


# --- Hyperparameter Search Loop ---

results_log = []
best_val_f1 = -1.0
best_config_details = {}
best_pipeline_instance = None  # To store the pipeline object of the best run

print("\n=== Starting Hyperparameter Search ===")

for layers_name, layers_tuple in layer_configs.items():
    for opt_name, (opt_class, lr) in optimizers_config.items():
        for wd in weight_decays:  # Loop through weight decays

            config_name = f"{layers_name}_{opt_name}_WD{wd}"
            print(f"\n--- Testing Config: {config_name} ---")

            # Create and execute the pipeline for this configuration
            # Note: We run each config only ONCE here for speed.
            # For more robustness, you could add N_REPEATS here and average validation scores.
            try:
                pipeline = create_pipeline_for_config(
                    layer_config=layers_tuple,
                    opt_class=opt_class,
                    lr=lr,
                    wd=wd,
                    criterion=nn.CrossEntropyLoss,  # Pass class, it will be instantiated
                )
                pipeline.execute()  # Trains and evaluates on VALIDATION set

                # Get validation scores for this run
                val_scores = pipeline.get_validation_scores()
                current_val_f1 = (
                    val_scores["f1"] if val_scores["f1"] is not None else -1
                )  # Handle potential None
                current_val_acc = (
                    val_scores["accuracy"] if val_scores["accuracy"] is not None else -1
                )

                print(
                    f"Config {config_name} - Validation Results: Accuracy={current_val_acc:.4f}, F1={current_val_f1:.4f}"
                )

                # Log results
                results_log.append(
                    {
                        "config_name": config_name,
                        "layers": layers_name,
                        "optimizer": opt_class.__name__,
                        "lr": lr,
                        "weight_decay": wd,
                        "val_f1": current_val_f1,
                        "val_acc": current_val_acc,
                        # 'losses': pipeline.get_losses() # Optional: store losses
                    }
                )

                # Check if this is the best configuration so far based on validation F1
                if current_val_f1 > best_val_f1:
                    best_val_f1 = current_val_f1
                    best_config_details = {
                        "config_name": config_name,
                        "layers_name": layers_name,
                        "layers_tuple": layers_tuple,
                        "optimizer_class": opt_class,
                        "lr": lr,
                        "weight_decay": wd,
                        "criterion": nn.CrossEntropyLoss,  # Store criterion used
                        "val_f1": current_val_f1,
                        "val_acc": current_val_acc,
                    }
                    # Keep the pipeline instance that achieved the best result
                    # We might re-train, but having the state can be useful
                    # best_pipeline_instance = pipeline
                    print(
                        f"*** New best validation F1 found: {best_val_f1:.4f} for config {config_name} ***"
                    )

            except Exception as e:
                print(f"!!! Error running config {config_name}: {e} !!!")
                # Log the error
                results_log.append(
                    {
                        "config_name": config_name,
                        "layers": layers_name,
                        "optimizer": opt_class.__name__,
                        "lr": lr,
                        "weight_decay": wd,
                        "val_f1": -1,
                        "val_acc": -1,
                        "error": str(e),
                    }
                )


# --- Search Results Summary ---
print("\n=== Hyperparameter Search Complete ===")

# Convert results log to DataFrame for easier viewing/saving
results_df = pd.DataFrame(results_log)
print("Search Results Summary:")
print(
    results_df.sort_values(by="val_f1", ascending=False).to_string()
)  # Print sorted results

if not best_config_details:
    print("\n!!! No successful runs completed. Cannot proceed to final evaluation. !!!")
else:
    print(
        f"\nBest configuration found based on Validation F1 ({best_config_details['val_f1']:.4f}):"
    )
    print(f"  Config Name: {best_config_details['config_name']}")
    print(
        f"  Layers: {best_config_details['layers_name']} {best_config_details['layers_tuple']}"
    )
    print(f"  Optimizer: {best_config_details['optimizer_class'].__name__}")
    print(f"  Learning Rate: {best_config_details['lr']}")
    print(f"  Weight Decay: {best_config_details['weight_decay']}")

    # --- Final Model Training and Evaluation ---
    print("\n=== Training Final Model with Best Configuration ===")

    # Create the final pipeline instance with the best hyperparameters
    # We re-instantiate to ensure a fresh start
    final_pipeline = create_pipeline_for_config(
        layer_config=best_config_details["layers_tuple"],
        opt_class=best_config_details["optimizer_class"],
        lr=best_config_details["lr"],
        wd=best_config_details["weight_decay"],
        criterion=best_config_details["criterion"],
    )

    # Adjust final training parameters if needed
    final_pipeline.n_epochs = N_EPOCHS_FINAL  # Use final epochs
    # final_pipeline.patience = ES_PATIENCE + 5 # Maybe slightly more patience

    # Train the final model (again, uses validation for early stopping)
    final_pipeline.execute()

    print("\n=== Evaluating Final Model on TEST Set ===")
    # Evaluate the final trained model ONCE on the TEST set
    final_test_scores = final_pipeline.test()

    print("\n--- FINAL REPORTED TEST SCORES ---")
    print(f"Best Config Name: {best_config_details['config_name']}")
    print(f"Test Accuracy: {final_test_scores['accuracy']:.4f}")
    print(f"Test Macro F1 Score: {final_test_scores['f1']:.4f}")
    print("------------------------------------")

    # --- Plot Final Losses ---
    print("\nPlotting losses for the final model run...")

    def plot_final_losses(pipeline_instance, config_name):
        """Plots the training and validation losses for a single pipeline run."""
        losses = pipeline_instance.get_losses()
        train_losses = losses["train"]
        val_losses = losses["val"]

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Training loss", color="blue")
        plt.plot(val_losses, label="Validation loss", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Final Model Training Losses - Config: {config_name}")
        plt.legend()
        plt.grid(True)
        plt.show()

    plot_final_losses(final_pipeline, best_config_details["config_name"])

# Optional: Save results DataFrame
# results_df.to_csv("mlp_hyperparameter_search_results.csv", index=False)

# %%
