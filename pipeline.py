from sklearn.metrics import (
    accuracy_score,
    f1_score,
    auc,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
import torch
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
import copy
import numpy as np
import time
from custom_models.mlp import MLP
from torchvision import transforms
from sklearn.preprocessing import label_binarize


class CustomModelPipeline:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer_class: torch.optim.Optimizer,
        optimizer_params: dict,  # e.g. {'lr': 0.001, 'weight_decay': 0}
        n_epochs: int,
        training_data: DataLoader,
        validation_data: DataLoader,
        test_data: DataLoader,
        patience: int = 10,
        min_delta: float = 0.001,
        device: str = "cpu",
    ):
        self.model_instance = model
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.n_epochs = n_epochs
        self.training_data = training_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.patience = patience
        self.min_delta = min_delta
        self.num_classes = 7  #  7 classes for this classification task

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def _reset_state(self):
        """Resets internal state variables before a new execution run."""
        # Create a fresh copy of the model to avoid state carry-over if class reused
        self.model = copy.deepcopy(self.model_instance)
        self.model.to(self.device)

        # Instantiate  optimizer for the current model 
        self.optimizer = self.optimizer_class(
            self.model.parameters(), **self.optimizer_params
        )

        self._model_is_trained = False
        self.__training_losses = []
        self.__validation_losses = []
        self.__best_model_state = None
        self.__validation_accuracy = None
        self.__validation_f1 = None
        self.__test_accuracy = None
        self.__test_f1 = None
        self.__run_duration = None

    def __train_model(self):
        """
        Internal method to handle the actual training loop, validation for
        early stopping, and saving the best model state.
        """
        self._reset_state()  # new state before training
        start_time = time.time()

        best_val_loss = float("inf")
        counter = 0

        print(f"Starting training for {self.n_epochs} epochs...")
        for epoch in range(self.n_epochs):
            # Training step
            self.model.train()
            training_loss = self.__training_loop(self.training_data)
            self.__training_losses.append(training_loss)

            # Validation step (for loss calculation and early stopping)
            self.model.eval()
            validation_loss = self.__run_eval_loop(self.validation_data)
            self.__validation_losses.append(validation_loss)

            print(
                f"Epoch {epoch + 1}/{self.n_epochs} - Train Loss: {training_loss:.4f}, Val Loss: {validation_loss:.4f}"
            )

            # Early stopping check
            if validation_loss < best_val_loss - self.min_delta:
                best_val_loss = validation_loss
                counter = 0
                self.__best_model_state = copy.deepcopy(self.model.state_dict())
                # print(f"Epoch {epoch+1}: Val loss improved to {best_val_loss:.4f}. Saving model.")
            else:
                counter += 1
                if counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    break

        # Load the best model state found during training
        if self.__best_model_state is not None:
            self.model.load_state_dict(self.__best_model_state)
            print("Loaded best model state based on validation loss.")
        else:
            print(
                "Warning: No improvement detected or only one epoch ran. Using final model state."
            )
            self.__best_model_state = self.model.state_dict()  # Keep the final state

        self._model_is_trained = True
        self.__run_duration = time.time() - start_time
        print(f"Training completed in {self.__run_duration:.2f} seconds.")

    def execute_and_validate(self):
        """
        Trains the model and evaluates performance on the VALIDATION set.
        Use this method for hyperparameter tuning.
        """
        self.__train_model()  # Run the core training logic

        # Evaluate the best model state on the VALIDATION set
        print("Evaluating best model on VALIDATION set...")
        self.model.eval()
        val_preds, val_labels, _ = self.__run_inference_loop(self.validation_data)

        self.__validation_accuracy = accuracy_score(val_labels, val_preds)
        self.__validation_f1 = f1_score(val_labels, val_preds, average="macro")

        print(f"Validation Accuracy: {self.__validation_accuracy:.4f}")
        print(f"Validation Macro F1 Score: {self.__validation_f1:.4f}")

        return self.get_validation_scores()

    def execute_and_test(self):
        """
        Trains the model and evaluates performance on the TEST set.
        For final evaluation on the TEST set.
        """
        self.__train_model() 

        # Evaluate the best model state on the TEST set
        print("Evaluating final model on TEST set...")
        self.model.eval()
        test_preds, test_labels, outputs = self.__run_inference_loop(self.test_data)

        test_labels = np.array(test_labels)
        test_preds = np.array(test_preds)
        outputs = np.array(outputs)

        # calculate OvR AUC
        # test_labels are binary (0 or 1) and test_preds are probabilities
        test_labels_bin = label_binarize(test_labels, classes=range(self.num_classes))

        print(f"Test Labels Shape: {test_labels_bin.shape}")
        print(f"Test Predictions Shape: {test_preds.shape}")
        print(f"Test Labels: {test_preds}")

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(test_labels_bin[:, i], outputs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            print(f"Class {i} AUC: {roc_auc[i]:.4f}")

        # micro-average ROC AUC
        fpr["micro"], tpr["micro"], _ = roc_curve(
            test_labels_bin.ravel(), outputs.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # macro-average ROC AUC
        roc_auc["macro"] = roc_auc_score(test_labels_bin, outputs, average="macro")

        # calculate Conf.Matrix for test set
        self.__test_confusion_matrix = confusion_matrix(test_labels, test_preds)
        self.__test_accuracy = accuracy_score(test_labels, test_preds)
        self.__test_f1 = f1_score(test_labels, test_preds, average="macro")
        self.__test_fpr = fpr
        self.__test_tpr = tpr
        self.__test_roc_auc = roc_auc

        print("-" * 30)
        print(f"FINAL TEST SET PERFORMANCE:")
        print(f"Test Accuracy: {self.__test_accuracy:.4f}")
        print(f"Test Macro F1 Score: {self.__test_f1:.4f}")
        print(f"Test Confusion Matrix:\n{self.__test_confusion_matrix}")
        print(f"Test ROC AUC (macro): {roc_auc['macro']:.4f}")
        print(f"Test ROC AUC (micro): {roc_auc['micro']:.4f}")
        print(f"Test ROC AUC (per class): {roc_auc}")

        print("-" * 30)

        # Return test scores
        return self.get_test_scores()

    def __training_loop(self, dataloader):
        """Runs one epoch of training."""
        batch_losses = []
        self.model.train()
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            if isinstance(self.model, MLP):  
                X_batch = X_batch.flatten(start_dim=1)
            X_batch = X_batch / 255.0  
            y_batch = y_batch.squeeze().long()

            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            batch_losses.append(loss.item())
        return np.mean(batch_losses)

    def __run_eval_loop(self, dataloader):
        """Runs evaluation for one epoch to calculate loss."""
        batch_losses = []
        self.model.eval()
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                if isinstance(
                    self.model, MLP
                ):  
                    X_batch = X_batch.flatten(start_dim=1)
                X_batch = X_batch / 255.0  # Normalize
                y_batch = y_batch.squeeze().long()

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                batch_losses.append(loss.item())
        return np.mean(batch_losses)

    def __run_inference_loop(self, dataloader):
        """Runs inference to get predictions and labels."""
        all_preds = []
        all_labels = []
        all_outputs = []
        self.model.eval()
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                if isinstance(
                    self.model, MLP
                ):  
                    X_batch = X_batch.flatten(start_dim=1)
                X_batch = X_batch / 255.0  # Normalize
                y_true = y_batch.squeeze().long()

                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_true.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())
        return all_preds, all_labels, all_outputs


    def get_validation_scores(self):
        """Returns the accuracy and F1 score calculated on the validation set."""
        if self.__validation_accuracy is None:
            return {"accuracy": None, "f1": None}
        return {"accuracy": self.__validation_accuracy, "f1": self.__validation_f1}

    def get_test_scores(self):
        """Returns the accuracy and F1 score calculated on the test set."""
        if self.__test_accuracy is None:
            return {"accuracy": None, "f1": None}
        return {
            "accuracy": self.__test_accuracy,
            "f1": self.__test_f1,
            "confusion_matrix": self.__test_confusion_matrix,
            "roc_auc": self.__test_roc_auc,
            "fpr": self.__test_fpr,
            "tpr": self.__test_tpr,
        }

    def get_losses(self):
        """Returns the training and validation losses per epoch."""
        return {"train": self.__training_losses, "val": self.__validation_losses}

    def get_run_duration(self):
        """Returns the duration of the last training run in seconds."""
        return self.__run_duration
