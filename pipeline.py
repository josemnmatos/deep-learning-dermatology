from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader
from torch import nn
from custom_models.mlp import MLP
import seaborn as sns
import matplotlib.pyplot as plt


class CustomModelPipeline:
    """
    Custom pipeline class to train a model that implements nn.Module.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: nn.Module,
        n_epochs: int,
        training_data: DataLoader,
        validation_data: DataLoader,
        test_data: DataLoader,
        training_eval_data: DataLoader,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.training_data = training_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.training_eval_data = training_eval_data
        self._model_is_trained = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__final_accuracy = None
        self.__final_f1 = None

    def execute(self):
        self.__training_losses = []
        self.__validation_losses = []

        self.model.to(self.device)

        # Train
        for epoch in range(self.n_epochs):
            # Training

            self.model.train()

            training_loss = self.__training_loop(
                dataloader=self.training_data,
                loss_fn=self.criterion,
                optimizer=self.optimizer,
                epoch_no=epoch,
            )

            # Validation

            self.model.eval()

            validation_loss = self.__validation(
                dataloader=self.validation_data,
                loss_fn=self.criterion,
            )

            self.__training_losses.append(training_loss)
            self.__validation_losses.append(validation_loss)

        self._model_is_trained = True

        # Evaluate
        self.evaluate_model()

        print(f"Final accuracy: {self.__final_accuracy}")
        print(f"Final F1 score: {self.__final_f1}")

    def __training_loop(self, dataloader, loss_fn, optimizer, epoch_no):
        size = len(dataloader.dataset)
        batch_no = len(dataloader)
        running_loss = 0.0

        for X_train, y_train in dataloader:
            # send everything to the device
            X_train = X_train.to(self.device)
            y_train = y_train.to(self.device)

            if isinstance(self.model, MLP):
                X_train = X_train.flatten(start_dim=1)

            X_train = X_train / 255.0  # Normalize the input data to 0,1

            # Remove unused dimension and convert to long
            y_train = y_train.squeeze().long()

            outputs = self.model(X_train)
            loss = loss_fn(outputs, y_train)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # Use item() to get scalar value

        epoch_loss = running_loss / batch_no
        print(f"Epoch {epoch_no}: training loss: {epoch_loss}")
        return epoch_loss  # Return scalar value

    def __validation(self, dataloader, loss_fn):
        loss = 0

        with torch.no_grad():
            for X_train, y_train in dataloader:
                # send everything to the device
                X_train = X_train.to(self.device)
                y_train = y_train.to(self.device)

                if isinstance(self.model, MLP):
                    X_train = X_train.flatten(start_dim=1)

                X_train = X_train / 255.0  # Normalize the input data to 0,1

                # Remove unused dimension and convert to long
                y_train = y_train.squeeze().long()

                # Forward pass
                outputs = self.model(X_train)
                loss += loss_fn(outputs, y_train).item()

        loss /= len(dataloader)
        return loss

    def evaluate_model(self):
        assert self._model_is_trained, "Model must be trained first."

        self.model.eval()
        self.model.to(self.device)

        all_labels = []
        all_preds = []

        with torch.no_grad():
            for X_test, y_test in self.test_data:
                X_test = X_test.to(self.device)
                y_test = y_test.to(self.device)

                if isinstance(self.model, MLP):
                    X_test = X_test.flatten(start_dim=1)

                X_test = X_test / 255.0  # Normalize the input data to 0,1

                y_test = y_test.squeeze().long()

                outputs = self.model(X_test)

                # Get predicted class indices
                _, predicted = torch.max(outputs, 1)

                # Move to CPU and convert to np array
                true_labels = y_test.cpu().numpy()
                predicted_labels = predicted.cpu().numpy()

                all_labels.extend(true_labels)
                all_preds.extend(predicted_labels)

        # Calculate metrics using all collected predictions
        self.__final_accuracy = accuracy_score(all_labels, all_preds)
        self.__final_f1 = f1_score(
            all_labels, all_preds, average="weighted"
        )  # Using weighted average for multi-class

    def get_losses(self):
        return {"train": self.__training_losses, "val": self.__validation_losses}

    def get_scores(self):
        return {
            "accuracy": self.__final_accuracy,
            "f1": self.__final_f1,
        }

    def plot_losses(self, y_lim=None):
        plt.plot(self.__training_losses, label="Training loss")
        plt.plot(self.__validation_losses, label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        if y_lim:
            plt.ylim(y_lim)

        plt.legend()
        plt.show()
