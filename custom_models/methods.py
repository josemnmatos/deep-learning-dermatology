import torch


def _validation(
    validation_data, nn, criterion, to_device=True, device="cuda", flatten_input=True
):
    if to_device:
        nn = nn.to(device)

    # Validation
    loss = 0

    with torch.no_grad():
        for X_train, y_train in validation_data:

            # send everything to the device (ideally a GPU)
            if to_device:
                X_train = X_train.to(device)
                y_train = y_train.to(device)

            if flatten_input:
                # Flatten RGB images into a single vector
                X_train = X_train.view(X_train.size(0), -1)

            # Remove unused dimension and convert to long
            y_train = y_train.squeeze().long()

            # Forward pass
            outputs = nn(X_train)
            loss += criterion(outputs, y_train).item()

    loss /= len(validation_data)
    return loss


def fit(
    training_data,
    validation_data,
    nn,
    criterion,
    optimizer,
    n_epochs,
    to_device=True,
    device="cuda",
    flatten_input=True,
):
    nn.train()

    # send everything to the device (ideally a GPU)
    if to_device:
        nn = nn.to(device)

    # Train the network
    loss_values = {
        "train": [],
        "validation": [],
    }
    for epoch in range(n_epochs):
        accu_loss = 0

        for X_train, y_train in training_data:

            # send everything to the device (ideally a GPU)
            if to_device:
                X_train = X_train.to(device)
                y_train = y_train.to(device)

            if flatten_input:
                # Flatten RGB images into a single vector
                X_train = X_train.view(X_train.size(0), -1)

            # Remove unused dimension and convert to long
            y_train = y_train.squeeze().long()

            # Forward pass
            outputs = nn(X_train)
            loss = criterion(outputs, y_train)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accu_loss += loss.item()

        accu_loss /= len(training_data)

        # if (epoch+1) % 10 == 0:
        loss_values["train"].append(accu_loss)

        # Validation
        val_loss = _validation(
            validation_data=validation_data,
            nn=nn,
            criterion=criterion,
            to_device=to_device,
            flatten_input=flatten_input,
            device=device,
        )
        loss_values["validation"].append(val_loss)

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch+1}/{n_epochs}], Training Loss: {accu_loss}, Validation Loss: {val_loss}"
            )

    return loss_values, nn


from sklearn.metrics import confusion_matrix, f1_score


def evaluate_network(nn, test_data, to_device=True, flatten_input=True, device="cuda"):
    # Set the model to evaluation mode
    nn.eval()

    if to_device:
        nn = nn.to(device)

    total_samples = len(test_data.dataset)
    correct_sample_predictions = 0
    all_labels = []
    all_preds = []

    # Run the model on the test data
    with torch.no_grad():
        for X, y in test_data:

            if to_device:
                X = X.to(device)
                y = y.to(device)

            if flatten_input:
                # Flatten RGB images into a single vector
                X = X.view(X.size(0), -1)

            # Remove unused dimension and convert to long
            y = y.squeeze().long()

            # Forward pass
            outputs = nn(X)

            # Get the predicted class
            _, predicted = torch.max(outputs, 1)

            # Accumulate labels and predictions for confusion matrix and F1 score
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            # Compute accuracy
            correct_sample_predictions += (predicted == y).sum().item()

    # Compute accuracy
    accuracy = correct_sample_predictions / total_samples

    # Compute confusion matrix
    conf_mat = confusion_matrix(all_labels, all_preds)

    # Compute F1 score
    # micro - Calculate metrics globally by counting the total true positives, false negatives and false positives.
    f1_global = f1_score(all_labels, all_preds, average="micro")
    # macro - Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    f1_unweighted = f1_score(all_labels, all_preds, average="macro")
    # weighted - Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    return {
        "acc": accuracy,
        "cm": conf_mat,
        "f1_global": f1_global,
        "f1_unweighted": f1_unweighted,
        "f1_weighted": f1_weighted,
    }
