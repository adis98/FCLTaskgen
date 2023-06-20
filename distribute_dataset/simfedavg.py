import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Define the hyperparameters
num_clients = 4
num_epochs = 2
num_rounds = 80
learning_rate = 0.1
batch_size = 32
alphas = [1, 10, 100, 1000]
n_trials = 10
n_outputs = 20


# Define your neural network model
class Net(nn.Module):
    def __init__(self, out_dim = n_outputs) -> None:
        super(Net, self).__init__()

        self.fc = torch.nn.Sequential(
            nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_outputs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return x


# Define the training function for each client
def train(model, optimizer, train_loader, device):
    criterion = nn.CrossEntropyLoss()
    model.train()
    model.to(device)

    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


# Define the evaluation function
def evaluate(model, test_loader, device):
    # Switch behaviour for special layers during eval
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            predicted = predicted.reshape((-1, 1))
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# Define the federated averaging function
def federated_averaging(client_models, glob_model, device):
    # Initialize the global model with average weights from the client models

    with torch.no_grad():
        for param in glob_model.parameters():
            param.zero_()

        for client_model in client_models:
            for global_param, client_param in zip(glob_model.parameters(), client_model.parameters()):
                global_param.add_(client_param.data / num_clients)

    return global_model


if __name__ == "__main__":
    features = np.load("stash/features_normencoded.npy")
    labels = np.load("stash/labels_coarse.npy")
    np.random.seed(41)
    torch.manual_seed(41)
    test_features = torch.from_numpy(features[50000:])
    test_labels = torch.from_numpy(labels[50000:])
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    # Create custom test data loaders
    test_loader = DataLoader(
        TensorDataset(test_features, test_labels),
        batch_size=batch_size, shuffle=True)

    for alpha in alphas:
        for trial in range(n_trials):
            filepath = "stash/" + "alpha_" + str(alpha) + "/client_sample_ids_trial_" + str(trial) + ".pkl"
            with open(filepath, 'rb') as f:
                client_label_ids = pickle.load(f)

                # Remove test label ids before training since evaluation is done on the federator
                for cli in range(num_clients):
                    client_label_ids[cli] = client_label_ids[cli][client_label_ids[cli] < 50000]

                # Extract features and labels
                client_features = [features[client_label_ids[cli]] for cli in range(num_clients)]
                client_labels = [np.reshape(labels[client_label_ids[cli]], (-1,)) for cli in range(num_clients)]

                # Convert data to PyTorch tensors
                client_data = [torch.from_numpy(client_feat) for client_feat in client_features]
                client_targets = [torch.from_numpy(client_lab) for client_lab in client_labels]

                # Create custom data loaders
                train_loaders = [DataLoader(
                    TensorDataset(client_data[i], client_targets[i]),
                    batch_size=batch_size, shuffle=True)
                    for i in range(num_clients)]

                # Create the client models and data loaders
                client_models = [Net() for _ in range(num_clients)]
                # Initialize the global model with average weights from the client models
                global_model = Net()
                global_model.to(device)
                # Create the optimizer for each client
                client_optimizers = [optim.SGD(client_model.parameters(), lr=learning_rate) for client_model in
                                     client_models]
                # Perform federated learning
                for round in range(num_rounds):
                    for epoch in range(num_epochs):
                        # Train the client models
                        for client_model, client_optimizer, train_loader in zip(client_models, client_optimizers,
                                                                                train_loaders):
                            train(client_model, client_optimizer, train_loader, device)

                    # Perform federated averaging
                    global_model = federated_averaging(client_models, global_model, device)
                    [model.load_state_dict(global_model.state_dict()) for model in client_models]

                # Stash test accuracy
                    acc = evaluate(global_model, test_loader, device)
                    print("alpha", alpha, "trial", "round", round, "accuracy", acc)
