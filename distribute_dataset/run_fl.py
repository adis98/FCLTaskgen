"""
Based on MVE of flower
"""
import argparse
import pickle
import warnings
from collections import OrderedDict
from typing import Tuple, Optional, Dict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common import NDArrays, Scalar
from flwr.server import History
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from sparse2coarse import sparse2coarse

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 2
ROUNDS = 100

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self, out_dim) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def load_data(client_id, alpha, repetition, course = True):
    """Load CIFAR-10 (training and test set). Uses pre-defined splits from user/
    :param client_id:
    :param alpha:
    :param repetition:
    """
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    with open(f'./stash/alpha_{alpha}/client_sample_ids_trial_{repetition}.pkl', 'rb') as f:
        indices = pickle.load(f)[client_id]
        train_indices = [index for index in indices if index < 50000]
        test_indices = [index - 50000 for index in indices if index >= 50000]

    trainset = CIFAR100("./data", train=True, download=True, transform=trf)
    testset = CIFAR100("./data", train=False, download=True, transform=trf)
    if course:
        trainset.targets = sparse2coarse(trainset.targets)
        testset.targets = sparse2coarse(testset.targets)
    return DataLoader(trainset, batch_size=32, sampler=SubsetRandomSampler(train_indices)), \
        DataLoader(testset, sampler=SubsetRandomSampler(test_indices))


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################




# Define Flower client
class FlowerClient(fl.client.NumPyClient):

    def __init__(self, client_id, alpha, repetition, coarse = True):
        self.client_id = client_id
        self.alpha = alpha
        self.repetition = repetition

        if coarse:
            # Load model and data (simple CNN, CIFAR-10)
            self.net = Net(20).to(DEVICE)
        else:
            self.net = Net(100).to(DEVICE)
        self.init_data()

    def init_data(self):
        trainloader, testloader = load_data(self.client_id, self.alpha, self.repetition)
        self.trainloader = trainloader
        self.testloader = testloader
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=EPOCHS)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}



def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_set = CIFAR100("./data", train=False, download=True, transform=trf)
    test_set.targets = sparse2coarse(test_set.targets)
    test_loader = DataLoader(test_set, batch_size=100)
    # Use the last 5k training examples as a validation set

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)  # Update model with the latest parameters

        with torch.no_grad():
            loss, accuracy = test(model, test_loader)
        return loss, {"accuracy": accuracy}

    return evaluate

def main(alpha, clientid, repetition):
    if clientid > -1:
        # Start Flower client
        client = FlowerClient(client_id=clientid,
                                alpha=alpha,
                                repetition=repetition)
        fl.client.start_numpy_client(
            server_address="127.0.0.1:8080",
            client=client,
        )
        model = client.net
    else:
        model = Net(20).to(DEVICE)

        # Define strategy
        strategy = fl.server.strategy.FedAvg(evaluate_fn=get_evaluate_fn(model))

        # Start Flower server
        history: History = fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=ROUNDS),
            strategy=strategy,
        )
    torch.save(model, f'./models/{alpha}_{clientid}_{repetition}.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Argument Parser Example")
    parser.add_argument("--alpha", type=int, help="The alpha value")
    parser.add_argument("--clientid", type=int, help="The client ID")
    parser.add_argument("--replication", type=int, help="The replication value")

    args = parser.parse_args()
    main(args.alpha, args.clientid, args.replication)
