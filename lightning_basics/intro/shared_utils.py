import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms


# basic pytorch mlp class
class TorchMLP(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TorchMLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = torch.nn.Linear(hidden_dim * 2, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x_in):

        x_in = torch.flatten(x_in, start_dim=1)
        z = self.fc1(x_in)
        z = self.relu(z)
        z = self.fc2(z)
        z = self.relu(z)
        z = self.fc3(z)
        return z


def dataset_reader():
    """Function to read the dataset and return the train, validation and test data loaders"""

    # define a transform to normalize the data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # download and load the training data
    trainset = datasets.FashionMNIST(
        "data/", download=True, train=True, transform=transform
    )
    testdata = datasets.FashionMNIST(
        "data/", download=True, train=False, transform=transform
    )

    train_data, val_data = random_split(
        trainset, [55000, 5000], generator=torch.Generator().manual_seed(42)
    )

    return train_data, val_data, testdata


def dataset_loader(train_data, val_data, testdata, num_workers):
    """Function to load the dataset and return the train, validation and test data loaders"""

    train_loader = DataLoader(
        train_data, batch_size=64, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_data, batch_size=64, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        testdata, batch_size=64, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def compute_accuracy(model, dataloader, device):
    """Function to compute the accuracy of the model on the data provided"""
    model = model.eval()

    correct = 0.0
    total_examples = 0

    for _, (features, labels) in enumerate(dataloader):

        features = features.to(device)
        labels = labels.to(device)

        # print(model.device)
        # print(features.device)

        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)

        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return correct / total_examples
