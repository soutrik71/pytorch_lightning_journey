# basic pytorch mlp class
import torch
import torchmetrics
import lightning as L
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np


class PyTorchMLP(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.all_layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_features, 100),
            torch.nn.BatchNorm1d(100),
            torch.nn.Dropout1d(0.2),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(100, 50),
            torch.nn.BatchNorm1d(50),
            torch.nn.ReLU(),
            torch.nn.Dropout1d(0.2),
            # output layer
            torch.nn.Linear(50, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        logits = self.all_layers(x)
        return logits


class LightningMLP(L.LightningModule):

    def __init__(self, model, learning_rate, lr_scheduling, num_iters=1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.save_hyperparameters(ignore=["model"])
        self.lr_scheduling = lr_scheduling
        self.num_iters = num_iters

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        """
        Shared step for training and validation
        """
        features, label = batch
        logits = self(features)
        loss = self.criterion(logits, label)
        predictions = torch.argmax(logits, dim=1)
        return label, loss, predictions

    def training_step(self, batch, batch_idx):
        """
        Training Step loss and metric calculation
        """

        label, loss, predictions = self._shared_step(batch)
        self.log(name="train_loss", value=loss, prog_bar=True)
        # metric calculation
        self.train_acc(predictions, label)
        self.log(
            name="train_acc",
            value=self.train_acc,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation Step loss and metric calculation
        """

        label, loss, predictions = self._shared_step(batch)
        self.log(name="val_loss", value=loss, prog_bar=True)
        # metric calculation
        self.test_acc(predictions, label)
        self.log(
            name="val_acc",
            value=self.test_acc,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

    def test_step(self, batch, batch_idx):
        """
        Test step metric calculation
        """
        label, loss, predictions = self._shared_step(batch)
        # metric calculation
        self.test_acc(predictions, label)
        self.log(
            name="test_acc",
            value=self.test_acc,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

    def configure_optimizers(
        self,
    ):
        """
        Optimizer config
        """
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=0.9
        )

        if not self.lr_scheduling:

            return optimizer

        elif self.lr_scheduling == "step":
            sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": sch,
                    "monitor": "train_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        elif self.lr_scheduling == "plateau":
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": sch,
                    "monitor": "train_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        elif self.lr_scheduling == "cosine":
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.num_iters
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": sch,
                    "monitor": "train_loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }


class CustomDataset(Dataset):
    def __init__(self, feature_array, label_array, transform=None):

        self.x = feature_array
        self.y = label_array
        self.transform = transform

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.transform is not None:
            x = self.transform(x)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)

    def __len__(self):
        return self.y.shape[0]


class LightningData(L.LightningDataModule):
    def __init__(self, data_dir="./data/", num_workers=0, batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage):
        X, y = make_classification(
            n_samples=20000,
            n_features=100,
            n_informative=10,
            n_redundant=40,
            n_repeated=25,
            n_clusters_per_class=5,
            flip_y=0.05,
            class_sep=0.5,
            random_state=123,
        )

        # train test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123
        )
        # train val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=123
        )
        if stage == "fit":
            self.train_dataset = CustomDataset(
                feature_array=X_train.astype(np.float32),
                label_array=y_train.astype(np.int64),
            )

            self.val_dataset = CustomDataset(
                feature_array=X_val.astype(np.float32),
                label_array=y_val.astype(np.int64),
            )
        if stage == "test":
            self.test_dataset = CustomDataset(
                feature_array=X_val.astype(np.float32),
                label_array=y_val.astype(np.int64),
            )

        if stage == "predict":

            self.test_dataset = CustomDataset(
                feature_array=X_test.astype(np.float32),
                label_array=y_test.astype(np.int64),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
