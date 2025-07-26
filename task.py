"""pytorch-example: A Flower / PyTorch app."""

import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
)
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

from flwr.common.typing import UserConfig

CIFAR10_NORMALIZATION = (
    (0.4914, 0.4822, 0.4465),  # mean (R, G, B)
    (0.2023, 0.1994, 0.2010),  # std  (R, G, B)
)
MNIST_NORMALIZATION = (
    (0.1307,),
    (0.3081,)
)

import yaml
with open("./config/dataset.yaml", "r") as f:
        dataset_config = yaml.safe_load(f)
DATASET = dataset_config.get("dataset", [])

if DATASET == "cifar10":
    img_key = "img"
elif DATASET == "mnist":
    img_key = "image"

def Net():
    if DATASET == "cifar10":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        # Modify classifier head for CIFAR-10 (10 output classes)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 10)
        return model
    elif DATASET == "mnist":
        from models.lenet5 import LeNet5
        return LeNet5()
    else:
        raise ValueError(f"Dataset {DATASET} not known!")

fds = None  # Cache FederatedDataset

def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=1.0,
            seed=42,
        )
        if DATASET == "cifar10":
            dataset_name = "uoft-cs/cifar10"
        elif DATASET == "mnist":
            dataset_name = "ylecun/mnist"
        fds = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    train_partition = partition_train_test["train"].with_transform(
        apply_train_transforms
    )
    test_partition = partition_train_test["test"].with_transform(apply_eval_transforms)
    # Use drop_last=True when the model uses Batch Normalization:
    #   This results in a slight loss of data, but enables models with batch normalization to be able to train 
    #   if there happens to be only 1 sample in the last batch (yields an error, since it can't do batch normalization with just 1 sample).
    trainloader = DataLoader(train_partition, batch_size=32, shuffle=True, drop_last=True) 
    testloader = DataLoader(test_partition, batch_size=32)

    return trainloader, testloader

def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch[img_key]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss

def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch[img_key].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def apply_train_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    if DATASET == "cifar10":
        NORMALIZATION = CIFAR10_NORMALIZATION
    elif DATASET == "mnist":
        NORMALIZATION = MNIST_NORMALIZATION
    else:
        raise ValueError(f"Dataset {DATASET} not known!")
    
    TRANSFORMS = Compose([
        ToTensor(), 
        Normalize(*NORMALIZATION)
    ])
    batch[img_key] = [TRANSFORMS(img) for img in batch[img_key]]
    return batch


def apply_eval_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    return apply_train_transforms(batch)

def apply_test_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    return apply_eval_transforms(batch)


def create_run_dir(config: UserConfig, run_name) -> Path:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_name}/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)

    return save_path, run_dir