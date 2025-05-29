import torch
from torch.utils.data import DataLoader
from sytorch import nn

"""
Evaulates top-1 accuracy of a model on a dataset.
"""


def evaluate(model: nn.Module, dataloader: DataLoader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy
