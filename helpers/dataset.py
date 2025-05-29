from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision

VAL_DIR = "data/imagenet-mini/val"

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def imagenet_mini(
    batch_size=64, num_workers=4, shuffle=False
) -> tuple[torchvision.datasets.ImageFolder, DataLoader]:
    dataset = torchvision.datasets.ImageFolder(VAL_DIR, transform=transform)
    return dataset, DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
