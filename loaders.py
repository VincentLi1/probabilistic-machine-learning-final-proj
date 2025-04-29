import os 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

def tiny_imagenet_loader(
    train,
    batch_size=128,
    root='./data/tiny-imagenet-200',
    num_workers=1
):
    """
    Load the Tiny ImageNet dataset.
    """

    path = os.path.join(root, 'train' if train else 'val')
    
    # Transform the data (resize, normalize)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975),
                             (0.2770, 0.2691, 0.2821))
    ])

    dataset = datasets.ImageFolder(root=path, transform=transform)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return loader

def tiny_imagenet_corrupted_loader(
    corruption='gaussian_noise',
    severity=1,
    batch_size=128,
    root='./data/Tiny-ImageNet-C',
    num_workers=1
):
    """
    Load the Tiny ImageNet-C dataset with specified corruption and severity.
    """
    
    assert 1 <= severity <= 5, "Severity must be between 1 and 5"
    
    corruption_path = os.path.join(root, corruption, str(severity))

    # Transform the data (resize, normalize)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # TODO: Should be mean and std of the original Tiny ImageNet dataset but check on this
        transforms.Normalize((0.4802, 0.4481, 0.3975),
                             (0.2770, 0.2691, 0.2821))
    ])

    dataset = datasets.ImageFolder(root=corruption_path, transform=transform)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return loader