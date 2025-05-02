import os 
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from PIL import Image

def tiny_imagenet_train_loader(
    batch_size=128,
    root='./data/tiny-imagenet-200',
    num_workers=1
):
    """
    Load the Tiny ImageNet dataset.
    """

    path = os.path.join(root, "train")
    
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

class TinyImageNetValDataset(Dataset):
    def __init__(self, val_dir, transform=None):
        self.images_dir = os.path.join(val_dir, "images")
        self.annotations_file = os.path.join(val_dir, "val_annotations.txt")
        self.transform = transform

        # Parse the annotations file
        self.image_labels = []
        self.class_to_idx = {}
        class_idx = 0
        
        with open(self.annotations_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split('\t')
                image_name = parts[0]
                class_name = parts[1]
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = class_idx
                    class_idx += 1
                self.image_labels.append((image_name, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        image_name, label = self.image_labels[idx]
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def tiny_imagenet_val_loader(
    batch_size=128,
    root='./data/tiny-imagenet-200',
    num_workers=1
):
    """
    Load the Tiny ImageNet dataset.
    """

    path = os.path.join(root, "val")
    
    # Transform the data (resize, normalize)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975),
                             (0.2770, 0.2691, 0.2821))
    ])

    dataset = TinyImageNetValDataset(path, transform)

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