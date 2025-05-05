import os 
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from PIL import Image

def map_strings_to_line_numbers(filepath):
    """
    Reads a file and returns a dictionary mapping each line's string to its line number (starting at 0).
    
    Args:
        filepath (str): Path to the text file.
        
    Returns:
        dict: {string: line_number}
    """
    with open(filepath, 'r') as f:
        return {line.strip(): idx for idx, line in enumerate(f)}

class TinyImageNetTrainDataset(Dataset):
    """ Dataset structure for the given file structure of Tiny Imagenet training """
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to root folder with class subdirectories.
            class_name_to_label_fn (function): Maps folder name (str) to label (int).
            transform (callable, optional): Transform to apply to images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Get the correspondence with the wid labels and those used for tiny imagenet
        label_map = map_strings_to_line_numbers("./data/tiny-imagenet-200/wnids.txt")

        print("Folders:")
        for class_name in os.listdir(root_dir):
            # print(class_name)
            class_folder = os.path.join(root_dir, class_name, "images")
            if not os.path.isdir(class_folder):
                continue
            label = label_map[class_name]
            for fname in os.listdir(class_folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG')):
                    path = os.path.join(class_folder, fname)
                    self.samples.append((path, label))

        # print(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class TinyImageNetValDataset(Dataset):
    """ Dataset structure for the given file structure of Tiny Imagenet validation (different file structure) """
    
    def __init__(self, val_dir, transform=None):
        self.images_dir = os.path.join(val_dir, "images")
        self.annotations_file = os.path.join(val_dir, "val_annotations.txt")
        self.transform = transform

        # Get the correspondence with the wid labels and those used for tiny imagenet
        label_map = map_strings_to_line_numbers("./data/tiny-imagenet-200/wnids.txt")

        # Parse the annotations file
        self.image_labels = []
        self.class_to_idx = {}
        class_idx = 0
        
        with open(self.annotations_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split('\t')
                image_name = parts[0]
                class_name = parts[1]
                self.image_labels.append((image_name, label_map[class_name]))

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        image_name, label = self.image_labels[idx]
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

class TinyImageNetCorruptedDataset(Dataset):
    """ Dataset structure for the given file structure of Tiny Imagenet training """
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to root folder with class subdirectories.
            class_name_to_label_fn (function): Maps folder name (str) to label (int).
            transform (callable, optional): Transform to apply to images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Get the correspondence with the wid labels and those used for tiny imagenet
        label_map = map_strings_to_line_numbers("./data/tiny-imagenet-200/wnids.txt")

        print("Folders:")
        for class_name in os.listdir(root_dir):
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_folder):
                continue
            label = label_map[class_name]
            for fname in os.listdir(class_folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG')):
                    path = os.path.join(class_folder, fname)
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

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

    dataset = TinyImageNetTrainDataset(path, transform)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return loader

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

    # dataset = datasets.ImageFolder(root=corruption_path, transform=transform)
    dataset = TinyImageNetCorruptedDataset(corruption_path, transform)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return loader