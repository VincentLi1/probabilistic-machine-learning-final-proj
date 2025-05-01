# Uniform method for getting test statistics for different models

import argparse
import torch
from torchvision import models
from loaders import tiny_imagenet_loader, tiny_imagenet_corrupted_loader
from tqdm import tqdm
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices = ["sgd", "swa", "swag"])

args = parser.parse_args()

# Set up model
device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
model.load_state_dict(torch.load(f"./models/{args.model}_model.pt", weights_only=True))
model.to(device)

corruptions = ["brightness", "contrast", "defocus_blur"]

def evaluate_model(model, loader, description="Testing"):
    """ Return the accuracy of the model on the given data set (percentage correct labels assigned) """
    total_correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=description, leave=False):
            images, labels = images.to(device), labels.to(device)
            prediction = model(images)
            correct = (torch.argmax(prediction, dim=1) == labels).sum()
            total += 128
            total_correct += correct

    return total_correct/total

# Iterate through all corruptions and test
keys = []
values = []
for corruption in corruptions:
    for level in range(1, 3):
        test_loader = tiny_imagenet_corrupted_loader(
            corruption,
            severity=level,
            batch_size=128,
            root='./data/Tiny-ImageNet-C',
            num_workers=1
        )
        test_accuracy = evaluate_model(model, test_loader, corruption + '-' + str(level))

        keys.append(corruption + "-" + str(level))
        values.append(test_accuracy.item())
    
        print(f"{corruption} L{level}: {test_accuracy}")

# Save all values to a json
with open(f"results/{args.model}_results.json", "w") as file:
    json.dump(dict(zip(keys, values)), file, indent=4)