import torch
from torchvision import datasets, transforms
import json
import os.path

def dataloader(data_dir):
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")

    # data_transforms:
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    v_transforms = transforms.Compose([transforms.Resize(225),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=train_transforms),
        "validation": datasets.ImageFolder(valid_dir, transform=v_transforms)
        }

    dataloaders = {
        "trainloder": torch.utils.data.DataLoader(image_datasets["train"], batch_size = 64, shuffle = True),
        "validationloader": torch.utils.data.DataLoader(image_datasets["validation"], batch_size = 64)
        }

    return dataloaders, image_datasets