# EECS 545 Fall 2021
import os
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms


class DogDataset:
    """
    Dog Dataset.
    """
    def __init__(self, batch_size=4, dataset_path='data/images/dogs', if_resize=True):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.if_resize = if_resize
        self.train_dataset = self.get_train_numpy()
        self.x_mean, self.x_std = self.compute_train_statistics()
        self.transform = self.get_transforms()
        self.train_loader, self.val_loader = self.get_dataloaders()

    def get_train_numpy(self):
        train_dataset = torchvision.datasets.ImageFolder(os.path.join(self.dataset_path, 'train'))
        train_x = np.zeros((len(train_dataset), 224, 224, 3))
        # train_x = np.zeros((len(train_dataset), 64, 64, 3))
        for i, (img, _) in enumerate(train_dataset):
            train_x[i] = img
        return train_x / 255.0

    def compute_train_statistics(self):
        # TODO (part a): compute per-channel mean and std with respect to self.train_dataset
        
        x_mean = np.mean(self.train_dataset, axis=(0, 1, 2))  # per-channel mean
        x_std = np.std(self.train_dataset, axis=(0, 1, 2))  # per-channel std
        return x_mean, x_std

    def get_transforms(self):
        if self.if_resize:
            # TODO (part a): fill in the data transforms
            transform_list = [
                # resize the image to 32x32x3
                transforms.Resize((32, 32)),
                # convert image to PyTorch tensor
                transforms.ToTensor(),
                # normalize the image (use self.x_mean and self.x_std)
                transforms.Normalize(mean = self.x_mean, std = self.x_std)
            ]
        else: 
            # TODO (part f): fill in the data transforms
            # Note: Only change from part a) is there is no need to resize the image 
                transform_list = [
                # convert image to PyTorch tensor
                transforms.ToTensor(),
                # normalize the image (use self.x_mean and self.x_std)
                transforms.Normalize(mean = self.x_mean, std = self.x_std)
            ]
        transform = transforms.Compose(transform_list)
        return transform

    def get_dataloaders(self):
        # train set
        train_set = torchvision.datasets.ImageFolder(os.path.join(self.dataset_path, 'train'), transform=self.transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        # validation set
        val_set = torchvision.datasets.ImageFolder(os.path.join(self.dataset_path, 'val'), transform=self.transform)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader

    def plot_image(self, image, label):
        image = np.transpose(image.numpy(), (1, 2, 0))
        image = image * self.x_std.reshape(1, 1, 3) + self.x_mean.reshape(1, 1, 3)  # un-normalize
        plt.title(label)
        plt.imshow((image*255).astype('uint8'))
        plt.show()

    def get_semantic_label(self, label):
        mapping = {'African_hunting_dog': 0, 'Chihuahua': 1, 'dhole': 3, 'dingo': 4, 'Japanese_spaniel': 2}
        reverse_mapping = {v: k for k, v in mapping.items()}
        return reverse_mapping[label]
    # def get_semantic_label(self, label):
    #     # Get the correct label mapping from ImageFolder's class_to_idx
    #     class_to_idx = self.train_loader.dataset.class_to_idx
    #     reverse_mapping = {v: k for k, v in class_to_idx.items()}  # reverse the mapping
    #     return reverse_mapping[label]

class DogCatDataset:
    """
    Cat vs. Dog Dataset.
    """
    def __init__(self, batch_size=4, dataset_path='data/images/dogs_vs_cats'):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.transform = self.get_transforms()
        self.train_loader, self.val_loader = self.get_dataloaders()

    def get_transforms(self):
        # TODO (part g): fill in the data transforms
        transform_list = [
            # resize the image to 256x256x3
            # crop the image at the center of size 224x224x3
            # convert image to PyTorch tensor
            # normalize the image
        ]
        transform = transforms.Compose(transform_list)
        return transform

    def get_dataloaders(self):
        # train set
        train_set = torchvision.datasets.ImageFolder(os.path.join(self.dataset_path, 'train'), transform=self.transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        # validation set
        val_set = torchvision.datasets.ImageFolder(os.path.join(self.dataset_path, 'val'), transform=self.transform)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader


if __name__ == '__main__':
    dataset = DogDataset()
    print(dataset.x_mean, dataset.x_std)
    images, labels = next(iter(dataset.train_loader))
    dataset.plot_image(
        torchvision.utils.make_grid(images),
        ', '.join([dataset.get_semantic_label(label.item()) for label in labels])
    )