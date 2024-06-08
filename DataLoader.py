from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
import torch
import gin
import os
import matplotlib.pyplot as plt
import numpy as np

WIDTH = 640


@gin.configurable
class AfricanWildlifeDataset(Dataset):
    def __init__(self, classes=4, transform=None, kind="train") -> None:
        super().__init__()
        assert kind in ["train", "test", "valid"]
        self.kind = kind
        self.img_path = os.path.join("datasets", "wildlife", kind, "images")
        self.list_dir = [a for a in os.listdir(self.img_path)]
        self.image_dimensions = [
            (-1, -1) for _ in range(len(self.list_dir))
        ]  # lazy evaluation, only when __getitem__
        self.classes = classes
        self.normalize = normalize_transform()
        if transform:
            try:
                self.transform = transform()
            except:
                self.transform = transform
        else:
            self.transform = transform

    def __len__(self):
        return len(self.list_dir)

    def __getitem__(self, index):
        name = self.list_dir[index]
        path = os.path.join(self.img_path, name)
        img = read_image(path)
        # Add the dimension to the list
        self.image_dimensions[index] = img.shape
        if self.transform is None:
            return (
                self.normalize(img),
                self.normalize(img),
            )
        else:
            # Return the original image as label
            return (
                self.transform(img),
                self.normalize(img),
            )


@gin.configurable
def gaussian_noise_transform(mu=0.0, sigma=0.2, width=None):
    if width:
        return transforms.Compose(
            [
                transforms.Resize(
                    (WIDTH, WIDTH)
                ),  # To make it perfectly match the enc -> dec
                transforms.Lambda(lambda x: x / x.max().item()),  # From 0 to 1
                GaussianNoise(mu, sigma),
                transforms.Lambda(
                    lambda x: (x - x.min().item()) / (x.max().item() - x.min().item())
                ),  # Renormalize from 0 to 1 because
                # gaussian noise could have changed the range
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Lambda(lambda x: x / x.max().item()),  # From 0 to 1
                GaussianNoise(mu, sigma),
                transforms.Lambda(
                    lambda x: (x - x.min().item()) / (x.max().item() - x.min().item())
                ),  # Renormalize from 0 to 1 because
                # gaussian noise could have changed the range
            ]
        )


def normalize_transform(width=640):
    return transforms.Compose(
        [
            transforms.Resize((WIDTH, WIDTH)),
            transforms.Lambda(lambda x: x / x.max().item()),
        ]
    )


class GaussianNoise(object):
    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        np.random.seed(314)  # to fix
        noise = torch.normal(self.mu, self.sigma, x.shape)
        x += noise

        return x

    def __repr__(self):
        return self.__class__.__name__ + f"({self.mu}, {self.sigma})"


@gin.configurable
def bernoulli_noise_transform(p=0.1, width=None):
    if width:
        return transforms.Compose(
            [
                transforms.Resize(
                    (WIDTH, WIDTH)
                ),  # To make it perfectly match the enc -> dec
                transforms.Lambda(lambda x: x / x.max().item()),  # From 0 to 1
                BernoulliNoise(p),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Lambda(lambda x: x / x.max().item()),  # From 0 to 1
                BernoulliNoise(p),
                transforms.Lambda(
                    lambda x: (x - x.min().item()) / (x.max().item() - x.min().item())
                ),  # Renormalize from 0 to 1 because
                # gaussian noise could have changed the range
            ]
        )


class BernoulliNoise(object):
    def __init__(self, p=0.1) -> None:
        self.p = p

    def __call__(self, x) -> torch.Any:
        np.random.seed(314)
        mask0 = np.where(torch.rand((x.shape[1], x.shape[2])) < self.p / 2, True, False)
        mask0 = np.stack([mask0, mask0, mask0])
        mask1 = np.where(torch.rand((x.shape[1], x.shape[2])) < self.p / 2, True, False)
        mask1 = np.stack([mask1, mask1, mask1])
        x[mask0] = 0.0
        x[mask1] = 1.0
        return x


def show_image(data):
    figure = plt.figure(figsize=(12, 12))
    cols, rows = 2, 1
    for i in range(1, int((cols * rows) / 2) + 1):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        img, label = data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.axis("off")
        try:
            plt.imshow(np.transpose(label, [1, 2, 0]))
        except:
            im = img.detach().numpy()
            plt.imshow(np.transpose(label, [1, 2, 0]))

        figure.add_subplot(rows, cols, i + 1)
        plt.axis("off")
        try:
            plt.imshow(np.transpose(img, [1, 2, 0]))
        except:
            im = img.detach().numpy()
            plt.imshow(np.transpose(im, [1, 2, 0]))

    plt.show()


def show_pair(im1, im2):
    figure = plt.figure(figsize=(12, 12))
    cols, rows = 2, 1
    for i in range(1, int((cols * rows) / 2) + 1):

        figure.add_subplot(rows, cols, i)
        plt.axis("off")
        try:
            plt.imshow(np.transpose(im1, [1, 2, 0]))
        except:
            im = im1.detach().numpy()
            plt.imshow(np.transpose(im1, [1, 2, 0]))

        figure.add_subplot(rows, cols, i + 1)
        plt.axis("off")
        try:
            plt.imshow(np.transpose(im2, [1, 2, 0]))
        except:
            im = im2.detach().numpy()
            plt.imshow(np.transpose(im, [1, 2, 0]))

    plt.show()


def show_triple(im1, im2, im3, tags=None):
    figure = plt.figure(figsize=(15, 15))
    cols, rows = 3, 1

    figure.add_subplot(rows, cols, 1)
    plt.axis("off")
    if tags:
        plt.title(tags[0])
    try:
        plt.imshow(np.transpose(im1, [1, 2, 0]))
    except:
        im = im1.detach().numpy()
        plt.imshow(np.transpose(im1, [1, 2, 0]))

    figure.add_subplot(rows, cols, 2)
    plt.axis("off")
    if tags:
        plt.title(tags[1])
    try:
        plt.imshow(np.transpose(im2, [1, 2, 0]))
    except:
        im = im2.detach().numpy()
        plt.imshow(np.transpose(im, [1, 2, 0]))

    figure.add_subplot(rows, cols, 3)
    plt.axis("off")
    if tags:
        plt.title(tags[2])
    try:
        plt.imshow(np.transpose(im3, [1, 2, 0]))
    except:
        im = im3.detach().numpy()
        plt.imshow(np.transpose(im, [1, 2, 0]))
    plt.show()
