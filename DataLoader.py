from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
import torch
import gin
import os
import matplotlib.pyplot as plt
import numpy as np


@gin.configurable
class ObjectDetectionDataset(Dataset):
    def __init__(self, classes=4, transform=None) -> None:
        super().__init__()
        self.img_path = "datasets/african_wildlife"
        self.classes = classes
        self.transform = transform()

    def __len__(self):
        return len([a for a in os.listdir(self.img_path)])

    def __getitem__(self, index):
        cls = 4
        path = os.path.join(self.img_path, "train", "images", f"{cls} ({index}).jpg")
        if self.transform is None:
            return read_image(path)
        else:
            return self.transform(read_image(path))


@gin.configurable
def gaussian_noise_transform(mu=0.0, sigma=0.2):
    return transforms.Compose(
        [
            # transforms.Resize(640),  # To make it perfectly match the enc -> dec
            transforms.Lambda(lambda x: x / x.max().item()),  # From 0 to 1
            GaussianNoise(mu, sigma),
        ]
    )


class GaussianNoise(object):
    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        # Independent gaussian noise for each channel AND each image.
        # This should make it harder.
        x[0, :, :] = np.clip(
            x[0, :, :]
            + np.random.normal(self.mu, self.sigma, (x.shape[1], x.shape[2])),
            0.0,
            1.0,
        )
        x[1, :, :] = np.clip(
            x[1, :, :]
            + np.random.normal(self.mu, self.sigma, (x.shape[1], x.shape[2])),
            0.0,
            1.0,
        )
        x[2, :, :] = np.clip(
            x[2, :, :]
            + np.random.normal(self.mu, self.sigma, (x.shape[1], x.shape[2])),
            0.0,
            1.0,
        )
        return x

    def __repr__(self):
        return self.__class__.__name__ + f"({self.mu}, {self.sigma})"


def show_image(data):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 2, 2
    for i in range(1, cols * rows + 1):
        sample_idx = [351, 352, 353, 358]
        for j in sample_idx:
            img = data[j]
            figure.add_subplot(rows, cols, i)
            plt.axis("off")
            # Need to transpose because the images have shape [C, W, H]
            plt.imshow(np.transpose(img, [1, 2, 0]))
    plt.show()
