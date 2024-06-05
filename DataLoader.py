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
    def __init__(self, transform=None) -> None:
        super().__init__()
        self.img_path = "data"
        self.transform = transform()

    def __len__(self):
        return len([a for a in os.listdir(self.img_path)])


    def __getitem__(self, index):
        path = os.path.join(self.img_path, f"{index}.jpg")
        if self.transform is None:
            return read_image(path)
        else:
            return self.transform(read_image(path))


@gin.configurable
def gaussian_noise_transform(mu=1.0, sigma=0.1):
    return transforms.Compose([
        transforms.Resize((375, 671)), # To make it perfectly match the enc -> dec 
        GaussianNoise(mu, sigma),
    ])


class GaussianNoise(object):
    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma
        
    def __call__(self, x):
        # Independent gaussian noise for each channel AND each image.
        # This should make it harder.
        x[0,:,:] += np.random.normal(self.mu, self.sigma, (x.shape[1], x.shape[2]))
        x[1,:,:] += np.random.normal(self.mu, self.sigma, (x.shape[1], x.shape[2]))
        x[2,:,:] += np.random.normal(self.mu, self.sigma, (x.shape[1], x.shape[2]))
        return x
    
    def __repr__(self):
        return self.__class__.__name__ + f"({self.mu}, {self.sigma})"


def show_image(data):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 1, 1
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        img = data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.axis("off")
        # Need to transpose because the images have shape [C, W, H]
        plt.imshow(np.transpose(img, [1, 2, 0])) 
    plt .show()

