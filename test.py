import gin
import torch
import lightning
import numpy as np
from DenoisingAE import DenoisingAE
from DataLoader import gaussian_noise_transform, show_image, ObjectDetectionDataset

np.random.seed(100)

gin.parse_config_file("config.cfg")

ae = DenoisingAE()
transform = gaussian_noise_transform()
print(transform)

dataset = ObjectDetectionDataset()

show_image(dataset)