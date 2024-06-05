import gin
import torch
import lightning
import numpy as np
from DenoisingAE import DenoisingAE
from DataLoader import gaussian_noise_transform, show_image, ObjectDetectionDataset
import torch
from ultralytics import YOLO

model = YOLO()

# model.model = torch.compile(model.model)

np.random.seed(100)

gin.parse_config_file("config.cfg")


dataset = ObjectDetectionDataset()

show_image(dataset)

img = dataset[13]

results = model(torch.tensor(img).unsqueeze(0).float())

# print(results)

ae = DenoisingAE()
ae(img)
