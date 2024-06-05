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

# show_image(dataset)

img = dataset[65]

results = model.train(
    data="african_wildlife.yaml", epochs=2, imgsz=640, project="aiproject", name="t"
)

# print(results)

ae = DenoisingAE()
ae(img)
