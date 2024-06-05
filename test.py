import gin
from wandb.integration.ultralytics import add_wandb_callback
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
add_wandb_callback(model, enable_model_checkpointing=True)
results = model.train(
    data="african_wildlife.yaml",
    epochs=20,
    imgsz=640,
    project="aiproject",
)

# print(results)

ae = DenoisingAE()
ae(img)
