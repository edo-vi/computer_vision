import gin

# from wandb.integration.ultralytics import add_wandb_callback
from Trainer import trainDenoisingAETrainer
import numpy as np
from DenoisingAE import DenoisingAE
from DataLoader import show_pair, show_image, AfricanWildlifeDataset
import torch
from ultralytics import YOLO

model = YOLO()

# model.model = torch.compile(model.model)

np.random.seed(100)

gin.parse_config_file("config.cfg")


dataset = AfricanWildlifeDataset()

show_image(dataset)

# add_wandb_callback(model, enable_model_checkpointing=True)
# results = model.train(
#     data="african_wildlife.yaml",
#     epochs=20,
#     imgsz=640,
#     # project="aiproject",
# )

# print(results)

img, label = dataset[1000]
ae = DenoisingAE()
decoded = ae(img)

show_pair(img, decoded)
