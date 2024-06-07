import gin

# from wandb.integration.ultralytics import add_wandb_callback
from Trainer import trainDenoisingAETrainer
import numpy as np
from DenoisingAE import DenoisingAE
from DataLoader import show_pair, show_image, AfricanWildlifeDataset

from ultralytics import YOLO

model = YOLO()

# model.model = torch.compile(model.model)

np.random.seed(100)

gin.parse_config_file("config.cfg")


dataset = AfricanWildlifeDataset()
