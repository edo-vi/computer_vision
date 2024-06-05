import gin
import torch
import lightning
from DenoisingAE import DenoisingAE

gin.parse_config_file("config.cfg")

ae = DenoisingAE()
