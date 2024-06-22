import gin
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from lightning import LightningModule
import torch.nn.functional as F
from torch.optim import SGD, AdamW
import torch

WIDTH = 320

"""The Resnet denoising model"""


class DenoisingResnet(LightningModule):
    def __init__(self, freeze=True) -> None:
        super().__init__()
        self.kind = "resnet"
        # Load pre-trained weights
        self.model = resnet18(ResNet18_Weights)
        # Change head
        self.model.fc = nn.Linear(512, out_features=3 * WIDTH * WIDTH)
        # Freeze intermediate layers
        if freeze:
            self._freeze()

    def _freeze(self):
        self.model.conv1.requires_grad_ = False
        for i in range(1, 5):
            self.model.get_submodule(f"layer{i}").requires_grad_ = False

    def forward(self, x):
        # to make the check correct
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, 0)
        if x.shape[1:] != (3, WIDTH, WIDTH):
            raise Exception(f"Only 3x{WIDTH}x{WIDTH} images are supported")
        return F.relu(self.model.forward(x)).reshape((-1, 3, WIDTH, WIDTH))

    def training_step(self, batch, batch_idx):
        # inputs are the noisy images, targets are the original images
        inputs, targets = batch
        output = self(inputs)
        loss = F.mse_loss(output, targets)
        # Log metric to logger (tensorboard or wandb etc.)
        self.log_dict({"train_loss": loss})

        return loss

    def validation_step(self, batch, batch_idx):
        # inputs are the noisy images, targets are the original images
        inputs, target2 = batch
        output = self(inputs)
        loss = F.mse_loss(output, target2)
        # Log metric to logger (tensorboard or wandb etc.)
        self.log_dict({"val_loss": loss})

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.01)

    def parameters(self):
        return self.model.parameters()
