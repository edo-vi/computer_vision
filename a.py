import gin
from torch import nn
import torchvision.transforms
from lightning import LightningModule
import torch.nn.functional as F
from torch.optim import AdamW

WIDTH = 640


@gin.configurable
class DenoisingAE(LightningModule):
    def __init__(self, input_c=3) -> None:
        super().__init__()

        self.enc_1 = nn.Conv2d(input_c, 6, (4, 4), stride=2)
        self.enc_2 = nn.Conv2d(6, 12, (4, 4), stride=2)
        self.enc_3 = nn.Conv2d(12, 24, (4, 4), stride=2)

        self.dec_1 = nn.ConvTranspose2d(24, 12, (4, 4), stride=2)
        self.dec_2 = nn.ConvTranspose2d(12, 6, (4, 4), stride=2)
        self.dec_3 = nn.ConvTranspose2d(6, input_c, (4, 4), stride=2)

        self.layers = [
            self.enc_1,
            self.enc_2,
            self.enc_3,
            self.dec_1,
            self.dec_2,
            self.dec_3,
        ]

        print([str(l) for l in self.layers])

    def forward(self, x):
        f = torchvision.transforms.Resize((WIDTH, WIDTH))
        # pool = nn.MaxPool2d((2, 2), return_indices=True)

        # unpool = nn.MaxUnpool2d((2, 2))
        # print(f"Input shape: {x.shape}")

        x1 = F.dropout(F.relu(self.enc_1(x)), p=0.2)
        x2 = F.dropout(F.relu(self.enc_2(x1)), p=0.2)
        z = F.relu(self.enc_3(x2))
        # print(f"Z shape: {z.shape}")

        y3 = F.dropout(F.relu(self.dec_1(z)), p=0.2)
        y2 = F.dropout(F.relu(self.dec_2(y3)), p=0.2)
        dec = F.relu(self.dec_3(y2))
        # print(f"Decoded shape: {dec.shape}")
        return f(dec)  # resize

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = F.mse_loss(output, target)
        self.log_dict({"train_loss": loss})

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = F.mse_loss(output, target)
        self.log_dict({"val_loss": loss})

        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.01)

    def parameters(self):
        return [l.weight for l in self.layers]
