import gin
from torch import nn
import torchvision.transforms
from lightning import LightningModule
import torch.nn.functional as F
from torch.optim import AdamW

WIDTH = 640


@gin.configurable
class DenoisingAE(LightningModule):
    def __init__(self, input_c=3, input_w=676, input_h=380) -> None:
        super().__init__()
        self.input_w = input_w
        self.input_h = input_h

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

        print(str(self.enc_1) + "\n" + str(self.enc_2) + "\n" + str(self.enc_3) + "\n")
        print(str(self.dec_1) + "\n" + str(self.dec_2) + "\n" + str(self.dec_3) + "\n")

    def forward(self, x):
        f = torchvision.transforms.Resize((WIDTH, WIDTH))
        # pool = nn.MaxPool2d((2, 2), return_indices=True)

        # unpool = nn.MaxUnpool2d((2, 2))
        # print(f"Input shape: {x.shape}")

        z = F.relu(self.enc_3(F.relu(self.enc_2(F.relu(self.enc_1(x))))))
        # print(f"Z shape: {z.shape}")

        dec = F.relu(self.dec_3(F.relu(self.dec_2(F.relu(self.dec_1(z))))))

        # print(f"Decoded shape: {dec.shape}")
        return f(dec)

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

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.01)

    def parameters(self):
        return [l.weight for l in self.layers]
