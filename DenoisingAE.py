import gin
from torch import nn
import torchvision.transforms
from lightning import LightningModule
import torch.nn.functional as F
from torch.optim import SGD, AdamW

WIDTH = 640


@gin.configurable
class DenoisingAE(LightningModule):
    def __init__(self, kind="v1") -> None:
        super().__init__()
        self.kind = kind

        if self.kind == "v0":
            self.enc_1 = nn.Conv2d(3, 6, (4, 4), stride=2)
            self.enc_2 = nn.Conv2d(6, 12, (4, 4), stride=2)
            self.dec_1 = nn.ConvTranspose2d(12, 6, (4, 4), stride=2)
            self.dec_2 = nn.ConvTranspose2d(6, 3, (4, 4), stride=2)

            self.layers = [
                self.enc_1,
                self.enc_2,
                self.dec_1,
                self.dec_2,
            ]
            # print(str(self.enc_1) + "\n" + str(self.enc_2) + "\n")
            # print(str(self.dec_1) + "\n" + str(self.dec_2) + "\n")

        elif self.kind == "v1":
            self.enc_1 = nn.Conv2d(3, 6, (4, 4), stride=2)
            self.enc_2 = nn.Conv2d(6, 12, (4, 4), stride=2)
            self.enc_3 = nn.Conv2d(12, 24, (4, 4), stride=2)

            self.dec_1 = nn.ConvTranspose2d(24, 12, (4, 4), stride=2)
            self.dec_2 = nn.ConvTranspose2d(12, 6, (4, 4), stride=2)
            self.dec_3 = nn.ConvTranspose2d(6, 3, (4, 4), stride=2)

            self.layers = [
                self.enc_1,
                self.enc_2,
                self.enc_3,
                self.dec_1,
                self.dec_2,
                self.dec_3,
            ]
            # print(
            #     str(self.enc_1) + "\n" + str(self.enc_2) + "\n" + str(self.enc_3) + "\n"
            # )
            # print(
            #     str(self.dec_1) + "\n" + str(self.dec_2) + "\n" + str(self.dec_3) + "\n"
            # )
        elif self.kind == "v2":
            self.enc_1 = nn.Conv2d(3, 6, (4, 4), stride=2)
            self.enc_2 = nn.Conv2d(6, 12, (4, 4), stride=2)
            self.enc_3 = nn.Conv2d(12, 24, (4, 4), stride=2)
            self.enc_4 = nn.Conv2d(24, 48, (4, 4), stride=2)

            self.dec_1 = nn.ConvTranspose2d(48, 24, (4, 4), stride=2)
            self.dec_2 = nn.ConvTranspose2d(24, 12, (4, 4), stride=2)
            self.dec_3 = nn.ConvTranspose2d(12, 6, (4, 4), stride=2)
            self.dec_4 = nn.ConvTranspose2d(6, 3, (4, 4), stride=2)

            self.layers = [
                self.enc_1,
                self.enc_2,
                self.enc_3,
                self.enc_4,
                self.dec_1,
                self.dec_2,
                self.dec_3,
                self.dec_4,
            ]
            # print(
            #     str(self.enc_1)
            #     + "\n"
            #     + str(self.enc_2)
            #     + "\n"
            #     + str(self.enc_3)
            #     + "\n"
            #     + str(self.enc_4)
            #     + "\n"
            # )
            # print(
            #     str(self.dec_1)
            #     + "\n"
            #     + str(self.dec_2)
            #     + "\n"
            #     + str(self.dec_3)
            #     + "\n"
            #     + str(self.dec_4)
            #     + "\n"
            # )

    def forward(self, x):
        f = torchvision.transforms.Resize((WIDTH, WIDTH))
        z = F.relu(self.enc_2(F.relu(self.enc_1(x))))
        if self.kind == "v1":  # Need to add enc_4
            z = F.relu(self.enc_3(z))
        elif self.kind == "v2":
            z = F.relu(self.enc_4(F.relu(self.enc_3(z))))
        dec = F.relu(self.dec_2(F.relu(self.dec_1(z))))
        if self.kind == "v1":  # Need to add dec_4
            dec = F.relu(self.dec_3(dec))
        elif self.kind == "v2":
            dec = F.relu(self.dec_4(F.relu(self.dec_3(dec))))

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


class DenoisingAEV1(LightningModule):
    def __init__(self, kind="v1") -> None:
        super().__init__()
        self.kind = kind

        if self.kind == "v0":
            self.enc_1 = nn.Conv2d(3, 6, (4, 4), stride=2)
            self.enc_2 = nn.Conv2d(6, 12, (4, 4), stride=2)
            self.dec_1 = nn.ConvTranspose2d(12, 6, (4, 4), stride=2)
            self.dec_2 = nn.ConvTranspose2d(6, 3, (4, 4), stride=2)

            self.layers = [
                self.enc_1,
                self.enc_2,
                self.dec_1,
                self.dec_2,
            ]
            # print(str(self.enc_1) + "\n" + str(self.enc_2) + "\n")
            # print(str(self.dec_1) + "\n" + str(self.dec_2) + "\n")

        elif self.kind == "v1":
            self.enc_1 = nn.Conv2d(3, 6, (4, 4), stride=2)
            self.enc_2 = nn.Conv2d(6, 12, (4, 4), stride=2)
            self.enc_3 = nn.Conv2d(12, 24, (4, 4), stride=2)

            self.dec_1 = nn.ConvTranspose2d(24, 12, (4, 4), stride=2)
            self.dec_2 = nn.ConvTranspose2d(12, 6, (4, 4), stride=2)
            self.dec_3 = nn.ConvTranspose2d(6, 3, (4, 4), stride=2)

            self.layers = [
                self.enc_1,
                self.enc_2,
                self.enc_3,
                self.dec_1,
                self.dec_2,
                self.dec_3,
            ]
            # print(
            #     str(self.enc_1) + "\n" + str(self.enc_2) + "\n" + str(self.enc_3) + "\n"
            # )
            # print(
            #     str(self.dec_1) + "\n" + str(self.dec_2) + "\n" + str(self.dec_3) + "\n"
            # )
        elif self.kind == "v2":
            self.enc_1 = nn.Conv2d(3, 6, (4, 4), stride=2)
            self.enc_2 = nn.Conv2d(6, 12, (4, 4), stride=2)
            self.enc_3 = nn.Conv2d(12, 24, (4, 4), stride=2)
            self.enc_4 = nn.Conv2d(24, 48, (4, 4), stride=2)

            self.dec_1 = nn.ConvTranspose2d(48, 24, (4, 4), stride=2)
            self.dec_2 = nn.ConvTranspose2d(24, 12, (4, 4), stride=2)
            self.dec_3 = nn.ConvTranspose2d(12, 6, (4, 4), stride=2)
            self.dec_4 = nn.ConvTranspose2d(6, 3, (4, 4), stride=2)

            self.layers = [
                self.enc_1,
                self.enc_2,
                self.enc_3,
                self.enc_4,
                self.dec_1,
                self.dec_2,
                self.dec_3,
                self.dec_4,
            ]
            # print(
            #     str(self.enc_1)
            #     + "\n"
            #     + str(self.enc_2)
            #     + "\n"
            #     + str(self.enc_3)
            #     + "\n"
            #     + str(self.enc_4)
            #     + "\n"
            # )
            # print(
            #     str(self.dec_1)
            #     + "\n"
            #     + str(self.dec_2)
            #     + "\n"
            #     + str(self.dec_3)
            #     + "\n"
            #     + str(self.dec_4)
            #     + "\n"
            # )

    def forward(self, x):
        f = torchvision.transforms.Resize((WIDTH, WIDTH))
        z = F.relu(self.enc_2(F.relu(self.enc_1(x))))
        if self.kind == "v1":  # Need to add enc_4
            z = F.relu(self.enc_3(z))
        elif self.kind == "v2":
            z = F.relu(self.enc_4(F.relu(self.enc_3(z))))
        dec = F.relu(self.dec_2(F.relu(self.dec_1(z))))
        if self.kind == "v1":  # Need to add dec_4
            dec = F.relu(self.dec_3(dec))
        elif self.kind == "v2":
            dec = F.relu(self.dec_4(F.relu(self.dec_3(dec))))

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


class DenoisingAEV2(LightningModule):
    def __init__(self, kind="v2") -> None:
        super().__init__()
        self.kind = kind

        if self.kind == "v0":
            self.enc_1 = nn.Conv2d(3, 6, (4, 4), stride=2)
            self.enc_2 = nn.Conv2d(6, 12, (4, 4), stride=2)
            self.dec_1 = nn.ConvTranspose2d(12, 6, (4, 4), stride=2)
            self.dec_2 = nn.ConvTranspose2d(6, 3, (4, 4), stride=2)

            self.layers = [
                self.enc_1,
                self.enc_2,
                self.dec_1,
                self.dec_2,
            ]
            # print(str(self.enc_1) + "\n" + str(self.enc_2) + "\n")
            # print(str(self.dec_1) + "\n" + str(self.dec_2) + "\n")

        elif self.kind == "v1":
            self.enc_1 = nn.Conv2d(3, 6, (4, 4), stride=2)
            self.enc_2 = nn.Conv2d(6, 12, (4, 4), stride=2)
            self.enc_3 = nn.Conv2d(12, 24, (4, 4), stride=2)

            self.dec_1 = nn.ConvTranspose2d(24, 12, (4, 4), stride=2)
            self.dec_2 = nn.ConvTranspose2d(12, 6, (4, 4), stride=2)
            self.dec_3 = nn.ConvTranspose2d(6, 3, (4, 4), stride=2)

            self.layers = [
                self.enc_1,
                self.enc_2,
                self.enc_3,
                self.dec_1,
                self.dec_2,
                self.dec_3,
            ]
            # print(
            #     str(self.enc_1) + "\n" + str(self.enc_2) + "\n" + str(self.enc_3) + "\n"
            # )
            # print(
            #     str(self.dec_1) + "\n" + str(self.dec_2) + "\n" + str(self.dec_3) + "\n"
            # )
        elif self.kind == "v2":
            self.enc_1 = nn.Conv2d(3, 6, (4, 4), stride=2)
            self.enc_2 = nn.Conv2d(6, 12, (4, 4), stride=2)
            self.enc_3 = nn.Conv2d(12, 24, (4, 4), stride=2)
            self.enc_4 = nn.Conv2d(24, 48, (4, 4), stride=2)

            self.dec_1 = nn.ConvTranspose2d(48, 24, (4, 4), stride=2)
            self.dec_2 = nn.ConvTranspose2d(24, 12, (4, 4), stride=2)
            self.dec_3 = nn.ConvTranspose2d(12, 6, (4, 4), stride=2)
            self.dec_4 = nn.ConvTranspose2d(6, 3, (4, 4), stride=2)

            self.layers = [
                self.enc_1,
                self.enc_2,
                self.enc_3,
                self.enc_4,
                self.dec_1,
                self.dec_2,
                self.dec_3,
                self.dec_4,
            ]
            # print(
            #     str(self.enc_1)
            #     + "\n"
            #     + str(self.enc_2)
            #     + "\n"
            #     + str(self.enc_3)
            #     + "\n"
            #     + str(self.enc_4)
            #     + "\n"
            # )
            # print(
            #     str(self.dec_1)
            #     + "\n"
            #     + str(self.dec_2)
            #     + "\n"
            #     + str(self.dec_3)
            #     + "\n"
            #     + str(self.dec_4)
            #     + "\n"
            # )

    def forward(self, x):
        f = torchvision.transforms.Resize((WIDTH, WIDTH))
        z = F.relu(self.enc_2(F.relu(self.enc_1(x))))
        if self.kind == "v1":  # Need to add enc_4
            z = F.relu(self.enc_3(z))
        elif self.kind == "v2":
            z = F.relu(self.enc_4(F.relu(self.enc_3(z))))
        dec = F.relu(self.dec_2(F.relu(self.dec_1(z))))
        if self.kind == "v1":  # Need to add dec_4
            dec = F.relu(self.dec_3(dec))
        elif self.kind == "v2":
            dec = F.relu(self.dec_4(F.relu(self.dec_3(dec))))

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
