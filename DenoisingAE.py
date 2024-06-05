import gin
from torch import nn
import torchvision.transforms


@gin.configurable
class DenoisingAE(nn.Module):
    def __init__(self, input_c=3, input_w=676, input_h=380) -> None:
        super().__init__()
        self.input_w = input_w
        self.input_h = input_h

        self.encoder_stack = nn.Sequential(
            nn.Conv2d(input_c, 6, (4, 4), stride=2),
            nn.Conv2d(6, 12, (4, 4), stride=2),
            nn.Conv2d(12, 24, (4, 4), stride=2),
            # nn.Conv2d(24, 48, (4, 4), stride=2),
        )

        self.decoder_stack = nn.Sequential(
            # nn.ConvTranspose2d(48, 24, (4, 4), stride=2),
            nn.ConvTranspose2d(24, 12, (4, 4), stride=2),
            nn.ConvTranspose2d(12, 6, (4, 4), stride=2),
            nn.ConvTranspose2d(6, input_c, (4, 4), stride=2),
        )
        print(self.encoder_stack, self.decoder_stack)

    def forward(self, x):
        pool = nn.MaxPool2d((2, 2), return_indices=True)
        unpool = nn.MaxUnpool2d((2, 2))
        transformer = torchvision.transforms.Resize(x.shape[1:])
        print(f"Input shape: {x.shape}")
        enc = self.encoder_stack(x.float())
        z, ind = pool(enc)
        unpooled = unpool(z, ind, output_size=(enc.shape[1], enc.shape[2]))
        print(f"Z shape: {z.shape}")
        dec = transformer(self.decoder_stack(unpooled))
        print(f"Decoded shape: {dec.shape}")
        return dec

    def params(self):
        return self.encoder_stack.parameters()
