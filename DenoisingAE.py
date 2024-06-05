import gin
from torch import nn
import torch

@gin.configurable
class DenoisingAE(nn.Module):
    def __init__(self, nlayers=32, d=8, input_c = 3, input_w = 676, input_h = 380) -> None:
        super().__init__()
        # Must be even, because the number of layers
        # is the full depth, encoder + decoder, so
        # any of the two has nlayers/2 layers
        self.input_w = input_w
        self.input_h = input_h
        assert nlayers % 2 == 0
        self.nlayers = nlayers
        self.d = d  # dimension of the latent space

        self.encoder_stack = nn.Sequential(
            nn.Conv2d(input_c, 6,  (3, 3), stride=2),
            nn.Conv2d(6, 12,  (3, 3), stride=2),
            nn.Conv2d(12, 24,  (3, 3), stride=2),
        )

        self.decoder_stack = nn.Sequential(
            nn.ConvTranspose2d(24, 12, (3,3), stride=2),
            nn.ConvTranspose2d(12, 6, (3,3), stride=2),
            nn.ConvTranspose2d(6, input_c, (3,3), stride=2),
        )
        print(nlayers, d, self.encoder_stack, self.decoder_stack)

    def forward(self, x):
        pool = nn.MaxPool2d(1, return_indices=True)
        unpool = nn.MaxUnpool2d(1)
        print(f"Input shape: {x.shape}")
        enc = self.encoder_stack(x.float())
        z, ind = pool(enc)
        unpooled = unpool(z, ind, output_size=(enc.shape[1],enc.shape[2]))
        print(f"Z shape: {z.shape}")
        dec = self.decoder_stack(unpooled)
        print(f"Decoded shape: {dec.shape}")
        return dec
        

    def params(self):
        return self.encoder_stack.parameters()
