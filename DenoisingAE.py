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
        assert nlayers % 2 == 0
        self.nlayers = nlayers
        self.d = d  # dimension of the latent space

        self.stack = nn.Sequential(
            nn.Conv2d(input_c, 6, (3,5))
        )
        print(nlayers, d, self.stack)

    def forward(self, x):
        return self.stack(x.float())

    def params(self):
        return self.stack.parameters()
