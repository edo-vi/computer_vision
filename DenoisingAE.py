import gin
from torch import nn


@gin.configurable
class DenoisingAE(nn.Module):
    def __init__(self, nlayers=32, d=8) -> None:
        # Must be even, because the number of layers
        # is the full depth, encoder + decoder, so
        # any of the two has nlayers/2 layers
        assert nlayers % 2 == 0
        self.nlayers = nlayers
        self.d = d  # width of the latent space
        print(nlayers, d)

    def forward(self, x):
        self.stack(x)

    def params(self):
        return self.stack.params()
