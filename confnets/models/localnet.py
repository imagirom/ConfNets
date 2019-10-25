import torch.nn as nn
from confnets.blocks import ValidPadResBlock


class LocalNet(nn.Module):
    """
    Architecture that with a very limited FOV. Inspired by BagNets (https://arxiv.org/abs/1904.00760).
    """
    def __init__(self, kernel_sizes=(3, 1, 1, 3, 1, 1), fmaps=(1, 16, 16, 16, 16, 16), bottleneck_factor=4):
        super(LocalNet, self).__init__()
        assert len(fmaps) == len(kernel_sizes), f'len({fmaps}) != len({kernel_sizes})'
        self.first_layer = nn.Conv3d(fmaps[0], fmaps[1], kernel_sizes[0], padding=0)
        self.blocks = nn.ModuleList([ValidPadResBlock(in_channels=f_in, main_channels=f_in // bottleneck_factor, kernel_size=k)
                                     for f_in, k in zip(fmaps[1:], kernel_sizes[1:])])
        self.fov = 1 + sum([k - 1 for k in kernel_sizes])
        print(f'FOV of the local net: {self.fov}')

    def forward(self, x):
        x = self.first_layer(x)
        for block in self.blocks:
            x = block(x)
        return x
