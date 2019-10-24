import torch.nn as nn
from inferno.extensions.layers.convolutional import ConvELU3D


class ResBlock(nn.Module):
    """
    General ResBlock with the option to have operations before and after and on the residual connection.
    """
    def __init__(self, main, pre=None, post=None, skip=None):
        super(ResBlock, self).__init__()
        self.main = main
        self.pre = pre
        self.post = post
        self.skip = skip

    def forward(self, x):
        if self.pre is not None:
            x = self.pre(x)
        if hasattr(self, 'outer') and self.skip is not None:
            skip = self.skip(x)
        else:
            skip = x
        x = skip + self.inner(x)
        if self.post is not None:
            x = self.post(x)
        return x


class StandardResBlock(ResBlock):
    """
    ResBlock as used in the vanilla ResNet
    """
    # TODO


class ValidPadResBlock(ResBlock):
    """
    Residual block with valid padding.
    """
    def __init__(self, f_in, f_main=None, kernel_size=1, conv_type=nn.Conv3d, activation='ELU'):
        f_main = f_in if f_main is None else f_main
        if isinstance(activation, str):
            activation = getattr(nn, activation)()
        main = nn.Sequential(
            conv_type(f_in, f_main, kernel_size=1),
            activation,
            conv_type(f_main, f_main, kernel_size=kernel_size, padding=0),
            activation,
            conv_type(f_main, f_in, kernel_size=1),
        )
        self.crop = (kernel_size - 1)//2
        super(ValidPadResBlock, self).__init__(main=main, outer=self.skip)

    def skip(self, x):
        crop = self.crop
        if crop == 0:
            return x
        else:
            return x[:, :, slice(crop, -crop), slice(crop, -crop), slice(crop, -crop)]


class SuperhumanSNEMIBlock(ResBlock):
    def __init__(self, f_in, f_main=None, f_out=None,
                 pre_kernel_size=(1, 3, 3), inner_kernel_size=(3, 3, 3),
                 conv_type=ConvELU3D):
        if f_main is None:
            f_main = f_in
        if f_out is None:
            f_out = f_main
        pre = conv_type(f_in, f_out, kernel_size=pre_kernel_size)
        skip = nn.Sequential(conv_type(f_out, f_main, kernel_size=inner_kernel_size),
                             conv_type(f_main, f_out, kernel_size=inner_kernel_size))
        super(SuperhumanSNEMIBlock, self).__init__(pre=pre, main=skip)
