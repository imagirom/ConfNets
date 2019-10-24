import torch.nn as nn
from collections import OrderedDict
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
        if self.skip is not None:
            skip = self.skip(x)
        else:
            skip = x
        x = skip + self.main(x)
        if self.post is not None:
            x = self.post(x)
        return x


class BasicResBlock(ResBlock):
    """
    ResBlock as used in the vanilla ResNet.
    See https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L35.
    """
    # TODO: test this.
    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 conv_type=None, norm_type=None, activation=None,
                 pre_kernel_size=1, main_kernel_size=3):
        if norm_type is None:
            norm_type = nn.BatchNorm2d
        if conv_type is None:
            conv_type = nn.Conv2d
        if activation is None:
            activation = nn.ReLU(inplace=False)
        if downsample is None and (in_channels != out_channels or stride != 1):
            downsample = nn.Sequential(
                conv_type(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=1, stride=stride),
                norm_type(out_channels),
            )
        assert isinstance(main_kernel_size, int) and main_kernel_size % 2 == 1, \
            f'main_kernel_size must be odd. Got {main_kernel_size}.'
        super(BasicResBlock, self).__init__(
            main=nn.Sequential(OrderedDict([
                ('conv1', conv_type(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=pre_kernel_size, stride=stride)),
                ('norm1', norm_type(out_channels)),
                ('activation', activation),
                ('conv2', conv_type(in_channels=out_channels, out_channels=out_channels,
                                    kernel_size=main_kernel_size, padding=(main_kernel_size-1)//2)),
                ('norm2', norm_type(out_channels)),
                ])),
            skip=downsample,
            post=activation
        )


class BottleneckBlock(ResBlock):
    """
    BottleneckResBlock as used in the vanilla ResNet.
    See https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L76.
    """

    def __init__(self, in_channels, main_channels, out_channels, stride=1, groups=1, dilation=1, downsample=None,
                 conv_type=None, norm_type=None, activation=None):
        if norm_type is None:
            norm_type = nn.BatchNorm2d
        if conv_type is None:
            conv_type = nn.Conv2d
        if activation is None:
            activation = nn.ReLU(inplace=False)
        if downsample is None and in_channels != out_channels:
            downsample = nn.Sequential(
                conv_type(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=1, stride=stride),
                norm_type(out_channels),
            )

        super(BottleneckBlock, self).__init__(
            main=nn.Sequential(OrderedDict([
                ('conv1', conv_type(in_channels=in_channels, out_channels=main_channels,
                                    kernel_size=1)),
                ('norm1', norm_type(main_channels)),
                ('activation1', activation),
                ('conv2', conv_type(in_channels=main_channels, out_channels=main_channels,
                                    kernel_size=3, stride=stride, groups=groups, dilation=dilation, padding=dilation)),
                ('norm2', norm_type(main_channels)),
                ('activation2', activation),
                ('conv3', conv_type(in_channels=main_channels, out_channels=out_channels,
                                    kernel_size=1)),
                ('norm3', norm_type(out_channels))
            ])),
            skip=downsample,
            post=activation
        )


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
        super(ValidPadResBlock, self).__init__(main=main, skip=self.skip)

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
