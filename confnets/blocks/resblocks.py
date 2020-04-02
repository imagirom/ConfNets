import torch.nn as nn
from collections import OrderedDict
from ..utils import skip_none_sequential, get_padding
from ..layers import ConvNormActivation

class ResBlock(nn.Module):
    """
    General ResBlock with the option to have operations before and after and on the residual connection.
    The signature of derived classes should be (in_channels, out_channels, ..)
    """
    def __init__(self, main, pre=None, post=None, skip=None):
        super(ResBlock, self).__init__()
        self.pre = pre
        self.main = main
        self.skip = skip
        self.post = post

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
    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 conv_type=None, norm_type=None, activation=None,
                 pre_kernel_size=1, main_kernel_size=3):
        if norm_type is None:
            norm_type = nn.BatchNorm2d
        if conv_type is None:
            conv_type = nn.Conv2d
        if activation is None:
            activation = nn.ReLU
        if downsample is None and (in_channels != out_channels or stride != 1):
            downsample = skip_none_sequential(
                conv_type(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=1, stride=stride),
                norm_type(out_channels),
            )
        assert isinstance(main_kernel_size, int) and main_kernel_size % 2 == 1, \
            f'main_kernel_size must be odd. Got {main_kernel_size}.'
        if pre_kernel_size > 1:
            raise NotImplementedError
        super(BasicResBlock, self).__init__(
            main=skip_none_sequential(OrderedDict([
                ('conv1', conv_type(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=pre_kernel_size, stride=stride)),
                ('norm1', norm_type(out_channels)),
                ('activation', activation()),
                ('conv2', conv_type(in_channels=out_channels, out_channels=out_channels,
                                    kernel_size=main_kernel_size, padding=(main_kernel_size-1)//2)),
                ('norm2', norm_type(out_channels)),
                ])),
            skip=downsample,
            post=activation()
        )


class BottleneckBlock(ResBlock):
    """
    BottleneckResBlock as used in the vanilla ResNet.
    See https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L76.
    """

    def __init__(self, in_channels, out_channels, main_channels=None, expansion=4,
                 stride=1, groups=1, dilation=1, downsample=None,
                 conv_type=None, norm_type=None, activation=None):
        if norm_type is None:
            norm_type = nn.BatchNorm2d
        if conv_type is None:
            conv_type = nn.Conv2d
        if activation is None:
            activation = nn.ReLU
        if downsample is None and in_channels != out_channels:
            downsample = skip_none_sequential(
                conv_type(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=1, stride=stride),
                norm_type(out_channels),
            )
        if main_channels is None:
            main_channels = out_channels // expansion

        super(BottleneckBlock, self).__init__(
            main=skip_none_sequential(OrderedDict([
                ('conv1', conv_type(in_channels=in_channels, out_channels=main_channels,
                                    kernel_size=1)),
                ('norm1', norm_type(main_channels)),
                ('activation1', activation()),
                ('conv2', conv_type(in_channels=main_channels, out_channels=main_channels,
                                    kernel_size=3, stride=stride, groups=groups, dilation=dilation, padding=dilation)),
                ('norm2', norm_type(main_channels)),
                ('activation2', activation()),
                ('conv3', conv_type(in_channels=main_channels, out_channels=out_channels,
                                    kernel_size=1)),
                ('norm3', norm_type(out_channels))
            ])),
            skip=downsample,
            post=activation()
        )


class ConvActConvNormBlock(ResBlock):
    """
    See https://arxiv.org/pdf/1604.04112.pdf.
    """
    def __init__(self, in_channels, out_channels, main_channels=None, expansion=4,
                 stride=1, groups=1, dilation=1, downsample=None,
                 conv_type=None, norm_type=None, activation=None):
        if norm_type is None:
            norm_type = nn.BatchNorm2d
        if conv_type is None:
            conv_type = nn.Conv2d
        if activation is None:
            activation = nn.ELU
        if downsample is None and in_channels != out_channels:
            downsample = skip_none_sequential(
                conv_type(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=1, stride=stride),
                norm_type(out_channels),
            )
        if main_channels is None:
            main_channels = out_channels

        super(ConvActConvNormBlock, self).__init__(
            main=skip_none_sequential(OrderedDict([
                ('conv1', conv_type(in_channels=in_channels, out_channels=main_channels,
                                    kernel_size=1)),
                ('activation1', activation()),
                ('conv2', conv_type(in_channels=main_channels, out_channels=out_channels,
                                    kernel_size=3, stride=stride, groups=groups, dilation=dilation, padding=dilation)),
                ('norm1', norm_type(out_channels)),
            ])),
            skip=downsample,
        )


class ValidPadResBlock(ResBlock):
    """
    Residual block with valid padding.
    """
    def __init__(self, in_channels, out_channels=None, main_channels=None,
                 kernel_size=1, conv_type=nn.Conv3d, activation='ELU'):
        if out_channels is not None:
            assert out_channels == in_channels, f'{in_channels}, {out_channels}'
        main_channels = in_channels if main_channels is None else main_channels
        if isinstance(activation, str):
            activation = getattr(nn, activation)
        main = skip_none_sequential(
            conv_type(in_channels, main_channels, kernel_size=1),
            activation(),
            conv_type(main_channels, main_channels, kernel_size=kernel_size, padding=0),
            activation(),
            conv_type(main_channels, in_channels, kernel_size=1),
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
    def __init__(self, in_channels, out_channels=None, main_channels=None,
                 conv_type=None, norm_type=None, activation=None,
                 pre_kernel_size=(1, 3, 3), inner_kernel_size=(3, 3, 3)):
        if main_channels is None:
            main_channels = out_channels
        if out_channels is None:
            out_channels = main_channels
        if norm_type is None:
            norm_type = None
        if conv_type is None:
            conv_type = nn.Conv3d
        if activation is None:
            activation = nn.ELU

        super(SuperhumanSNEMIBlock, self).__init__(
            pre=skip_none_sequential(OrderedDict([
                ('conv1', conv_type(in_channels, out_channels, kernel_size=pre_kernel_size,
                                    padding=get_padding(pre_kernel_size))),
                ('norm1', norm_type(out_channels) if norm_type is not None else None),
                ('activation1', activation()),
            ])),
            main=skip_none_sequential(OrderedDict([
                ('conv1', conv_type(out_channels, main_channels, kernel_size=inner_kernel_size,
                                    padding=get_padding(inner_kernel_size))),
                ('norm1', norm_type(main_channels) if norm_type is not None else None),
                ('activation1', activation()),
                ('conv2', conv_type(main_channels, out_channels, kernel_size=inner_kernel_size,
                                    padding=get_padding(inner_kernel_size))),
                ('norm2', norm_type(out_channels) if norm_type is not None else None),
                ('activation2', activation())
            ]))
        )


class SamePadResBlock(ResBlock):
    """
    ResBlock with 'SAME' padding used for Single-Instance Mask prediction
    and inspired by http://arxiv.org/abs/1909.09872
    """

    def __init__(self,
                 f_in,
                 f_inner=None,
                 f_out=None,
                 pre_kernel_size=(1, 3, 3),
                 kernel_size=(3, 3, 3),
                 apply_final_activation=True,
                 apply_final_normalization=True,
                 dim=3,
                 activation="ReLU",
                 stride=1,
                 normalization="GroupNorm",
                 nb_norm_groups=None,
                 dilation=1):

        f_inner = f_in if f_inner is None else f_inner
        f_out = f_inner if f_out is None else f_out

        skip_con = None
        if stride != 1 or f_in != f_out:
            skip_con = ConvNormActivation(f_in, f_out, kernel_size=1, dim=dim,
                                          activation=activation,
                                          stride=stride,
                                          nb_norm_groups=nb_norm_groups,
                                          normalization=normalization)

        # Construct main layers:
        conv1 = ConvNormActivation(f_in, f_inner, kernel_size=pre_kernel_size,
                                   dim=dim,
                                   activation=activation,
                                   nb_norm_groups=nb_norm_groups,
                                   dilation=dilation,
                                   normalization=normalization)
        conv2 = ConvNormActivation(f_inner, f_inner,
                                   kernel_size=kernel_size, dim=dim,
                                   activation=activation,
                                   stride=stride,
                                   dilation=dilation,
                                   nb_norm_groups=nb_norm_groups,
                                   normalization=normalization)
        conv3 = ConvNormActivation(f_inner, f_out,
                                   kernel_size=kernel_size, dim=dim,
                                   activation=activation,
                                   dilation=dilation,
                                   nb_norm_groups=nb_norm_groups,
                                   normalization=normalization)

        main = nn.Sequential(
            conv1, conv2, conv3.conv
        )

        # Construct post-layers:
        post = []
        if apply_final_normalization and conv3.normalization is not None:
            post.append(conv3.normalization)
        if apply_final_activation and conv3.activation is not None:
            post.append(conv3.activation)
        post = None if len(post) == 0 else nn.Sequential(*post)

        super(SamePadResBlock, self).__init__(
            main,
            skip=skip_con,
            post=post
        )
