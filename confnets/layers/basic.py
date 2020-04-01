import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from inferno.extensions.layers.convolutional import ConvActivation


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Concatenate(nn.Module):
    """
    Concatenate input tensors along a specified dimension.
    """
    def __init__(self, dim=1):
        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)


class Sum(nn.Module):
    """Sum all inputs."""
    def forward(self, *inputs):
        return torch.stack(inputs, dim=0).sum(0)


class DepthToChannel(nn.Module):
    def forward(self, input_):
        assert len(input_.shape) == 5, \
            f'input must be 5D tensor of shape (B, C, D, H, W), but got shape {input_.shape}.'
        input_ = input_.permute((0, 2, 1, 3, 4))
        return input_.contiguous().view((-1, ) + input_.shape[-3:])


class Normalize(nn.Module):
    def __init__(self, dim=1):
        super(Normalize, self).__init__()
        self.dim=dim

    def forward(self, input_):
        return F.normalize(input_, dim=self.dim)


class MultiplyByScalar(nn.Module):
    """
    Multiplies the whole input tensor by a single scalar.
    """

    def __init__(self, factor):
        super(MultiplyByScalar, self).__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor


class Upsample(nn.Module):
    """
    Wrapper of nn.functional.interpolate as a nn.Module.
    """
    def __init__(self, scale_factor=2, mode='nearest'):
        super(Upsample, self).__init__()
        self.mode = mode
        self.scale_factor = scale_factor

    def forward(self, input):
        return nn.functional.interpolate(input, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)


class ConvNormActivation(ConvActivation):
    """
    Convolutional layer with 'SAME' padding by default followed by a normalization and activation layer.
    (generalization of ConvActivation in inferno)
    """
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 dim,
                 activation,
                 normalization=None,
                 nb_norm_groups=None,
                 **super_kwargs):
        super(ConvNormActivation, self).__init__(in_channels, out_channels, kernel_size,
                                                 dim, activation, **super_kwargs)

        if isinstance(normalization, str):
            if normalization == "GroupNorm":
                assert nb_norm_groups is not None
                self.normalization = getattr(nn, normalization)(num_groups=nb_norm_groups,
                                                            num_channels=out_channels)
            else:
                self.normalization = getattr(nn, normalization)(out_channels)
        elif isinstance(normalization, nn.Module):
            if isinstance(normalization, nn.GroupNorm):
                assert nb_norm_groups is not None
                self.normalization = normalization(num_groups=nb_norm_groups,
                                                   num_channels=out_channels)
            else:
                self.normalization = normalization(out_channels)
        elif normalization is None:
            self.normalization = None
        else:
            raise NotImplementedError

    def forward(self, input):
        conved = self.conv(input)
        if self.normalization is not None:
            normalized = self.normalization(conved)
        else:
            normalized = conved
        if self.activation is not None:
            activated = self.activation(normalized)
        else:
            # No activation
            activated = normalized
        return activated
