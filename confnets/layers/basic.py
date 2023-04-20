import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from ..utils import parse_data_slice

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
        if self.mode == 'nearest':
            # align corners is not supported for mode 'nearest'
            return nn.functional.interpolate(input, scale_factor=self.scale_factor, mode=self.mode)
        else:
            return nn.functional.interpolate(input, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)


class Crop(nn.Module):
    """
    Crop a tensor according to the given string representing the crop slice
    """
    def __init__(self, crop_slice):
        super(Crop, self).__init__()
        self.crop_slice = crop_slice

        if self.crop_slice is not None:
            assert isinstance(self.crop_slice, str)
            self.crop_slice = (slice(None), slice(None)) + parse_data_slice(self.crop_slice)

    def forward(self, input):
        if isinstance(input, tuple):
            raise NotImplementedError("At the moment only one input is accepted")
        if self.crop_slice is not None:
            return input[self.crop_slice]
        else:
            return input


class UpsampleAndCrop(nn.Module):
    """
    Combination of Upsample and Crop
    """
    def __init__(self, scale_factor, mode,
                                  crop_slice=None):
        super(UpsampleAndCrop, self).__init__()
        self.upsampler = Upsample(scale_factor=scale_factor, mode=mode)
        self.crop_module = Crop(crop_slice=crop_slice)

    def forward(self, input):
        if isinstance(input, tuple):
            input = input[0]
        input = self.crop_module(input)
        output = self.upsampler(input)
        return output

