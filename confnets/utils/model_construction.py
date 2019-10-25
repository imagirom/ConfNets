from collections import OrderedDict
import torch.nn as nn


def skip_none_sequential(*args):
    """
    Construct nn.Sequential, skipping modules that are None
    :param args: tuple
    list of inputs to sequential
    :return: nn.Sequential
    """
    if len(args) == 1 and isinstance(args[0], OrderedDict):
        args = (OrderedDict([(name, value) for name, value in args[0].items() if value is not None]),)
    else:
        args = tuple(arg for arg in args if arg is not None)
    return nn.Sequential(*args)


def get_single_padding(kernel_size, dilation):
    assert isinstance(kernel_size, int)
    assert isinstance(dilation, int)
    assert kernel_size % 2 == 1
    return ((kernel_size - 1) // 2) * dilation


def get_padding(kernel_size, dilation=1):
    """
    Get padding arg for 'same' type padding
    :param kernel_size: int or list/tuple of int
    :param dilation: int or list/tuple of int
    :return: int or tuple of int
    Padding value to pass to init of convolution.
    """
    if isinstance(kernel_size, int) and isinstance(dilation, int):
        return get_single_padding(kernel_size, dilation)
    else:
        if isinstance(kernel_size, int):
            assert isinstance(dilation, (list, tuple))
            kernel_size = (kernel_size,) * len(dilation)
        if isinstance(dilation, int):
            assert isinstance(kernel_size, (list, tuple))
            dilation = (dilation,) * len(kernel_size)
        return tuple(get_single_padding(ker, dil)
                     for ker, dil in zip(kernel_size, dilation))