import torch
import torch.nn as nn


class TakeChannels(nn.Module):
    """
    Take the first n channels of the inputs.
    """

    def __init__(self, stop):
        super(TakeChannels, self).__init__()
        self.stop = stop

    def forward(self, input):
        if type(input) is tuple:
            return tuple(x[:, :self.stop] for x in input)
        else:
            return input[:, :self.stop]


class ReduceIntermediateWith1x1(nn.Module):
    """
    Apply 1x1 convolutions to all but the last input, to reduce the number of channels to out_channels.
    """

    def __init__(self, in_channels, out_channels, dim=2):
        super(ReduceIntermediateWith1x1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if dim == 2:
            self.convs = nn.ModuleList([nn.Conv2d(c, out_channels, kernel_size=1)
                                        for c in in_channels])
        else:
            raise NotImplementedError(f'Only dim=2 is implemented. Got dim={dim}')

    def forward(self, input):
        assert type(input) is tuple, f'input must be tuple. Got {type(input)}'
        input = list(input)
        input[:-1] = [conv(t) for t, conv in zip(input[:-1], self.convs)]
        return tuple(input)
