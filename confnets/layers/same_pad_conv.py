import torch.nn as nn


class ConvActivation(nn.Module):
    """
    Convolutional layer with 'SAME' padding by default followed by an activation.
    (also available in inferno)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dim,
        activation,
        stride=1,
        dilation=1,
        groups=None,
        depthwise=False,
        bias=True,
        deconv=False,
        valid_conv=False,
    ):
        super(ConvActivation, self).__init__()
        # Validate dim
        assert dim in [1, 2, 3], "`dim` must be one of [1, 2, 3], got {}.".format(dim)

        self.dim = dim
        # Check if depthwise
        if depthwise:

            # We know that in_channels == out_channels, but we also want a consistent API.
            # As a compromise, we allow that out_channels be None or 'auto'.
            out_channels = in_channels if out_channels in [None, "auto"] else out_channels
            assert in_channels == out_channels, "For depthwise convolutions, number of input channels (given: {}) " \
                                                "must equal the number of output channels (given {}).".format(
                    in_channels, out_channels
                )

            assert groups is None or groups == in_channels, "For depthwise convolutions, groups (given: {}) must equal the number of channels (given: {}).".format(groups, in_channels)
            groups = in_channels
        else:
            groups = 1 if groups is None else groups
        self.depthwise = depthwise
        if valid_conv:
            self.conv = getattr(nn, "Conv{}d".format(self.dim))(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        elif not deconv:
            # Get padding
            padding = self.get_padding(kernel_size, dilation)
            self.conv = getattr(nn, "Conv{}d".format(self.dim))(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        else:
            self.conv = getattr(nn, "ConvTranspose{}d".format(self.dim))(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )

        if isinstance(activation, str):
            self.activation = getattr(nn, activation)()
        elif isinstance(activation, nn.Module):
            self.activation = activation
        elif activation is None:
            self.activation = None
        else:
            raise NotImplementedError

    def forward(self, input):
        conved = self.conv(input)
        if self.activation is not None:
            activated = self.activation(conved)
        else:
            # No activation
            activated = conved
        return activated

    def _pair_or_triplet(self, object_):
        if isinstance(object_, (list, tuple)):
            assert len(object_) == self.dim
            return object_
        else:
            object_ = [object_] * self.dim
            return object_

    def _get_padding(self, _kernel_size, _dilation):
        assert isinstance(_kernel_size, int)
        assert isinstance(_dilation, int)
        assert _kernel_size % 2 == 1
        return ((_kernel_size - 1) // 2) * _dilation

    def get_padding(self, kernel_size, dilation):
        kernel_size = self._pair_or_triplet(kernel_size)
        dilation = self._pair_or_triplet(dilation)
        padding = [
            self._get_padding(_kernel_size, _dilation)
            for _kernel_size, _dilation in zip(kernel_size, dilation)
        ]
        return tuple(padding)


class ConvNormActivation(ConvActivation):
    """Convolutional layer with 'SAME' padding by default followed by a normalization and activation layer."""
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
