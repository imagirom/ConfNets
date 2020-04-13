import torch
import numpy as np
from collections import OrderedDict
from functools import partial

from ..nn import delayed_nn as nn
from ..layers import Identity, Concatenate, Sum, ConvGRU, ShakeShakeMerge, Upsample, MultiplyByScalar
from ..blocks import SuperhumanSNEMIBlock
from .. import blocks
from ..utils import skip_none_sequential, get_padding


class EncoderDecoderSkeleton(nn.Module):
    """
    Base class for Networks with Encoder Decoder Structure, such as UNet.
    To use, inherit from this and implement a selection of the construct_* methods.
    To add side-outputs, use a wrapper
    """
    # TODO: add input_module to draw.io
    def __init__(self, depth):
        super(EncoderDecoderSkeleton, self).__init__()
        self.depth = depth
        # construct all the layers
        self.initial_module = self.construct_input_module()
        self.encoder_modules = nn.ModuleList(
            [self.construct_encoder_module(i) for i in range(depth)])
        self.skip_modules = nn.ModuleList(
            [self.construct_skip_module(i) for i in range(depth)])
        self.downsampling_modules = nn.ModuleList(
            [self.construct_downsampling_module(i) for i in range(depth)])
        self.upsampling_modules = nn.ModuleList(
            [self.construct_upsampling_module(i) for i in range(depth)])
        self.decoder_modules = nn.ModuleList(
            [self.construct_decoder_module(i) for i in range(depth)])
        self.merge_modules = nn.ModuleList(
            [self.construct_merge_module(i) for i in range(depth)])
        self.base_module = self.construct_base_module()
        self.final_module = self.construct_output_module()

    def forward(self, input):
        encoded_states = []
        current = self.initial_module(input)
        for encode, downsample in zip(self.encoder_modules, self.downsampling_modules):
            current = encode(current)
            encoded_states.append(current)
            current = downsample(current)
        current = self.base_module(current)
        for encoded_state, upsample, skip, merge, decode in reversed(list(zip(
                encoded_states, self.upsampling_modules, self.skip_modules, self.merge_modules, self.decoder_modules))):
            current = upsample(current)
            encoded_state = skip(encoded_state)
            current = merge(current, encoded_state)
            current = decode(current)
        current = self.final_module(current)
        return current

    def construct_input_module(self):
        return Identity()

    def construct_encoder_module(self, depth):
        return Identity()

    def construct_decoder_module(self, depth):
        return self.construct_encoder_module(depth)

    def construct_downsampling_module(self, depth):
        return Identity()

    def construct_upsampling_module(self, depth):
        return Identity()

    def construct_skip_module(self, depth):
        return Identity()

    def construct_merge_module(self, depth):
        return Concatenate()

    def construct_base_module(self):
        return Identity()

    def construct_output_module(self):
        return Identity()


class UNetSkeleton(EncoderDecoderSkeleton):

    def __init__(self, depth, in_channels, out_channels, fmaps, **kwargs):
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(fmaps, (list, tuple)):
            self.fmaps = fmaps
        else:
            assert isinstance(fmaps, int)
            if 'fmap_increase' in kwargs:
                self.fmap_increase = kwargs['fmap_increase']
                self.fmaps = [fmaps + i * self.fmap_increase for i in range(self.depth + 1)]
            elif 'fmap_factor' in kwargs:
                self.fmap_factor = kwargs['fmap_factor']
                self.fmaps = [fmaps * self.fmap_factor**i for i in range(self.depth + 1)]
            else:
                self.fmaps = [fmaps, ] * (self.depth + 1)
        assert len(self.fmaps) == self.depth + 1

        self.merged_fmaps = [2 * n for n in self.fmaps]

        super(UNetSkeleton, self).__init__(depth)

    def construct_layer(self, f_in, f_out):
        pass

    def construct_merge_module(self, depth):
        return Concatenate()

    def construct_encoder_module(self, depth):
        f_in = self.in_channels if depth == 0 else self.fmaps[depth - 1]
        f_out = self.fmaps[depth]
        return nn.Sequential(
            self.construct_layer(f_in, f_out),
            self.construct_layer(f_out, f_out)
        )

    def construct_decoder_module(self, depth):
        f_in = self.merged_fmaps[depth]
        f_intermediate = self.fmaps[depth]
        # do not reduce to f_out yet - this is done in the output module
        f_out = f_intermediate if depth == 0 else self.fmaps[depth - 1]
        return nn.Sequential(
            self.construct_layer(f_in, f_intermediate),
            self.construct_layer(f_intermediate, f_out)
        )

    def construct_base_module(self):
        f_in = self.fmaps[self.depth - 1]
        f_intermediate = self.fmaps[self.depth]
        f_out = self.fmaps[self.depth - 1]
        return nn.Sequential(
            self.construct_layer(f_in, f_intermediate),
            self.construct_layer(f_intermediate, f_out)
        )


class UNet(UNetSkeleton):

    def __init__(self,
                 dim=2,
                 scale_factor=2,
                 conv_type=None,
                 norm_type=None,
                 activation='ReLU',
                 final_activation=None,
                 upsampling_mode='nearest',
                 skip_factor=1,
                 *super_args, **super_kwargs):

        self.dim = dim
        self.final_activation = [final_activation] if final_activation is not None else None

        # parse conv_type
        if conv_type is None:
            conv_type = f'Conv{self.dim}d'
        if isinstance(conv_type, str):
            assert hasattr(nn, conv_type), f'{conv_type} not found in nn'
            conv_type = getattr(nn, conv_type)
        assert callable(conv_type), f'conv_type has to be string or callable.'
        self.conv_type = conv_type

        # parse norm_type
        if isinstance(norm_type, str):
            if norm_type.startswith('GroupNorm'):
                # Set requested number of groups
                norm_type = partial(nn.GroupNorm, int(norm_type[9:]))
            else:
                # Get normalization from nn
                assert hasattr(nn, norm_type), f'{norm_type} not found in nn'
                norm_type = getattr(nn, norm_type)
        if norm_type is None:
            norm_type = lambda in_channels: None
        assert callable(norm_type), \
            f'norm_type has to be string, callable or None'
        self.norm_type = norm_type

        # parse activation
        if isinstance(activation, str):
            assert hasattr(nn, activation), f'{activation} not found in nn'
            activation = getattr(nn, activation)
            assert isinstance(activation, type), f'{activation}, {type(activation)}'
        if isinstance(activation, nn.Module):
            activation_module = activation
            activation = lambda: activation_module
        assert callable(activation), \
            f'activation has to be string, nn.Module or callable.'
        self.activation = activation
        
        # shorthand dictionary for conv_type, norm_type and activation, e.g. for the initialization of blocks
        self.conv_norm_act_dict = dict(conv_type=self.conv_type, norm_type=self.norm_type, activation=self.activation)

        # parse scale factor
        if isinstance(scale_factor, int):
            scale_factor = [scale_factor, ] * super_kwargs['depth']
        scale_factors = scale_factor
        normalized_factors = []
        for scale_factor in scale_factors:
            assert isinstance(scale_factor, (int, list, tuple))
            if isinstance(scale_factor, int):
                scale_factor = self.dim * [scale_factor, ]
            assert len(scale_factor) == self.dim
            normalized_factors.append(scale_factor)
        self.scale_factors = normalized_factors
        self.upsampling_mode = upsampling_mode

        self.skip_factor = skip_factor

        # compute input size divisibility constraints
        divisibility_constraint = np.ones(len(self.scale_factors[0]))
        for scale_factor in self.scale_factors:
            divisibility_constraint *= np.array(scale_factor)
        self.divisibility_constraint = list(divisibility_constraint.astype(int))

        super(UNet, self).__init__(*super_args, **super_kwargs)

        # run a forward pass to initialize all submodules (with in_channels=nn.INIT_DELAYED)
        with torch.no_grad():
            inp = torch.zeros((2, self.in_channels, *self.divisibility_constraint), dtype=torch.float32)
            self(inp)

        # delete attributes that are only relevant for construction and might lead to errors when model is saved
        del self.conv_type
        del self.norm_type
        del self.activation
        del self.conv_norm_act_dict

    def construct_layer(self, f_in, f_out, kernel_size=3):
        return skip_none_sequential(OrderedDict([
            ('conv', self.conv_type(f_in, f_out, kernel_size=kernel_size, padding=get_padding(kernel_size))),
            ('norm', self.norm_type(f_out)),
            ('activation', self.activation())
        ]))

    def construct_output_module(self):
        if self.final_activation is not None:
            return skip_none_sequential(OrderedDict([
                ('final_conv', self.conv_type(self.fmaps[0], self.out_channels, kernel_size=1)),
                ('final_activation', self.final_activation[0])
            ]))
        else:
            return self.conv_type(self.fmaps[0], self.out_channels, kernel_size=1)

    def construct_skip_module(self, depth):
        if self.skip_factor == 1:
            return Identity()
        else:
            return MultiplyByScalar(self.skip_factor)

    def construct_downsampling_module(self, depth):
        scale_factor = self.scale_factors[depth]
        maxpool = getattr(nn, f'MaxPool{self.dim}d')
        return maxpool(kernel_size=scale_factor,
                          stride=scale_factor,
                          padding=0)

    def construct_upsampling_module(self, depth):
        scale_factor = self.scale_factors[depth]
        if scale_factor[0] == 1:
            assert scale_factor[1] == scale_factor[2]
        if self.upsampling_mode != 'transpose_convolution':
            sampler = Upsample(scale_factor=scale_factor, mode=self.upsampling_mode)
        else:
            sampler = nn.ConvTranspose2d(kernel_size=scale_factor, stride=scale_factor)
        return sampler

    def forward(self, input_):
        input_dim = len(input_.shape)
        assert all(input_.shape[-i] % self.divisibility_constraint[-i] == 0 for i in range(1, input_dim-1)), \
            f'Input shape {input_.shape[2:]} not suited for downsampling with factors {self.scale_factors}.' \
            f'Lengths of spatial axes must be multiples of {self.divisibility_constraint}.'
        return super(UNet, self).forward(input_)


class UNet2d(UNet):
    def __init__(self, conv_type='Conv2d', **super_kwargs):
        super_kwargs['conv_type'] = conv_type
        super(UNet2d, self).__init__(dim=2, **super_kwargs)


class UNet3d(UNet):
    def __init__(self, conv_type='Conv3d', **super_kwargs):
        super_kwargs['conv_type'] = conv_type
        super(UNet3d, self).__init__(dim=3, **super_kwargs)


class BlockyUNet(UNet):
    # Note: The numbers of channels per depth works differently compared to default UNet..
    # TODO: draw in draw.io
    # TODO: add option to have true ResBlocks, with 1x1 between to change number of channels
    def __init__(self, block_type='BasicResBlock', **super_kwargs):
        # parse block_type
        if isinstance(block_type, str):
            assert hasattr(blocks, block_type), f'{block_type} not found in confnets.blocks'
            block_type = getattr(blocks, block_type)
        assert callable(block_type), f'block_type has to be string or callable.'
        self.block_type = block_type
        super(BlockyUNet, self).__init__(**super_kwargs)
        # delete attributes that are only relevant for construction and might lead to errors when model is saved
        del self.block_type

    def construct_input_module(self):
        return self.conv_type(self.in_channels, self.fmaps[0], kernel_size=3, padding=1)

    def construct_encoder_module(self, depth):
        return self.block_type(
            in_channels=self.fmaps[depth],
            out_channels=self.fmaps[depth+1],
            **self.conv_norm_act_dict)

    def construct_decoder_module(self, depth):
        return self.block_type(
            in_channels=nn.INIT_DELAYED,  # use delayed init
            out_channels=self.fmaps[depth],
            **self.conv_norm_act_dict)

    def construct_base_module(self):
        return self.block_type(
            in_channels=self.fmaps[self.depth],
            out_channels=self.fmaps[self.depth],
            **self.conv_norm_act_dict)


class SuperhumanSNEMINet(UNet3d):
    # see https://arxiv.org/pdf/1706.00120.pdf
    def __init__(self,
                 in_channels=1, out_channels=1,
                 fmaps=(28, 36, 48, 64, 80),
                 conv_type='Conv3d',
                 scale_factor=(
                     (1, 2, 2),
                     (1, 2, 2),
                     (1, 2, 2),
                     (1, 2, 2)
                 ),
                 depth=None,
                 **kwargs):
        if depth is None:
            depth = len(fmaps) - 1
        super(SuperhumanSNEMINet, self).__init__(
            conv_type=conv_type,
            depth=depth,
            fmaps=fmaps,
            in_channels=in_channels,
            out_channels=out_channels,
            scale_factor=scale_factor,
            **kwargs
        )

    def construct_merge_module(self, depth):
        return Sum()

    def construct_encoder_module(self, depth):
        f_in = self.fmaps[depth - 1] if depth != 0 else self.in_channels
        f_out = self.fmaps[depth]
        if depth != 0:
            return SuperhumanSNEMIBlock(in_channels=f_in, out_channels=f_out, **self.conv_norm_act_dict)
        if depth == 0:
            return SuperhumanSNEMIBlock(in_channels=f_in, out_channels=f_out, **self.conv_norm_act_dict,
                                        pre_kernel_size=(1, 5, 5), inner_kernel_size=(1, 3, 3))

    def construct_decoder_module(self, depth):
        f_in = self.fmaps[depth]
        f_out = self.fmaps[0] if depth == 0 else self.fmaps[depth - 1]
        if depth != 0:
            return SuperhumanSNEMIBlock(in_channels=f_in, out_channels=f_out, **self.conv_norm_act_dict)
        if depth == 0:
            return nn.Sequential(
                SuperhumanSNEMIBlock(in_channels=f_in, out_channels=f_out, **self.conv_norm_act_dict,
                                     pre_kernel_size=(3, 3, 3), inner_kernel_size=(1, 3, 3)),
                self.conv_type(f_out, self.out_channels, kernel_size=(1, 5, 5), padding=(0, 2, 2))
            )

    def construct_base_module(self):
        f_in = self.fmaps[self.depth - 1]
        f_intermediate = self.fmaps[self.depth]
        f_out = self.fmaps[self.depth - 1]
        return SuperhumanSNEMIBlock(in_channels=f_in, main_channels=f_intermediate, out_channels=f_out, 
                                    **self.conv_norm_act_dict)


class ShakeShakeSNEMINet(SuperhumanSNEMINet):

    def construct_merge_module(self, depth):
        return ShakeShakeMerge()


class IsotropicSuperhumanSNEMINet(SuperhumanSNEMINet):

    def construct_encoder_module(self, depth):
        f_in = self.fmaps[depth - 1] if depth != 0 else self.in_channels
        f_out = self.fmaps[depth]
        if depth != 0:
            return SuperhumanSNEMIBlock(in_channels=f_in, out_channels=f_out, **self.conv_norm_act_dict,
                                        pre_kernel_size=(3, 3, 3), inner_kernel_size=(3, 3, 3))
        if depth == 0:
            return SuperhumanSNEMIBlock(in_channels=f_in, out_channels=f_out, **self.conv_norm_act_dict,
                                        pre_kernel_size=(5, 5, 5), inner_kernel_size=(3, 3, 3))

    def construct_decoder_module(self, depth):
        f_in = self.fmaps[depth]
        f_out = self.fmaps[0] if depth == 0 else self.fmaps[depth - 1]
        if depth != 0:
            return SuperhumanSNEMIBlock(in_channels=f_in, out_channels=f_out, **self.conv_norm_act_dict,
                                        pre_kernel_size=(3, 3, 3), inner_kernel_size=(3, 3, 3))
        if depth == 0:
            return nn.Sequential(
                SuperhumanSNEMIBlock(in_channels=f_in, out_channels=f_out, **self.conv_norm_act_dict,
                                     pre_kernel_size=(3, 3, 3), inner_kernel_size=(3, 3, 3)),
                self.conv_type(f_out, self.out_channels, kernel_size=(5, 5, 5), padding=(2, 2, 2))
            )

    def construct_base_module(self):
        f_in = self.fmaps[self.depth - 1]
        f_intermediate = self.fmaps[self.depth]
        f_out = self.fmaps[self.depth - 1]
        return SuperhumanSNEMIBlock(in_channels=f_in, main_channels=f_intermediate, out_channels=f_out,
                                    **self.conv_norm_act_dict,
                                    pre_kernel_size=(3, 3, 3), inner_kernel_size=(3, 3, 3))


# The RecurrentUNet2d is currently not supported. Ask Roman if you are interested in using it.

# class RecurrentUNet2d(UNet2d):
#
#     def __init__(self,
#                  scale_factor=2,
#                  conv_type='Conv2d',
#                  final_activation=None,
#                  rec_n_layers=3,
#                  rec_kernel_sizes=(3, 5, 3),
#                  rec_hidden_size=(16, 32, 8),
#                  *super_args, **super_kwargs):
#         self.rec_n_layers = rec_n_layers
#         self.rec_kernel_sizes = rec_kernel_sizes
#         self.rec_hidden_size = rec_hidden_size
#         self.rec_layers = []
#
#         super(UNet2d, self).__init__(scale_factor=scale_factor,
#                                      conv_type=conv_type,
#                                      final_activation=final_activation,
#                                      *super_args, **super_kwargs)
#
#     def construct_skip_module(self, depth):
#         self.rec_layers.append(ConvGRU(input_size=self.fmaps[depth],
#                                        hidden_size=self.fmaps[depth],
#                                        kernel_sizes=self.rec_kernel_sizes,
#                                        n_layers=self.rec_n_layers,
#                                        conv_type=self.conv_type))
#
#         return self.rec_layers[-1]
#
#     def set_sequence_length(self, sequence_length):
#         for r in self.rec_layers:
#             r.sequence_length = sequence_length
#
#     def forward(self, input_):
#         sequence_length = input_.shape[2]
#         batch_size = input_.shape[0]
#         flat_shape = [batch_size * sequence_length] + [input_.shape[1]] + list(input_.shape[3:])
#         flat_output_shape = [batch_size, sequence_length] + [self.out_channels] + list(input_.shape[3:])
#
#         transpose_input = input_.permute((0, 2, 1, 3, 4))\
#                                 .contiguous()\
#                                 .view(*flat_shape).detach()
#
#         self.set_sequence_length(sequence_length)
#         output = super(UNet2d, self).forward(transpose_input)
#         return output.view(*flat_output_shape).permute((0, 2, 1, 3, 4))

