import torch
import torch.nn as nn
from copy import deepcopy

from ..blocks import SamePadResBlock
from ..layers import Identity, ConvNormActivation, UpsampleAndCrop, Crop
from .unet import EncoderDecoderSkeleton


class MultiScaleInputMultiOutputUNet(EncoderDecoderSkeleton):
    def __init__(self,
                 depth,
                 in_channels,
                 encoder_fmaps,
                 output_branches_specs,
                 ndim=3,
                 decoder_fmaps=None,
                 number_multiscale_inputs=1,
                 scale_factor=(1,2,2),
                 res_blocks_specs=None,
                 res_blocks_specs_decoder=None,
                 upsampling_mode='nearest',
                 decoder_crops=None,
                 return_input=False,
                 nb_norm_groups=16
                 ):
        """
        TODO: use delayed_init trick and allow inputs with different numbers of channels

        Generalized UNet model with the following features:

         - Possibility to build one or more output-branches at any level of the UNet decoder for deep supervision
         - Optionally, pass multiple inputs at different scales. Features at different levels of
                the UNet encoder are auto-padded and concatenated to the given inputs.
         - Sum of skip-connections (no concatenation)
         - Downscale with strided conv
         - ResBlocks implemented with GroupNorm+ReLU
         - Custom number of 2D or 3D (same-pad-)ResBlocks at different levels of the UNet hierarchy
         - Optionally, perform spatial-crops in the UNet decoder to save memory and crop boundary artifacts
                (skip-connections are automatically cropped to match)



        :param encoder_fmaps: (list, tuple) of length depth+1. Make sure to pass values that are divisible by
                `nb_norm_groups` (ResBlocks are currently implemented with GroupNorm)

        :param decoder_fmaps: (list, tuple) of length depth+1. By default equal to `encoder_fmaps`

        :param output_branches_specs: either list or dictionary (easily configurable from config file) specifying how
                to construct output branches. For each branch, `out_channels` and `depth` should be specified.
                Multiple branches can be associated to the same depth.

                Example 1 (list of dictionaries):
                [
                    # First branch: foreground-background prediction
                    {depth: 0, out_channels: 1, activation: "Sigmoid", normalization: None},
                    # Second branch: embedding prediction
                    {depth: 0, out_channels: 32, activation: None, normalization: None}
                ]

                Example 2 (dictionary shown in .yml style):
                {
                    # These specs are applied to all branches:
                    global:
                        out_channels: 32
                        normalization: GroupNorm
                        activation: ReLU
                        nb_norm_groups: 16
                    # Foreground-background prediction branch rewrites global specs:
                    0:
                        depth: 0
                        out_channels: 1
                        activation: "Sigmoid"
                        normalization: None
                    # Other emb. branches at different levels in the UNet decoder:
                    1: {depth: 0}
                    2: {depth: 1}
                    3: {depth: 2}
                }


        :param number_multiscale_inputs: At the moment, each input is expected to have the same number
                of channels `in_channels`.
                Multiple inputs are expected to be passed in the correct order to the forward method:
                    self.forward(input_depth_0_full_res, input_depth_1_res_2x, input_depth_2_res_4x, ...)

        :param res_blocks_specs: List of length depth+1 specifying how many resBlocks we should concatenate
                at each level. Example:
                    [
                        [False, False], # Two 2D ResBlocks at the highest level
                        [True],         # One 3D ResBlock at the second level of the UNet hierarchy
                        [True, True],   # Two 3D ResBlock at the third level of the UNet hierarchy
                    ]
                Default value is None and one 3D block is placed at all levels.

        :param res_blocks_specs_decoder: Same for the decoder. By default `res_blocks_specs` is copied and the
                UNet architecture is symmetric.

        :param scale_factor: The following options are allowed:

                - Int, e.g. scale_factor=3 gives scale factors (3,3,3) for all depths
                - List of len==ndim, e.g. [1,3,3] applied to all depths
                - List of lists specifying factors for each depth, e.g. [ [1,2,2], [1,3,3], [1,3,3] ]

        :param decoder_crops: dictionary of strings specifying the 3D crops to be performed at each level of the
                UNet decoder. By default is None

                Example:
                {
                    0: "1:-1, 8:-8, 8:-8",  # Crop applied at the high-res scale before to output
                    2: ":, 2:-2, 2:-2"      # Crop applied at depth 2 in the decoder before to upscale
                }

        :param return_input: If True, the forward method concatenates the input tensors to the output ones.
                By default is False.

        :param nb_norm_groups: ResBlocks are currently implemented with GroupNorm. So all the passed fmaps should
                be divisible by this number

        """
        assert isinstance(ndim, int)
        assert ndim == 2 or ndim == 3
        self._dim = ndim

        assert isinstance(return_input, bool)
        self.return_input = return_input

        assert isinstance(depth, int)
        self.depth = depth

        assert isinstance(in_channels, int)
        self.in_channels = in_channels

        assert isinstance(upsampling_mode, str)
        self.upsampling_mode = upsampling_mode

        assert isinstance(number_multiscale_inputs, int)
        self.number_multiscale_inputs = number_multiscale_inputs

        assert isinstance(nb_norm_groups, int)
        self.nb_norm_groups = nb_norm_groups

        def assert_depth_args(f_maps):
            assert isinstance(f_maps, (list, tuple))
            assert len(f_maps) == depth + 1

        # Parse feature maps:
        assert_depth_args(encoder_fmaps)
        self.encoder_fmaps = encoder_fmaps
        if decoder_fmaps is None:
            # By default use symmetric architecture:
            self.decoder_fmaps = encoder_fmaps
        else:
            assert_depth_args(decoder_fmaps)
            assert decoder_fmaps[-1] == encoder_fmaps[-1], "Number of layers at the base module should be the same"
            self.decoder_fmaps = decoder_fmaps

        # Parse scale factor:
        if isinstance(scale_factor, int):
            # Only one int factor is passed:
            scale_factor = [scale_factor, ] * depth
        elif isinstance(scale_factor, (list, tuple)):
            if isinstance(scale_factor[0], int):
                # Only one global scale_factor is passed for all depths
                assert len(scale_factor) == self.dim
                scale_factor = [scale_factor, ] * depth
            else:
                assert len(scale_factor) == self.depth
                assert all(len(fact) == self.dim for fact in scale_factor)

        self.scale_factors = scale_factor

        # Parse res-block specifications:
        if res_blocks_specs is None:
            # Default: one 3D block per level for 3D UNet, otherwise one 2D block
            default_setup = True if self.dim == 3 else False
            self.res_blocks_specs = [[default_setup] for _ in range(depth+1)]
        else:
            assert_depth_args(res_blocks_specs)
            assert all(isinstance(itm, list) for itm in res_blocks_specs)
            self.res_blocks_specs = res_blocks_specs
        # Same for the decoder:
        if res_blocks_specs_decoder is None:
            # In this case copy setup of the encoder:
            self.res_blocks_specs_decoder = self.res_blocks_specs
        else:
            assert_depth_args(res_blocks_specs_decoder)
            assert all(isinstance(itm, list) for itm in res_blocks_specs_decoder)
            self.res_blocks_specs_decoder = res_blocks_specs_decoder

        # Parse decoder crops:
        self.decoder_crops = decoder_crops if decoder_crops is not None else {}
        assert len(self.decoder_crops) <= depth, "For the moment maximum one crop is supported"

        # Build the skeleton:
        super(MultiScaleInputMultiOutputUNet, self).__init__(depth)

        # Parse output_branches_specs:
        assert isinstance(output_branches_specs, (dict, list))
        nb_branches = len(output_branches_specs)
        assert nb_branches > 0, "At least one output branch should be defined"
        # Create a list from the given dictionary:
        if isinstance(output_branches_specs, dict):
            # Apply global specs to all branches:
            global_specs = output_branches_specs.pop("global")
            nb_branches = len(output_branches_specs)
            collected_specs = [deepcopy(global_specs) for _ in range(nb_branches)]
            for i in range(nb_branches):
                idx = i
                if idx not in output_branches_specs:
                    idx = str(idx)
                    assert idx in output_branches_specs, "Not all the {} specs for the output branches were " \
                                                              "passed".format(nb_branches)
                collected_specs[i].update(output_branches_specs[idx])
            output_branches_specs = collected_specs
        self.output_branches_specs = output_branches_specs

        # Build output branches:
        self.output_branches_indices = branch_idxs = {}
        output_branches_collected = []
        for i, branch_specs in enumerate(output_branches_specs):
            assert "out_channels" in branch_specs, "Number of output channels missing for branch {}".format(i)
            assert "depth" in branch_specs, "Depth missing for branch {}".format(i)
            depth = branch_specs["depth"]
            assert isinstance(depth, int)
            # Keep track of ordering of the branches (multiple branches can be passed at the same depth):
            if depth in branch_idxs:
                branch_idxs[depth].append(i)
            else:
                branch_idxs[depth] = [i]
            output_branches_collected.append(self.construct_output_branch(**branch_specs))
        self.output_branches = nn.ModuleList(output_branches_collected)
        print(self.output_branches_indices)


        self.autopad_feature_maps = AutoPad() if number_multiscale_inputs > 1 else None

        self.properly_init_normalizations()

    def forward(self, *inputs):
        nb_inputs = len(inputs)
        assert nb_inputs == self.number_multiscale_inputs, "The number of inputs does not match the one expected " \
                                                           "by the model"
        assert all(in_tensor.dim() == self.dim + 2 for in_tensor in inputs), "The dimension of the input tensors do" \
                                                                             "not match the dimension of the model"


        encoded_states = []
        current = inputs[0]
        for encode, downsample, depth in zip(self.encoder_modules, self.downsampling_modules,
                                      range(self.depth)):
            if depth > 0 and depth < self.number_multiscale_inputs:
                # Pad the features and concatenate the next input:
                current_lvl_padded = self.autopad_feature_maps(current, inputs[depth].shape)
                current = torch.cat((current_lvl_padded, inputs[depth]), dim=1)
                current = encode(current)
            else:
                current = encode(current)
            encoded_states.append(current)
            current = downsample(current)
        current = self.base_module(current)

        outputs = [None for _ in self.output_branches]
        for skip_connection, upsample, merge, decode, depth in reversed(list(zip(
                encoded_states, self.upsampling_modules, self.merge_modules,
                self.decoder_modules, range(len(self.decoder_modules))))):
            current = upsample(current)
            current = merge(current, skip_connection)
            current = decode(current)

            if depth in self.output_branches_indices:
                for branch_idx in self.output_branches_indices[depth]:
                    outputs[branch_idx] = self.output_branches[branch_idx](current)

        if self.return_input:
            outputs = outputs + list(inputs)

        outputs = outputs[0] if len(outputs) == 1 else outputs

        return outputs

    def construct_output_branch(self,
                                depth,
                                out_channels,
                                activation="Sigmoid",
                                normalization=None,
                                **extra_conv_kwargs):
        out_branch = ConvNormActivation(self.decoder_fmaps[depth],
                                         out_channels=out_channels,
                                         kernel_size=1,
                                         dim=self.dim,
                                         activation=activation,
                                         normalization=normalization,
                                         **extra_conv_kwargs)
        crop = self.decoder_crops.get(depth, None)
        out_branch = nn.Sequential(out_branch, Crop(crop)) if crop is not None else out_branch
        return out_branch

    def construct_encoder_module(self, depth):
        if depth == 0:
            f_in = self.in_channels
        elif depth < self.number_multiscale_inputs:
            f_in = self.encoder_fmaps[depth - 1] + self.in_channels
        else:
            f_in = self.encoder_fmaps[depth - 1]
        f_out = self.encoder_fmaps[depth]

        # Build blocks:
        blocks_spec = deepcopy(self.res_blocks_specs[depth])

        if depth == 0:
            first_conv = ConvNormActivation(f_in, f_out, kernel_size=self.pre_kernel_size,
                                           dim=self.dim,
                                           activation="ReLU",
                                           nb_norm_groups=self.nb_norm_groups,
                                           normalization="GroupNorm")
            # Here the block has a different number of inpiut channels:
            res_block = self.concatenate_res_blocks(f_out, f_out, blocks_spec)
            res_block = nn.Sequential(first_conv, res_block)
        else:
            res_block = self.concatenate_res_blocks(f_in, f_out, blocks_spec)

        return res_block

    def construct_decoder_module(self, depth):
        f_in = self.decoder_fmaps[depth]
        f_out = self.decoder_fmaps[depth]

        # Build blocks:
        blocks_spec = deepcopy(self.res_blocks_specs_decoder[depth])
        res_block = self.concatenate_res_blocks(f_in, f_out, blocks_spec)
        if depth == 0:
            last_conv = ConvNormActivation(f_out, f_out, kernel_size=self.pre_kernel_size,
                       dim=self.dim,
                       activation="ReLU",
                       nb_norm_groups=self.nb_norm_groups,
                       normalization="GroupNorm")
            res_block = nn.Sequential(res_block, last_conv)
        return res_block

    def construct_base_module(self):
        f_in = self.encoder_fmaps[self.depth - 1]
        f_out = self.encoder_fmaps[self.depth]
        blocks_spec = deepcopy(self.res_blocks_specs[self.depth])
        return self.concatenate_res_blocks(f_in, f_out, blocks_spec)


    def construct_upsampling_module(self, depth):
        # First we need to reduce the numer of channels:
        conv = ConvNormActivation(self.decoder_fmaps[depth+1], self.decoder_fmaps[depth], kernel_size=1,
                           dim=self.dim,
                           activation="ReLU",
                           nb_norm_groups=self.nb_norm_groups,
                           normalization="GroupNorm")

        scale_factor = self.scale_factors[depth]

        sampler = UpsampleAndCrop(scale_factor=scale_factor, mode=self.upsampling_mode,
                                  crop_slice=self.decoder_crops.get(depth+1, None))

        return nn.Sequential(conv, sampler)

    def construct_downsampling_module(self, depth):
        scale_factor = self.scale_factors[depth]
        sampler = ConvNormActivation(self.encoder_fmaps[depth], self.encoder_fmaps[depth],
                           kernel_size=scale_factor,
                           dim=self.dim,
                           stride=scale_factor,
                           valid_conv=True,
                           activation="ReLU",
                           nb_norm_groups=self.nb_norm_groups,
                           normalization="GroupNorm")
        return sampler


    def construct_merge_module(self, depth):
        return MergeSkipConnAndAutoCrop(self.decoder_fmaps[depth], self.encoder_fmaps[depth],
                                        dim=self.dim,
                                        nb_norm_groups=self.nb_norm_groups)

    def concatenate_res_blocks(self, f_in, f_out, blocks_spec):
        """
        Concatenate multiple residual blocks according to the config file
        """
        assert f_out % self.nb_norm_groups == 0, "fmaps {} is not divisible by the given nb_norm_groups {}!".format(
            f_out, self.nb_norm_groups
        )

        blocks_list = []
        for is_3D in blocks_spec:
            assert isinstance(is_3D, bool)
            if is_3D:
                assert self.dim == 3, "3D res-blocks are only supported with a 3D model"
                blocks_list.append(SamePadResBlock(f_in,
                                                   dim=self.dim,
                                                   f_inner=f_out,
                                                   pre_kernel_size=self.resblock_kernel_size,
                                                   kernel_size=self.resblock_kernel_size,
                                                   activation="ReLU",
                                                   normalization="GroupNorm",
                                                   nb_norm_groups=self.nb_norm_groups,
                                                   ))
            else:
                blocks_list.append(SamePadResBlock(f_in,
                                                   dim=self.dim,
                                                   f_inner=f_out,
                                                   pre_kernel_size=self.resblock_kernel_size_2D,
                                                   kernel_size=self.resblock_kernel_size_2D,
                                                   activation="ReLU",
                                                   normalization="GroupNorm",
                                                   nb_norm_groups=self.nb_norm_groups,
                                                   ))
            f_in = f_out

        return nn.Sequential(*blocks_list)

    def properly_init_normalizations(self):
        """
        This was sometimes mentioned online as a trick to avoid normalization init problems
        """
        # TODO: check if it is making any difference
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @property
    def dim(self):
        return self._dim

    @property
    def pre_kernel_size(self):
        if self.dim == 3:
            return (1, 5, 5)
        elif self.dim == 2:
            return (5, 5)
        else:
            raise NotImplementedError("Only ndim=2,3 supported atm")

    @property
    def resblock_kernel_size(self):
        if self.dim == 3:
            return (3, 3, 3)
        elif self.dim == 2:
            return (3, 3)
        else:
            raise NotImplementedError("Only ndim=2,3 supported atm")

    @property
    def resblock_kernel_size_2D(self):
        if self.dim == 3:
            return (1, 3, 3)
        elif self.dim == 2:
            return (3, 3)
        else:
            raise NotImplementedError("Only ndim=2,3 supported atm")


class MultiScaleInputUNet(MultiScaleInputMultiOutputUNet):
    def __init__(self,
                 depth,
                 in_channels,
                 encoder_fmaps,
                 out_channels,
                 final_activation=None,
                 final_normalization=None,
                 final_nb_norm_groups=None,
                 **super_kwargs
                 ):
        """
        Subclass of MultiScaleInputMultiOutputUNet with only one output at the highest level of the UNet decoder.
        By default only one input is expected.

        For more details about other args/kwargs, see docstring MultiScaleInputMultiOutputUNet
        """
        # Build specifications for the only output at highest res:
        output_branches_specs = {
            "depth": 0,
             "out_channels": out_channels
        }
        if final_activation is not None:
            output_branches_specs["activation"] = final_activation
        if final_normalization is not None:
            output_branches_specs["normalization"] = final_normalization
        if final_nb_norm_groups is not None:
            output_branches_specs["nb_norm_groups"] = final_nb_norm_groups

        super(MultiScaleInputUNet, self).__init__(
            depth,
            in_channels,
            encoder_fmaps,
            [output_branches_specs,],
            **super_kwargs
        )


class MergeSkipConnAndAutoCrop(nn.Module):
    """
    Used in the UNet decoder to merge skip connections from feature maps at lower scales
    """
    def __init__(self, nb_prev_fmaps, nb_fmaps_skip_conn,
                 dim=3,
                 nb_norm_groups=16):
        super(MergeSkipConnAndAutoCrop, self).__init__()
        if nb_prev_fmaps == nb_fmaps_skip_conn:
            self.conv = Identity()
        else:
            self.conv = ConvNormActivation(nb_fmaps_skip_conn, nb_prev_fmaps, kernel_size=1,
                                           dim=dim,
                                           activation="ReLU",
                                           normalization="GroupNorm",
                                           nb_norm_groups=nb_norm_groups)

    def forward(self, tensor, skip_connection):
        if tensor.shape[2:] != skip_connection.shape[2:]:
            target_shape = tensor.shape[2:]
            orig_shape = skip_connection.shape[2:]
            diff = [orig-trg for orig, trg in zip(orig_shape, target_shape)]
            crop_backbone = True
            if not all([d>=0 for d in diff]):
                crop_backbone = False
                orig_shape, target_shape = target_shape, orig_shape
                diff = [orig - trg for orig, trg in zip(orig_shape, target_shape)]
            left_crops = [int(d/2) for d in diff]
            right_crops = [shp-int(d/2) if d%2==0 else shp-(int(d/2)+1)  for d, shp in zip(diff, orig_shape)]
            crop_slice = (slice(None), slice(None)) + tuple(slice(lft,rgt) for rgt,lft in zip(right_crops, left_crops))
            if crop_backbone:
                skip_connection = skip_connection[crop_slice]
            else:
                tensor = tensor[crop_slice]

        return self.conv(skip_connection) + tensor


class AutoPad(nn.Module):
    """
    Used to auto-pad the multiple UNet inputs passed at different resolutions
    """
    def __init__(self):
        super(AutoPad, self).__init__()

    def forward(self, to_be_padded, out_shape):
        in_shape = to_be_padded.shape[2:]
        out_shape = out_shape[2:]
        if in_shape != out_shape:
            diff = [trg-orig for orig, trg in zip(in_shape, out_shape)]
            assert all([d>=0 for d in diff]), "Output shape should be bigger"
            assert all([d % 2 == 0 for d in diff]), "Odd difference in shape!"
            # F.pad expects the last dim first:
            diff.reverse()
            pad = []
            for d in diff:
                pad += [int(d/2), int(d/2)]
            to_be_padded = torch.nn.functional.pad(to_be_padded, tuple(pad), mode='constant', value=0)
        return to_be_padded
