import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# optional imports

try:
    from speedrun.log_anywhere import log_image
except ImportError:
    def log_image(tag, value):
        assert False, f'Image logging cannot be used without speedrun.'

try:
    from gpushift import MeanShift
except ImportError:
    class MeanShift:
        def __init__(self, *args, **kwargs):
            assert False, f'gpushift not found. please install from https://github.com/imagirom/gpushift.'

try:
    from embeddingutils.affinities import offset_slice, offset_padding, get_offsets
except ImportError:
    def embeddingutils_not_found(*args, **kwargs):
        assert False, f'embeddingutils not found. Please install from https://github.com/Steffen-Wolf/embeddingutils.'
    offset_slice, offset_padding, get_offsets = [embeddingutils_not_found] * 3


class ContiguousBackward(torch.autograd.Function):
    """
    Function to ensure contiguous gradient in backward pass. To be applied after PyKeOps reduction.
    """

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.contiguous()


class MeanShiftLayer(MeanShift):
    """
    Wrapper for MeanShift that handles appropriate reshaping.
    """

    def forward(self, embedding):
        in_shape = embedding.shape  # B E (D) H W
        embedding = embedding.view(in_shape[:2] + (-1,))  # B E N
        embedding = embedding.transpose(1, 2)  # B N E
        embedding = super(MeanShiftLayer, self).forward(embedding)
        embedding = ContiguousBackward().apply(embedding)
        embedding = embedding.transpose(1, 2)  # B E N
        embedding = embedding.view(in_shape)  # B E (D) H W
        return embedding


class ShakeShakeFn(torch.autograd.Function):
    """ modified from https://github.com/owruby/shake-shake_pytorch/blob/master/models/shakeshake.py """
    @staticmethod
    def forward(ctx, x1, x2, training=True):
        # first dim is assumed to be batch
        if training:
            alpha = torch.rand(x1.size(0), *((1,)*(len(x1.shape)-1)), dtype=x1.dtype, device=x1.device)
        else:
            alpha = 0.5
        return alpha * x1 + (1 - alpha) * x2

    @staticmethod
    def backward(ctx, grad_output):
        beta = torch.rand(grad_output.size(0), *((1,) * (len(grad_output.shape) - 1)),
                          dtype=grad_output.dtype, device=grad_output.device)

        return beta * grad_output, (1 - beta) * grad_output, None


class ShakeShakeMerge(nn.Module):
    def forward(self, x1, x2):
        return ShakeShakeFn.apply(x1, x2, self.training)


class SampleChannels(nn.Module):
    def __init__(self, n_selected_channels):
        super(SampleChannels, self).__init__()
        self.n_selected_channels = n_selected_channels

    def sample_ind(self, n_channels):
        assert self.n_selected_channels <= n_channels
        result = np.zeros(n_channels)
        result[np.random.choice(np.arange(n_channels), self.n_selected_channels, replace=False)] = 1
        return result

    def forward(self, input):
        n_channels = input.size(1)
        ind = np.stack([self.sample_ind(n_channels) for _ in range(input.size(0))])
        ind = torch.ByteTensor(ind).to(input.device)
        return input[ind].view(input.size(0), self.n_selected_channels, *input.shape[2:])


class AffinityBasedAveraging(torch.nn.Module):
    def __init__(self, offsets, extra_dims=2, softmax=True, activation=None, normalize=True, **pad_kwargs):
        super(AffinityBasedAveraging, self).__init__()
        self.pad_kwargs = dict(mode='replicate', **pad_kwargs)
        self.offsets = get_offsets(offsets)
        self.offset_slices = [offset_slice(off, extra_dims=extra_dims) for off in self.offsets]
        self.reverse_offset_slices = [offset_slice(-off, extra_dims=extra_dims) for off in self.offsets]
        self.offset_padding = [offset_padding(off) for off in self.offsets]
        self.use_softmax = softmax
        if self.use_softmax:
            assert activation is None, f'activation function is overriden by using softmax!'
        if isinstance(activation, str):
            activation = getattr(torch.nn, activation)()
        self.activation = activation
        self.normalize = normalize

    def forward(self, affinities, embedding):
        padded_embeddings = []
        for sl, pad in zip(self.offset_slices, self.offset_padding):
            padded_embeddings.append(F.pad(embedding[sl], pad, **self.pad_kwargs))
        padded_embeddings = torch.stack(padded_embeddings, dim=1)
        if self.use_softmax:
            affinities = F.softmax(affinities, dim=1)
        elif hasattr(self, 'activation') and self.activation:
            affinities = self.activation(affinities)
        if hasattr(self, 'normalize') and self.normalize:
            affinities = F.normalize(affinities, dim=1, p=1)
        counts = affinities.new_zeros((affinities.shape[0], 1) + affinities.shape[2:])
        return (padded_embeddings * affinities[:, :, None]).sum(1)


class HierarchicalAffinityAveraging(torch.nn.Module):
    def __init__(self, levels=2, dim=2, stride=1, append_affinities=False, ignore_n_first_channels=0, log_images=False,
                 **kwargs):
        """ averages iteratively with thrice as long offsets in every level """
        super(HierarchicalAffinityAveraging, self).__init__()

        self.base_neighborhood = stride * np.mgrid[dim*(slice(-1, 2),)].reshape(dim, -1).transpose()
        self.stages = nn.ModuleList([AffinityBasedAveraging(3**i * self.base_neighborhood, **kwargs)
                                     for i in range(levels)])
        self.levels = levels
        self.dim = dim
        self.append_affinities = append_affinities
        self.ignore_n_first_channels = ignore_n_first_channels
        self.log_images = log_images

    def forward(self, input):
        ignored = input[:, :self.ignore_n_first_channels]
        input = input[:, self.ignore_n_first_channels:]

        affinity_groups = input[:, :len(self.base_neighborhood) * self.levels]
        affinity_groups = affinity_groups.reshape(
            input.size(0), self.levels, len(self.base_neighborhood), *input.shape[2:])\
            .permute(1, 0, *range(2, 3 + self.dim))
        embedding = input[:, len(self.base_neighborhood) * self.levels:]
        for i, (affinities, stage) in enumerate(zip(affinity_groups, self.stages)):
            if self.log_images:
                log_image(f'embedding_stage_{i}', embedding)
                log_image(f'affinities_stage_{i}', affinities)

            embedding = stage(affinities, embedding)
        return torch.cat([ignored, embedding], 1)
