import torch
import torch.nn as nn


def _get_submodule(module, path):
    result = module
    for sub in path.split('/'):
        assert hasattr(result, sub), f'Submodule not found: {sub} (in path{path}). ' \
            f'Available submodules: {[name for name, _ in result.named_submodules()]}'
        result = getattr(result, sub)
    return result


class IntermediateOutputWrapper(nn.Module):
    def __init__(self, module, output_paths):
        """
        Wrap a nn.Module to return a list of intermediate activations instead of the final output.
        :param module: nn.Module
        The module to wrap
        :param output_paths: list of str
        Paths to the modules whose outputs are the intermediate activations.
        Specify 'module.blocks.1' as 'module/blocks/1'.
        """
        super(IntermediateOutputWrapper, self).__init__()
        self.module = module
        assert isinstance(output_paths, (tuple, list)) and all([isinstance(path, str) for path in output_paths]), \
            f'output_paths must be a list of strings. Got {output_paths}'
        self.saved_outputs = None
        self.output_modules = [_get_submodule(self.module, path) for path in output_paths]
        self.output_paths = output_paths
        for module in self.output_modules:
            print(module)
            module.register_forward_hook(self.save_output)

    def save_output(self, module, input, output):
        self.saved_outputs[module] = output

    def forward(self, *input):
        self.saved_outputs = dict()
        self.module.forward(*input)
        return tuple((self.saved_outputs[module] for module in self.output_modules))


class ChannelSliceWrapper(torch.nn.Module):
    """
    Wrapper to apply a module only to some channels. The module is assumed to not change the input tensors shape.
    """
    def __init__(self, module, start=0, stop=None):
        super(ChannelSliceWrapper, self).__init__()
        self.slice = slice(start, stop)
        self.module = module

    def forward(self, input):
        input = input.clone()  # to make sure we stay clear of unwanted side effects
        input[:, self.slice] = self.module(input[:, self.slice])
        return input
