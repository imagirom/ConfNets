import torch
import torch.nn as nn
import torch.nn.modules.conv as torch_conv
from functools import partial

# TODO: make this a drop-in replacement for torch.nn
# TODO: current disadvantage: positional arguments do not work. could fix using f.__code__.co_varnames .
# Disadvantage of this approach: Docstring of class is not preserved

INIT_DELAYED = None  # Tag for arguments that are left unassigned at first init


def delayed_init_wrapper(cls, infer_delayed_dict):
    """
    :param cls: type
    class to wrap
    :param infer_delayed_dict: dict
    Keys are the arguments that are inferred only at first forward.
    The values are the functions that get the arguments based on the input to forward.
    :return: type
    """
    assert issubclass(cls, nn.Module)

    class DelayedInitClass(cls):
        def __init__(self, in_tensor=None, **kwargs):
            # save kwargs for delayed init
            self.kwargs = kwargs

            # if all arguments are specified, do the full init now.
            if all([kwargs.get(arg, INIT_DELAYED) is not INIT_DELAYED
                    for arg in infer_channels_dict]):
                self.do_init()

            # if an in_tensor is given, call forward now to infer unassigned arguments
            if in_tensor is not None:
                with torch.no_grad():
                    self(in_tensor)

        def __call__(self, input):
            reference = input
            if not isinstance(input, torch.Tensor):
                reference = input[0]
            assert isinstance(reference, torch.Tensor), \
                f'Input to the module should be a torch.Tensor or a list thereof.'

            # infer unassigned arguments from reference tensor
            for arg, infer_arg in infer_delayed_dict.items():
                if self.kwargs.get(arg, INIT_DELAYED) is INIT_DELAYED:
                    self.kwargs[arg] = infer_arg(reference)

            # do the init, using the updated kwargs
            self.do_init()
            return self.__call__(input)

        def do_init(self):
            # call the init of the base class
            super(DelayedInitClass, self).__init__(**self.kwargs)

            # xlean up: selete kwargs object
            del self.kwargs

            # aet the class to the base class. This should remove all traces of this class ever existing.
            assert len(self.__class__.__bases__) == 1, f'{self.__class__.__bases__}'
            self.__class__ = self.__class__.__bases__[0]

        def __repr__(self):
            # repr is similar of that for nn.Module, showing all kwargs inculding those left unassigned
            result = f'{cls.__name__}('
            for arg, value in self.kwargs.items():
                result += f'{arg}={value}, '
            for arg in infer_channels_dict:
                if arg not in self.kwargs:
                    result += f'{arg} unassigned, '
            result = result[:-2] + ')'
            return result

    return DelayedInitClass


infer_channels_dict = dict(in_channels=lambda x: x.shape[1])
delayed_in_channels_wrapper = partial(delayed_init_wrapper, infer_delayed_dict=infer_channels_dict)

# wrap all the convolutions from pytorch
Conv1D = delayed_in_channels_wrapper(torch_conv.Conv1d)
Conv2D = delayed_in_channels_wrapper(torch_conv.Conv2d)
Conv3D = delayed_in_channels_wrapper(torch_conv.Conv3d)
ConvTranspose1d = delayed_in_channels_wrapper(torch_conv.ConvTranspose1d)
ConvTranspose2d = delayed_in_channels_wrapper(torch_conv.ConvTranspose2d)
ConvTranspose3d = delayed_in_channels_wrapper(torch_conv.ConvTranspose3d)


if __name__ == '__main__':

    # do not need to specify in_channels at initialization
    model = nn.Sequential(
        Conv1D(out_channels=5, kernel_size=1),
        Conv1D(out_channels=3, kernel_size=1),
        Conv1D(in_channels=None, out_channels=2, kernel_size=1),
        Conv1D(in_channels=2, out_channels=2, kernel_size=1),
        ConvTranspose1d(out_channels=3, kernel_size=1)
    )

    print('before first forward pass:')
    print(model)

    inp = torch.ones(1, 2, 2)
    print('\noutput:')
    print(model(inp))

    print('\nafter first forward pass:')
    print(model)

