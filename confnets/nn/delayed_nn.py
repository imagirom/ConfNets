import torch as _torch
from functools import partial as _partial
from inspect import signature as _signature, _POSITIONAL_OR_KEYWORD
import torch.nn as old
from torch.nn import *  # such that unmodified modules can also be imported from here

"""
This is a drop-in replacement of torch.nn, with the option to not specify the number of input channels at initialization
of modules that require them. Hence, constructing complex models can sometimes be less of a headache.
"""

# TODO: add option to have verbose delayed __init__
# TODO: make it possible to load state_dicts

INIT_DELAYED = None  # Tag for arguments that are left unassigned at first init


def _delayed_init_wrapper(cls, **infer_delayed_dict):
    """
    :param cls: type
    class to wrap
    :param infer_delayed_dict: dict
    Keys are the arguments that are inferred only at first forward.
    The values are the functions that get the arguments based on the input to forward.
    :return: type
    """
    assert issubclass(cls, old.Module)
    sig = _signature(cls)

    class DelayedInitClass(cls):

        def __init__(self, *args, init_tensor=None, **kwargs):
            # save kwargs for delayed init
            self.kwargs = kwargs

            # if positional args are specified, get corresponding names and also save in self.kwargs
            # positional only arguments are not supported
            if len(args) > 0:
                parameters = sig.parameters
                # get available argument names
                available_argnames = []
                for argname in list(parameters.keys()):
                    if parameters[argname].kind == _POSITIONAL_OR_KEYWORD:
                        available_argnames.append(argname)
                    else:
                        break

                assert len(available_argnames) >= len(args), \
                    f'Too many positional arguments! {cls.__name__} only takes {len(available_argnames)} ' \
                    f'positional arguments but got {len(args)}.'

                argnames = available_argnames[:len(args)]
                for arg, name in zip(args, argnames):
                    assert name not in kwargs, f'Argument {name} specified both as positional and keyword argument.'
                    kwargs[name] = arg

            # if all arguments that could be inferred later are already specified, do the full init now
            if all([kwargs.get(arg, INIT_DELAYED) is not INIT_DELAYED
                    for arg in infer_delayed_dict]):
                self.do_init()

            # if an in_tensor is given, call forward now to infer unassigned arguments
            if init_tensor is not None:
                with _torch.no_grad():
                    self(init_tensor)

        def __call__(self, *input):
            # infer unassigned arguments from input
            for arg, infer_arg in infer_delayed_dict.items():
                if self.kwargs.get(arg, INIT_DELAYED) is INIT_DELAYED:
                    self.kwargs[arg] = infer_arg(*input)

            # do the init, using the updated kwargs
            self.do_init()

            # as the class of self is now cls, this is now the usual call of the base class
            return self.__call__(*input)

        def do_init(self):
            # call the init of the base class
            super(DelayedInitClass, self).__init__(**self.kwargs)

            # clean up: delete kwargs object
            del self.kwargs

            # aet the class to the base class. This should remove all traces of this class ever existing.
            assert len(self.__class__.__bases__) == 1, f'{self.__class__.__bases__}'
            self.__class__ = self.__class__.__bases__[0]

        def __repr__(self):
            # repr is similar of that for nn.Module, showing all kwargs including those left unassigned
            result = f'{cls.__name__}('
            for arg, value in self.kwargs.items():
                result += f'{arg}={value}, '
            for arg in infer_delayed_dict:
                if arg not in self.kwargs:
                    result += f'{arg} unassigned, '
            result = result[:-2] + ')'
            return result

    # use docstring and __init__ signature of wrapped class
    DelayedInitClass.__doc__ = cls.__doc__
    # TODO: add init_tensor to signature
    DelayedInitClass.__signature__ = sig

    return DelayedInitClass


def _infer_in_channels(in_tensor):
    assert isinstance(in_tensor, _torch.Tensor), f'Can only infer in_channels from tensor. Got {type(input)}'
    return in_tensor.shape[1]


def _infer_input_size(in_tensor):
    assert isinstance(in_tensor, _torch.Tensor), f'Can only infer in_channels from tensor. Got {type(input)}'
    return in_tensor.shape[-1]


# define wrapping functions for different init arguments
_wrap_in_and_out_channels = _partial(_delayed_init_wrapper, in_channels=_infer_in_channels, out_channels=_infer_in_channels)
_wrap_num_channels = _partial(_delayed_init_wrapper, num_channels=_infer_in_channels)
_wrap_in_features = _partial(_delayed_init_wrapper, in_features=_infer_in_channels)
_wrap_num_features = _partial(_delayed_init_wrapper, num_features=_infer_in_channels)
_wrap_input_size = _partial(_delayed_init_wrapper, input_size=_infer_input_size)
_wrap_layer_norm = _partial(_delayed_init_wrapper, normalized_shape=lambda x: x.shape[1:])
_wrap_bilinear = _partial(
    _delayed_init_wrapper, 
    in1_features=lambda x, y: _infer_input_size(x), 
    in2_features=lambda x, y: _infer_input_size(y)
)

# wrap convolutional layers
Conv1d = _wrap_in_and_out_channels(old.Conv1d)
Conv2d = _wrap_in_and_out_channels(old.Conv2d)
Conv3d = _wrap_in_and_out_channels(old.Conv3d)
ConvTranspose1d = _wrap_in_and_out_channels(old.ConvTranspose1d)
ConvTranspose2d = _wrap_in_and_out_channels(old.ConvTranspose2d)
ConvTranspose3d = _wrap_in_and_out_channels(old.ConvTranspose3d)

# wrap normalization layers
BatchNorm1d = _wrap_num_features(old.BatchNorm1d)
BatchNorm2d = _wrap_num_features(old.BatchNorm2d)
BatchNorm3d = _wrap_num_features(old.BatchNorm3d)
if hasattr(old, "SyncBatchNorm"):
    SyncBatchNorm = _wrap_num_features(old.SyncBatchNorm)
GroupNorm = _wrap_num_channels(old.GroupNorm)
InstanceNorm1d = _wrap_num_features(old.InstanceNorm1d)
InstanceNorm2d = _wrap_num_features(old.InstanceNorm2d)
InstanceNorm3d = _wrap_num_features(old.InstanceNorm3d)
LayerNorm = _wrap_layer_norm(old.LayerNorm)

# wrap recurrent layers
RNNBase = _wrap_input_size(old.RNNBase)
RNN = _wrap_input_size(old.RNN)
RNNCell = _wrap_input_size(old.RNNCell)
LSTMCell = _wrap_input_size(old.LSTMCell)
GRUCell = _wrap_input_size(old.GRUCell)

# wrap other layers
Linear = _wrap_in_features(old.Linear)
AdaptiveLogSoftmaxWithLoss = _wrap_in_features(old.AdaptiveLogSoftmaxWithLoss)
Bilinear = _wrap_bilinear(old.Bilinear)

# nn.PReLU is not wrapped since num_parameters can be either the number of input channels or 1.
# TODO: wrap Transformers


if __name__ == '__main__':
    # TODO: move to test script
    import torch
    # do not need to specify in_channels at initialization
    model = old.Sequential(
        Conv1d(out_channels=5, kernel_size=1),
        BatchNorm1d(),
        Conv1d(in_channels=INIT_DELAYED, out_channels=2, kernel_size=1),
        GroupNorm(num_groups=2),
        Conv1d(in_channels=2, out_channels=2, kernel_size=1),
        Conv1d(INIT_DELAYED, 2, kernel_size=1),  # positional arguments work, too
        InstanceNorm1d(track_running_stats=True),
        ConvTranspose1d(out_channels=3, kernel_size=1, init_tensor=torch.zeros(1, 2, 10))
    )

    class TestModel(Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.x = Conv1d(out_channels=5, kernel_size=1)

        def forward(self, input):
            return self.x(input)

    model = TestModel()

    print('before first forward pass:')
    print(model)

    inp = torch.ones(1, 2, 2)
    print('\noutput:')
    print(model(inp))

    print('\nafter first forward pass:')
    print(model)

    torch.save(model, 'test_module.pytorch')
    model = torch.load('test_module.pytorch')
    print(model)
    print(model(inp))
