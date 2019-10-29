## convnets.nn

`convents.nn` is a drop-in replacement for `torch.nn` where layer parameters that can be inferred from the input, such as the number of input channels, do not have to be specified at initialization.

It works by wrapping the classes of layers in `torch.nn` and only calling the `__init__` of the original class during the first forward pass. 
After that, everything is fixed and the layer will be indistinguishable from a layer from `torch.nn` that was initialized in the usual way.
Saving and loading models is possible with `torch.save` and `torch.load` as usual, after all layers are fully initialized (i.e. after the first forward pass).

### Example

```python
import torch
import confnets.nn as nn  # replace torch.nn with confnets.nn

model = nn.Sequential(
    nn.Conv1d(out_channels=5, kernel_size=1),      # No need to specify the number of in_channels!
    nn.BatchNorm1d(),                              # Same for num_features in e.g. BatchNorm.
    nn.Conv1d(in_channels=nn.INIT_DELAYED,         # Use nn.INIT_DELAYED to make it more explicit.
              out_channels=2, kernel_size=1),
    nn.Conv1d(nn.INIT_DELAYED, 5, kernel_size=1),  # Positional arguments work as well.
    nn.Conv1d(in_channels=5,                       # Of cause, you can still specify in_channels directly.
              out_channels=3, kernel_size=1)
)

print('before first forward pass:', model)         # will show not-yet specified arguments
input = torch.randn((1, 3, 10))
result = model(input)
print('\nafter first forward pass:', model)        # now everything looks normal

```
