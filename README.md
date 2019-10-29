## ConfNets: Highly configurable U-Nets for PyTorch

![U-Net_structure.jpg](diagrams/UNet.jpg)

This is a work in progress, a most things will continue to change.

Features of this repository:
- Simple sharing of pytorch models, only dependencies to load a model are pytorch, numpy and this repo.
- Generality: Easily tweak most hyperparameters, such as model depht, number of features features, activation and normalization type, e.g. from a config file using [speedrun](https://www.github.com/inferno-pytorch/speedrun).
- Extensibility: Implement your own models using general base classes, such as the [U-Net skeleton](https://github.com/imagirom/ConfNets/blob/master/confnets/models/unet.py#L85).
- Easy [side outputs](https://github.com/imagirom/ConfNets/blob/master/confnets/wrappers/multi_io.py#L14), logging of intermediate activations
- A [drop-in replacement](https://github.com/imagirom/ConfNets/tree/master/confnets/nn) of `torch.nn`, where layer parameters such as number of input channels are inferred during the first forward pass, instead of having to specify them at initialization.
