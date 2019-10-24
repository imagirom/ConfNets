## ConfNets: Highly configurable U-Nets for PyTorch
This is a work in progress, most things are up to change.

Goals of this repository:
- Simple sharing of pytorch models, only dependencies to load a model should be pytorch and this repo (TODO: remove dependecy on inferno)
- Generality: Tweak key Hyperparameters easily from a config file 
(e.g. with [speedrun](https://www.github.com/inferno-pytorch/speedrun)).
- Easy side outputs, logging of intermediate activations (TODO: how to handle logging? Depend on speedruns `log_anywhere`?)

---

Current structure of the U-Net Skeleton (here for depth 3):
![U-Net_structure.jpg](./U-Net_structure.jpg)
Make sure to add an appropriate final layer via the `final_activation` argument if you do not want your model 
to end with an activation function.
