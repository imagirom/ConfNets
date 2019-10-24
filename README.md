## ConfNets: Highly configurable U-Nets for PyTorch

Goals of this repository:
- Simple sharing of pytorch models, only dependenciss to load a model should be pytorch and this repo
- Generality: Tweak key Hyperparameters easily from a config file 
(e.g. with [speedrun](https://www.github.com/inferno-pytorch/speedrun)).
- Easy side outputs, logging of intermediate activations

---

Current structure of the U-Net Skeleton:
![U-Net_structure.jpg](images/U-Net_structure.jpg)
Make sure to add an appropriate final layer via the `final_activation` argument if you do not want your model 
to end with an activation function.
