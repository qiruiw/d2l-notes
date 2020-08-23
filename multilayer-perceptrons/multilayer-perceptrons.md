#### Linear Models May Go Wrong

Linearity implies the *weaker* assumption of *monotonicity*: that any increase in our feature must either always cause an increase in our model’s output (if the corresponding weight is positive), or always cause a decrease in our model’s output (if the corresponding weight is negative).

We can overcome these limitations of linear models and handle a more general class of functions by incorporating one or more hidden layers. The easiest way to do this is to stack many fully-connected layers on top of each other. Each layer feeds into the layer above it, until we generate outputs. This architecture is commonly called a *multilayer perceptron*, often abbreviated as *MLP*.

#### From Linear to Nonlinear

In order to realize the potential of multilayer architectures, we need one more key ingredient: a nonlinear *activation function* 𝜎 to be applied to each hidden unit following the affine transformation. The outputs of activation functions (e.g., 𝜎(⋅)) are called *activations*. In general, with activation functions in place, it is no longer possible to collapse our MLP into a linear model.

#### Activation Functions

ReLU function: `y = torch.relu(x)`

Sigmoid function: `y = torch.sigmoid(x)`

Tanh function: tanh(x) = (1-exp(-2x))/(1+exp(-2x)) `y = torch.tanh(x)`

#### Concise Implementation of Multilayer Perceptrons

```python
import torch
from torch import nn

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

# Sequential(
#   (0): Flatten()
#   (1): Linear(in_features=784, out_features=256, bias=True)
#   (2): ReLU()
#   (3): Linear(in_features=256, out_features=10, bias=True)
# )
```

