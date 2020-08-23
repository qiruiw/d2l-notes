### Concise Implementation of Softmax Regression

```python
from d2l import torch as d2l
import torch
from torch import nn

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

#### Defining the Model

```python
# PyTorch does not implicitly reshape the inputs. Thus we define a layer to
# reshape the inputs in our network
class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1,784)

net = nn.Sequential(Reshape(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

# Sequential(
#   (0): Reshape()
#   (1): Linear(in_features=784, out_features=10, bias=True)
# )
```

#### Revisiting Softmax

1. If exp(..) is larger than the largest number we can have for scertain data types (overflow), we could first subtract max(o_k) from all o_k before proceeding with the softmax calculation.
1. After normalization, it might be possible that some o_j have large negative values and thus that the corresponding exp⁡(o_j) will take values close to zero. Then log(exp(o_j)) gives us -inf.

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghzrz2262uj30lc0bkt9q.jpg" alt="image-20200822034049658" style="zoom:45%;" />

```python
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```



















