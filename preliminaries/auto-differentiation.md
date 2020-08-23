#### A simple example

As a toy example, say that we are interested in differentiating the function 𝑦=2**𝐱**⊤**𝐱** with respect to the column vector **x**.

```python
x = torch.arange(4.0)
x.requires_grad_(True) # Same as 'x = torch.arange(4.0, requires_grad=True)'
x.grad # default None

y = 2 * torch.dot(x, x)
y
tensor(28., grad_fn=<MulBackward0>)

y.backward()
x.grad
tensor([ 0.,  4.,  8., 12.])

# PyTorch accumulates the gradient in default, we need to clear the previous values
x.grad.zero_()
```

#### Detaching

Sometimes, we wish to move some calculations outside of the recorded computational graph.

Imagine that we wanted to calculate the gradient of `z` with respect to `x`, but wanted for some reason to treat `y` as a constant.

Here, we can detach `y` to return a new variable `u` that has the same value as `y` but discards any information about how `y` was computed in the computational graph. In other words, the gradient will not flow backwards through `u` to `x`.

```python
TENSORFLOW
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

