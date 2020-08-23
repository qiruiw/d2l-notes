#### Saving Memory

Running operations can cause new memory to be allocated to host results.
This might be undesirable for two reasons:

1. Not want to allocate memory unnecessarily all the time. Typically, we want to update parameters **in place**.
1. We might point at the same parameters from multiple variables

We can assign the result of an operation to a previously aloocated array with slice notation `X[:] = <expression>`.
For arithmetic operations like `*` or `+`, It can also be achieved by `X += Y`.

#### Conversion to Other Python Objects

The converted result does not share the same memory. To numpy: `A = X.numpy()`

To convert a size-1 tensor to a Python scalar, use `a.item()`, `float(a)` and `int(a)`.

#### Dimenionality Reduction

Reducing a matrix along both rows and columns via summation is equivalent to summing up all the elements of the matrix.
`A.sum(axis=[0, 1])  # Same as A.sum()`

However, sometimes it can be useful to keep the number of axes unchanged when calculating the sum or mean.

```python
sum_A = A.sum(axis=1, keepdims=True)
sum_A
tensor([[ 6.],
        [22.],
        [38.],
        [54.],
        [70.]])
```

Since `sum_A` still keeps its two axes after summing each row, we can divide `A` by `sum_A` with broadcasting.

If we want to calculate the cumulative sum of elements of `A` along some axis, say `axis=0` (row by row), we can call the `cumsum` function. (`A.cumsum(axis=0)`) This function will not reduce the input tensor along any axis.

#### Linear Algebra

Matrix-Vector Products: `torch.mv(A, x)`
Matrix Multiplication: `torch.mm(A, B)`

#### Norms

L2-norm: `torch.norm(u)`
L1-norm: `torch.abs(u).sum()`
Forbenius-norm: `torch.norm(A)`

