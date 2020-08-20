### Data Manipulation

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



