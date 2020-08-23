#### Vanishing and Exploding Gradients

Consider a MLP, gradient is the product of 𝐿−𝑙 matrices **𝐌**(𝐿)⋅…⋅**𝐌**(𝑙+1) and the gradient vector **𝐯**(𝑙). Thus we are susceptible to the same problems of numerical underflow that often crop up when multiplying together too many probabilities. 

When dealing with probabilities, a common trick is to switch into log-space, i.e., shifting pressure from the mantissa to the exponent of the numerical representation. 

Unfortunately, our problem above is more serious: initially the matrices **𝐌**(𝑙) may have a wide variety of eigenvalues. They might be small or large, and their product might be *very large* or *very small*.

#### Vanishing Gradients

The sigmoid’s gradient vanishes both when its inputs are large and when they are small. Moreover, when backpropagating through many layers, unless we are in the Goldilocks zone, where the inputs to many of the sigmoids are close to zero, the gradients of the overall product may vanish.

#### Exploding Gradients

```python
M = torch.normal(0, 1, size=(4, 4))
print('a single matrix \n', M)
for i in range(100):
    M = torch.mm(M, torch.normal(0, 1, size=(4, 4)))

print('after multiplying 100 matrices\n', M)

a single matrix
 tensor([[ 0.4868, -1.6793, -1.1682,  0.0326],
        [-0.5070,  0.0198, -0.2048, -0.1458],
        [ 0.4087, -0.2652,  0.4997,  0.2541],
        [ 0.2839,  0.2668, -0.8584,  0.8015]])
after multiplying 100 matrices
 tensor([[-5.4238e+21, -2.0831e+21, -5.4327e+21,  9.9635e+20],
        [-8.7594e+20, -3.3629e+20, -8.7729e+20,  1.6102e+20],
        [ 3.9873e+20,  1.5311e+20,  3.9937e+20, -7.3269e+19],
        [ 1.1302e+21,  4.3360e+20,  1.1317e+21, -2.0801e+20]])
```

#### Parameter Initialization

Default initialization: normal distribution

**Xavier initialization**: 

Let us look at the scale distribution of an output (e.g., a hidden variable) $o_{i}$ for some fully-connected layer *without nonlinearities*. With $n_\mathrm{in}$ inputs $x_j$ and their associated weights $w_{ij}$ for this layer, an output is given by $o_{i} = \sum_{j=1}^{n_\mathrm{in}} w_{ij} x_j.$

The weights $w_{ij}$ are all drawn independently from the same distribution. Furthermore, let us assume that this distribution has zero mean and variance $\sigma^2$. For now, let us assume that the inputs to the layer $x_j$ also have zero mean and variance $\gamma^2$ and that they are independent of $w_{ij}$ and independent of each other. In this case, we can compute the mean and variance of $o_i$ as follows:
$$
\begin{aligned} 
E[o_i] &= \sum_{j=1}^{n_\mathrm{in}} E[w_{ij} x_j] = \sum_{j=1}^{n_\mathrm{in}} E[w_{ij}] E[x_j] = 0, \\ \mathrm{Var}[o_i] &= E[o_i^2] - (E[o_i])^2  = \sum_{j=1}^{n_\mathrm{in}} E[w^2_{ij} x^2_j] - 0  = \sum_{j=1}^{n_\mathrm{in}} E[w^2_{ij}] E[x^2_j] = n_\mathrm{in} \sigma^2 \gamma^2. 
\end{aligned}
$$
One way to keep the variance fixed is to set $n_\mathrm{in} \sigma^2 = 1$. 

Now consider backpropagation. We can also see that the gradients' variance can blow up unless $n_\mathrm{out} \sigma^2 = 1$, where $n_\mathrm{out}$ is the number of outputs of this layer. 

This leaves us in a dilemma: we cannot possibly satisfy both conditions simultaneously. Instead, we simply try to satisfy:

$$ \begin{aligned} \frac{1}{2} (n_\mathrm{in} + n_\mathrm{out}) \sigma^2 = 1 \text{ or equivalently } \sigma = \sqrt{\frac{2}{n_\mathrm{in} + n_\mathrm{out}}}. \end{aligned} $$

This is the reasoning underlying the now-standard and practically beneficial *Xavier initialization*. 

Typically, the Xavier initialization samples weights from a Gaussian distribution with zero mean and variance $\sigma^2 = \frac{2}{n_\mathrm{in} + n_\mathrm{out}}$. 

We can also adapt Xavier's intuition to choose the variance when sampling weights from a uniform distribution. Note that the uniform distribution $U(-a, a)$ has variance $\frac{a^2}{3}$. Plugging $\frac{a^2}{3}$ into our condition on $\sigma^2$ yields the suggestion to initialize according to

$$U\left(-\sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}, \sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}\right).$$













