# Neural Networks

Multi-Layer-Perceptron

## Feedforward

A feedforward network defines a mapping $\mathbf{y} = f(\mathbf{x; \theta})$ and learns the value of the parameters $\mathbf{\theta}$ that result in the best function approximation.

Feedforward: Information flows from the function evaluated at $\mathbf{x}$ through the intermediate layers to the output $\mathbf{y}$ without feedback.

The network is composed of multiple different functions, e.g.
$$
f(\mathbf{x}) = f^{(3)}(f^{(2)}(f^{(1)}(\mathbf{x})))
$$
whereby the length of the chain is the *depth* of the model.

The final layer is the output layer.

The behavior of intermediate layers is not specified but rather their function is determined by the algorithm making them **hidden layers**.

Extending linear models to non-linear function through a feature mapping $\phi(\mathbf{x})$. Rather than specifying a feature mapping the algorithm learns the mapping.
$$
y = f(\mathbf{x}; \theta, \mathbf{w}) = \phi(\mathbf{x}; \theta)^T\mathbf{w}
$$

### Activation function

**ReLU (Rectified Linear Unit):** 

* $g(x) = \max\{0, y\}$
* Piecewise linear function suitable for gradient descent.

### Gradient-Based Learning

Problem is non-convex s.t. there is no guarantee to converge on a global minimum. 

**Cross-entropy**: The cross entropy between the training data and the model distribution is used as the cost function.
$$
J(\theta) = - \mathbb{E}_{\mathbf{x, y} \sim \hat p_{\text{data}}}\log p_{\text{model}}(\mathbf{y|x})
$$
**Normal distribution:** For the Normal distribution $p_{\text{model}}(\mathbf{y\vert x}) = \mathcal{N}(\mathbf{y}; f(\mathbf{x; \theta}), \mathbf{I})$, the cross-entropy is the mean squared loss
$$
J(\theta) = \frac{1}{2} \mathbb{E}_{\mathbf{x, y} \sim \hat p_{\text{data}}}||\mathbf{y} - f(\mathbf{x;\theta})||^2 + \text{const}
$$
**Learning conditional statistics rather than full distributions**



The perceptron is not suitable for gradient descent and data that is not linearly separable.
$$
f(x) = \frac{1}{1 + \exp(-x)}
$$

## Backpropagation algorithm

Backpropagation stands for backward propagation of errors

Given a network with a fixed architecture (neuros and interconnections)

Use gradient descent to minimize the squared error between the network output valeu $o$ and the groudn truth $y$

Search for all possible weight values

#### Feedforward-Backpropagation

<img src="/Users/merlinheidemanns/Library/Application Support/typora-user-images/Screen Shot 2020-06-24 at 2.00.48 PM.png" alt="Screen Shot 2020-06-24 at 2.00.48 PM" style="zoom:50%;" />

* **Input:** training examples $(x, y)$, learning rate $\alpha$, $n_1$, $n_k$, and $n_o$
* **Output** A neural network with one input layer, one hidden layer with  $n_i$, $n_h$, $n_o$ 

![Screen Shot 2020-06-24 at 2.05.22 PM](/Users/merlinheidemanns/Library/Application Support/typora-user-images/Screen Shot 2020-06-24 at 2.05.22 PM.png)













