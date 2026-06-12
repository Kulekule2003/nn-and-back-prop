# Neural Networks and Backpropagation from Scratch

This repository contains a from-scratch implementation of an automatic differentiation engine named `lilygrad`. The project demystifies the core mechanics of backpropagation, the fundamental algorithm used to train neural networks. It builds a `Value` object that tracks computational graphs and computes gradients for all nodes in the graph using the chain rule.

The implementation is heavily inspired by Andrej Karpathy's micrograd engine.

## Features

*   **`Value` Class:** A custom scalar value object that builds a computational graph on the fly.
*   **Automatic Differentiation:** Implementation of the `.backward()` method to automatically compute gradients for all nodes in a graph using a topological sort.
*   **Elementary Operations:** Supports addition, multiplication, powers, and activation functions like `tanh` and `exp`.
*   **Graph Visualization:** A utility function `draw_dot` uses `graphviz` to visualize the computational graph, showing the data and gradient at each node.
*   **Neural Network Components:** Demonstrates building a simple neuron using the `Value` objects.

## Installation

1.  Clone the repository to your local machine:
    ```bash
    git clone https://github.com/kulekule2003/nn-and-back-prop.git
    cd nn-and-back-prop
    ```

2.  Install the required Python packages from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The core of this project is the `Value` class, which wraps a single scalar and enables automatic gradient computation.

### Basic Example: Computing Gradients

You can define a mathematical expression, and `lilygrad` will compute the gradient of the final output with respect to every intermediate value and input.

```python
# From lilygrad.ipynb
from lilygrad import Value

# Define some inputs
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
f = Value(-2.0, label='f')

# Define an expression
e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
L = d * f; L.label = 'L'

# Run backpropagation
L.backward()
```

### Visualizing the Computational Graph

After running backpropagation, you can visualize the entire graph, including the data and gradient at each node.

```python
# From lilygrad.ipynb
from lilygrad import draw_dot

# After running L.backward(), visualize the graph
draw_dot(L)
```

This will generate an SVG image representing the graph, showing the forward pass calculations and the backward pass gradients for `L`, `d`, `e`, `a`, `b`, `c`, and `f`.

### Building a Single Neuron

You can use the `Value` objects to construct a simple neuron and compute its output. The backpropagation mechanism works seamlessly on this structure.

```python
# From lilygrad.ipynb

# Inputs
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

# Weights
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

# Neuron bias
b = Value(6.8813735870195432, label='b')

# The neuron's computation: x1*w1 + x2*w2 + b
x1w1 = x1 * w1; x1w1.label = 'x1w1'
x2w2 = x2 * w2; x2w2.label = 'x2w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1 + x2w2'
n = x1w1x2w2 + b; n.label = 'n'

# Apply tanh activation function
o = n.tanh(); o.label = 'o'

# Perform backpropagation
o.backward()

# Visualize the neuron's computational graph and gradients
draw_dot(o)
```

## Repository Contents

*   `lilygrad.ipynb`: The main Jupyter notebook containing the full implementation of the `Value` class, the backpropagation logic (`backward` method), and examples of its use, from simple expressions to a complete neuron.
*   `graphviz.ipynb`: A supplementary notebook with simple examples demonstrating the usage of the `graphviz` library for creating directed graphs.
*   `requirements.txt`: A file listing all the Python dependencies required to run the notebooks.
