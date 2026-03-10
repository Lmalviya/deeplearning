import nbformat as nbf

nb = nbf.v4.new_notebook()

markdown_1 = """# Convolutional Neural Networks (CNNs): The Core Mechanics

Welcome to the first critical phase of understanding Convolutional Neural Networks. In this notebook, we won't jump into building a full network just yet. Instead, we are going to deeply dissect the fundamental engine that powers them: **The Convolution Operation**.

## 1. Intution

Imagine you are looking at a large painting. You don't take in the entire painting at once in high detail. Instead, your eyes act like a magnifying glass, scanning small sections of the painting one by one, extracting patterns like edges, textures, and colors. Your brain then pieces these local patterns together to understand the whole picture.

A **Convolution** in a Neural Network does exactly this. It's a specialized mathematical operation that slides a "magnifying glass" (called a filter or kernel) over an image to find specific features.

## 2. Simple Explanation

Unlike traditional Dense (Fully Connected) Networks where every input is connected to every neuron (which is incredibly computationally expensive for images and completely destroys 2D spatial relationships), CNNs use convolutions to process data in small, localized chunks. 

A convolution simply takes a small matrix of numbers (the filter) and multiplies it against a small patch of the input image, summing the result into a single number. It repeats this across the entire image.

### Terminology: Filter vs. Kernel
While often used interchangeably in tutorials, there is a technical difference, especially when dealing with color (multi-channel) images:
*   **Kernel:** A simple 2D array of weights. For example, a $3 \\times 3$ matrix.
*   **Filter:** A 3D structure comprising multiple kernels. If your input image is RGB (3 channels), a single *filter* will actually be composed of three *kernels* (one for Red, one for Green, one for Blue). The results of these three kernels are combined and summed together to produce a single 2D feature map.
"""

markdown_2 = """# 3. Deep Dive: The Mathematics and Manual Calculation

At its core, the convolution operation is an element-wise multiplication followed by a sum.

### The Formula
For a 2D input matrix $I$ and a 2D kernel $K$ of size $m \\times n$, the discrete convolution operation at position $(i, j)$ is often implemented in deep learning libraries as *cross-correlation*:

$$ S(i,j) = (I * K)(i,j) = \\sum_{m} \\sum_{n} I(i+m, j+n) K(m,n) $$

*(Note: True mathematical convolution requires flipping the kernel by 180 degrees before multiplication, but in Deep Learning, we don't flip it because the weights within the kernel are learned autonomously anyway. The operation is technically cross-correlation, but everyone in Deep Learning calls it convolution).*

### Manual Calculation Example (2D)

Let's calculate a 2D convolution. We'll use a stride of 1 and "valid" padding (which just means no padding, we only calculate where the kernel fits perfectly).

**Input Image ($3 \\times 3$):**
$$ \\begin{bmatrix} 1 & 2 & 3 \\\\ 4 & 5 & 6 \\\\ 7 & 8 & 9 \\end{bmatrix} $$

**Kernel ($2 \\times 2$):**
$$ \\begin{bmatrix} 1 & 0 \\\\ 0 & -1 \\end{bmatrix} $$

**Step-by-Step Execution:**
The kernel slides over the $3 \\times 3$ input. Since the kernel is $2 \\times 2$, there are 4 possible positions (top-left, top-right, bottom-left, bottom-right).

**Position 1 (Top-Left):**
Extract the top-left $2 \\times 2$ patch:
$$ \\begin{bmatrix} 1 & 2 \\\\ 4 & 5 \\end{bmatrix} $$
Multiply element-wise with Kernel and sum: $(1\\times1) + (2\\times0) + (4\\times0) + (5\\times-1) = 1 + 0 + 0 - 5 = -4$

**Position 2 (Top-Right):**
Extract the top-right $2 \\times 2$ patch:
$$ \\begin{bmatrix} 2 & 3 \\\\ 5 & 6 \\end{bmatrix} $$
Calculate: $(2\\times1) + (3\\times0) + (5\\times0) + (6\\times-1) = 2 - 6 = -4$

**Position 3 (Bottom-Left):**
Extract the bottom-left $2 \\times 2$ patch:
$$ \\begin{bmatrix} 4 & 5 \\\\ 7 & 8 \\end{bmatrix} $$
Calculate: $(4\\times1) + (5\\times0) + (7\\times0) + (8\\times-1) = 4 - 8 = -4$

**Position 4 (Bottom-Right):**
Extract the bottom-right $2 \\times 2$ patch:
$$ \\begin{bmatrix} 5 & 6 \\\\ 8 & 9 \\end{bmatrix} $$
Calculate: $(5\\times1) + (6\\times0) + (8\\times0) + (9\\times-1) = 5 - 9 = -4$

**Final Output Feature Map ($2 \\times 2$):**
$$ \\begin{bmatrix} -4 & -4 \\\\ -4 & -4 \\end{bmatrix} $$
"""

markdown_3 = """# 4. Controls: Stride and Padding

When sliding our kernel over the image, we can control how it behaves using two crucial hyper-parameters: **Stride** and **Padding**.

## Stride
**What it is:** The number of pixels the kernel shifts at each step. By default, Stride = 1.
**Why use it:** A larger stride (e.g., Stride = 2) skips pixels. This actively downsamples the spatial dimensions of the image, significantly reducing computational cost and focusing the network on wider, more global features.
*   **Analogy:** Instead of taking baby steps (stride 1) checking every inch of a path, you take giant leaps (stride 2), getting to the end faster but examining the ground in less detail.

## Padding
**What it is:** Adding a border of "fake" pixels (usually zeros, hence called "Zero Padding") around the edges of the input image before applying the convolution.
**Why use it:** 
1.  **Preserve Dimensions:** Every time you apply a convolution (without padding), the output image shrinks (as seen in our manual example where a $3\\times3$ became $2\\times2$). Padding allows you to keep the spatial dimensions exactly the same (known as "Same Padding").
2.  **Preserve Edge Information:** Without padding, pixels in the center of the image are passed over multiple times by the kernel as it slides, while pixels on the extreme edges are only touched once. Padding ensures edge pixels contribute more equally to the output.

## The Output Dimension Formula

There is a master equation you need to memorize. If you have a square input image of size $W \\times W$, a kernel of size $F \\times F$, Padding amount $P$, and Stride $S$, the square output size $O \\times O$ is calculated as:

$$ O = \\lfloor \\frac{W - F + 2P}{S} \\rfloor + 1 $$

*(Where $\\lfloor \\cdot \\rfloor$ means rounding down to the nearest integer).*
"""

code_1 = """import numpy as np
import torch
import torch.nn.functional as F

print("--- Verifying Manual Calculation with PyTorch ---")

# 1. Define Input
# PyTorch expects shape: (Batch Size, Channels, Height, Width)
# We have a single 1-channel 3x3 input image
input_matrix = torch.tensor([[[[1., 2., 3.],
                               [4., 5., 6.],
                               [7., 8., 9.]]]])

# 2. Define Kernel
# PyTorch expects shape: (Output Channels, Input Channels, Height, Width)
# We have a single 1-channel 2x2 kernel
kernel = torch.tensor([[[[1., 0.],
                         [0., -1.]]]])

# 3. Perform Convolution using functional API
# We used stride 1 and no padding (valid) in our manual example
output = F.conv2d(input_matrix, kernel, stride=1, padding=0)

print("\\nInput Matrix:")
print(input_matrix.squeeze().numpy())

print("\\nKernel:")
print(kernel.squeeze().numpy())

print("\\nOutput Feature Map (PyTorch):")
print(output.squeeze().numpy())
"""

markdown_4 = """# 5. Senior-Level Interview Questions

**Q1: Why are convolutional layers typically placed before fully connected (dense) layers in a CNN architecture?**

**Answer:** They serve distinctly different roles. Convolutional layers act as localized **feature extractors**. They exploit the spatial structure of an image by learning local, hierarchical patterns (e.g., edges $\\rightarrow$ shapes $\\rightarrow$ objects) while remaining incredibly parameter efficient due to *weight sharing*. Fully connected layers, conversely, are global reasoners. They take the high-level, flattened feature representations distilled by the convolutional layers to make final probability classifications. Starting a network with Fully Connected layers mapping directly to raw pixels would instantly destroy structural spatial relationships and result in an astronomical, un-trainable number of parameters.

**Q2: What is the exact difference between "Valid" padding and "Same" padding?**

**Answer:**
*   **Valid Padding:** This simply means *no padding* is applied ($P=0$). The filter only operates on valid input pixels where it can fully fit inside the boundaries. Consequently, the output feature map shrinks spatially.
*   **Same Padding:** This means adding just enough symmetrical zero-padding around the input boundary so that the output feature map has the *exact same* width and height as the original input (assuming your stride is 1). 

**Q3: Explain the concept of 1x1 convolutions (so-called "Bottleneck Layers"). Why are they so heavily used in modern architectures like ResNet or Inception?**

**Answer:** A $1 \\times 1$ convolution might intuitively sound useless since it only looks at a single spatial pixel ($1 \\times 1$). However, the key is that it operates across the *entire depth* (all channels) of the input volume! They are primarily used for two critical reasons:
1.  **Dimensionality Reduction:** They act as an efficient way to reduce the number of channels (feature maps), greatly decreasing the volume of parameters and computational cost before passing the data into more expensive $3 \\times 3$ or $5 \\times 5$ convolutions (creating a "bottleneck" in the network pipe).
2.  **Adding Non-Linearity:** Though the convolution itself is just a linear channel-wise sum, they are immediately followed by an activation function (like ReLU). This adds complex non-linear combinations of the existing features across the depth dimension, all without altering the spatial height and width.

**Q4: Mathematical Sanity Check:**
**If an input volume is $32 \\times 32 \\times 3$ and you apply 10 individual filters of size $5 \\times 5$ with a stride of 1 and padding of 2, what is the dimension of the output volume, and exactly how many learnable parameters are there?**

**Answer:**
*   **Output Dimensions:** Apply the standard formula for spatial size: $O = \\frac{W - F + 2P}{S} + 1 \\Rightarrow \\frac{32 - 5 + 2(2)}{1} + 1 = 32$. The spatial dimensions remain $32 \\times 32$. Because we explicitly applied 10 distinct filters, our resulting output depth is identically 10. The final output volume is: **$32 \\times 32 \\times 10$**.
*   **Learnable Parameters:** Let's calculate for a single filter first. Each filter is spatially $5 \\times 5$, but importantly it must extend through the full depth of the input volume (which is $3$). Thus, each individual filter has dimensions $5 \\times 5 \\times 3 = 75$ weights. Every filter also natively has $1$ bias term. Therefore, a single filter has $75 + 1 = 76$ parameters. Since we are using 10 filters, the total parameter count is $10 \\times 76 =$ **$760$ total learnable parameters**.
"""

nb.cells = [
    nbf.v4.new_markdown_cell(markdown_1),
    nbf.v4.new_markdown_cell(markdown_2),
    nbf.v4.new_markdown_cell(markdown_3),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(markdown_4)
]

with open('notes/06_cnn_convolutions.ipynb', 'w') as f:
    nbf.write(nb, f)
    
print("Notebook generated successfully!")
