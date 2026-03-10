import nbformat as nbf

nb = nbf.v4.new_notebook()

markdown_1 = r"""# CNN Phase 2: Feature Maps and Pooling Layers

In Phase 1, we mastered the core engine: the Convolution operation. Now, we'll look at what happens when these operations are stacked and how we manage the resulting data volume.

## 1. Feature Map Intuition: The Hierarchy of Vision

When we apply a filter to an image, the output is a **Feature Map**. 

Think of a feature map as a "heat map" that answers the question: *"Where in the image did I find the pattern this specific filter was looking for?"*

### The Hierarchical Progress
CNNs don't see objects immediately. They build understanding layer by layer:
1.  **Early Layers:** Detect simple **Low-Level Features** (Edges, Corners, Gradients).
2.  **Middle Layers:** Combine edges to detect **Mid-Level Features** (Textures, Simple Shapes like circles or squares).
3.  **Deep Layers:** Combine shapes to detect **High-Level Features** (Complex objects like eyes, wheels, or even entire faces).

Each channel in our output volume represents one of these specific features.
"""

markdown_2 = r"""## 2. Deep Dive: Channel-wise Summation

A common point of confusion is how an input with multiple channels (like an RGB image with 3 channels) results in a single feature map for one filter.

### The Mechanics
If our input has 3 channels ($C_{in}=3$), a single filter is not just a 2D matrix; it is a **3D volume** with a depth of 3.
1.  **Kernel 1** performs convolution on the **Red** channel.
2.  **Kernel 2** performs convolution on the **Green** channel.
3.  **Kernel 3** performs convolution on the **Blue** channel.
4.  **The Result:** The three resulting 2D maps are **summed together** element-wise, and a single **Bias** term is added.

This produces **ONE** 2D feature map. If a convolutional layer has 64 filters, it will repeat this process 64 times, creating an output volume with a depth of 64.

**Math Representation:**
For input $X$ and filter $W$ with $C$ channels:
$$ FeatureMap = \left( \sum_{c=1}^{C} X_c * W_c \right) + b $$
"""

markdown_3 = r"""## 3. Pooling Layers: Downsampling and Invariance

While convolutions extract features, **Pooling** layers are all about efficiency and robustness.

### Why do we need Pooling?
1.  **Reduce Spatial Size:** It shrinks the height and width, drastically reducing the number of parameters and computation for later layers.
2.  **Translation Invariance:** Pooling makes the network robust to small shifts. If a "cat ear" moves slightly to the left, the pooling layer (especially Max Pooling) will likely still pick up the same high activation.
3.  **Prevent Overfitting:** By discarding redundant, fine-grained spatial information, the model focuses on the most important global patterns.

### Types of Pooling
| Type | Mechanism | Best Use Case |
| :--- | :--- | :--- |
| **Max Pooling** | Picks the highest value in the window. | **The Standard.** Excellent for detecting sharp features (edges). |
| **Average Pooling** | Calculates the mean of values in the window. | Used in specific architectures (like GANs or older networks) where smooth transitions matter. |
| **Global Average Pooling (GAP)** | Averages the **entire** feature map into a single number. | Used at the very end of modern networks to replace flattening and prevent overfitting. |
"""

markdown_4 = r"""## 4. Manual Calculation Example: Pooling

Let's apply $2 \times 2$ pooling with a **stride of 2** to a $4 \times 4$ input.

**Input ($4 \times 4$):**
$$ \begin{bmatrix} 1 & 2 & 8 & 9 \\ 3 & 4 & 5 & 1 \\ 0 & 3 & 2 & 6 \\ 4 & 1 & 7 & 8 \end{bmatrix} $$

### Max Pooling (Stride 2)
We look at $2 \times 2$ quadrants:
*   Top-Left $\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \rightarrow$ **Max = 4**
*   Top-Right $\begin{bmatrix} 8 & 9 \\ 5 & 1 \end{bmatrix} \rightarrow$ **Max = 9**
*   Bottom-Left $\begin{bmatrix} 0 & 3 \\ 4 & 1 \end{bmatrix} \rightarrow$ **Max = 4**
*   Bottom-Right $\begin{bmatrix} 2 & 6 \\ 7 & 8 \end{bmatrix} \rightarrow$ **Max = 8**

**Output:** $\begin{bmatrix} 4 & 9 \\ 4 & 8 \end{bmatrix}$

### Average Pooling (Stride 2)
*   Top-Left: $(1+2+3+4)/4 = $ **2.5**
*   Top-Right: $(8+9+5+1)/4 = $ **5.75**
*   Bottom-Left: $(0+3+4+1)/4 = $ **2.0**
*   Bottom-Right: $(2+6+7+8)/4 = $ **5.75**

**Output:** $\begin{bmatrix} 2.5 & 5.75 \\ 2.0 & 5.75 \end{bmatrix}$
"""

code_1 = """import torch
import torch.nn as nn

# Create a 4x4 input
# Shape: (Batch, Channels, Height, Width)
x = torch.tensor([[[[1., 2., 8., 9.],
                    [3., 4., 5., 1.],
                    [0., 3., 2., 6.],
                    [4., 1., 7., 8.]]]])

print("Input Plane:")
print(x.squeeze())

# 1. Max Pooling 2x2, Stride 2
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
print("\\nMax Pooling Output:")
print(max_pool(x).squeeze())

# 2. Average Pooling 2x2, Stride 2
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
print("\\nAverage Pooling Output:")
print(avg_pool(x).squeeze())

# 3. Global Average Pooling
# AdaptiveAvgPool2d(1) forces the output size to be 1x1 regardless of input size
gap = nn.AdaptiveAvgPool2d(1)
print("\\nGlobal Average Pooling Output:")
print(gap(x).squeeze())
"""

markdown_5 = r"""## 5. Senior-Level Interview Questions

**Q1: Contrast Max Pooling and Average Pooling. Why did the industry move almost exclusively toward Max Pooling?**

**Answer:** Max Pooling focuses on the **most activated feature** in a region. In image recognition, features like edges are sparse; out of 4 pixels, maybe only one has a strong edge signal. Max Pooling preserves that signal perfectly. Average Pooling, however, "dilutes" that strong signal by averaging it with its weaker neighbors, effectively blurring the feature map. In most classification tasks, the presence of a feature is more important than its exact average intensity, making Max Pooling superior for discriminative tasks.

**Q2: What is "Global Average Pooling" (GAP), and why is it preferred over a Flatten + Dense layer at the end of a CNN?**

**Answer:** In older architectures (like VGG), feature maps were flattened and fed into huge Fully Connected (FC) layers. This led to:
1.  **Parameter Explosion:** FC layers accounted for >80% of total parameters.
2.  **Overfitting:** So many parameters easily memorized the training data.
3.  **Fixed Input Size:** Flattening requires a fixed input size.

**GAP** solves this by averaging each feature map into a single scalar. If you have 1000 feature maps, you get a 1000-length vector.
*   **Benefits:** It has **zero learnable parameters**, acting as a powerful regularizer. It also allows the network to accept images of **any size**, as the output is always fixed to the number of feature maps.

**Q3: How does pooling contribute to "Translation Invariance"?**

**Answer:** Translation Invariance means the model can recognize an object even if it shifts by a few pixels. Because pooling (especially Max Pooling) selects the maximum value in a $2 \times 2$ or $3 \times 3$ neighborhood, the exact position of the high activation within that neighborhood doesn't matter—the output of the pooling layer will be the same. This makes the network robust to minor spatial jitters or object movements.
"""

nb.cells = [
    nbf.v4.new_markdown_cell(markdown_1),
    nbf.v4.new_markdown_cell(markdown_2),
    nbf.v4.new_markdown_cell(markdown_3),
    nbf.v4.new_markdown_cell(markdown_4),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(markdown_5)
]

with open('notes/07_cnn_feature_maps_pooling.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook 07 generated successfully with raw strings!")
