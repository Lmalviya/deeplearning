# Deep Learning Index: Core Architectures & Training

## 1. Neural Building Blocks
*   **The Artificial Neuron**
    *   Weights, Biases, and Summation
    *   **Activation Functions**
        *   Sigmoid & Tanh (The Vanishing Gradient Problem)
        *   ReLU and its variants (Leaky ReLU, ELU, SELU)
        *   Softmax (for Classification)
*   **Feedforward Mechanism**
    *   Layer-wise transformations
    *   Matrix representation of layers

## 2. Training & Optimization
*   **The Learning Loop**
    *   Backpropagation algorithm (Error signal flow)
    *   Manual computation vs. Automatic Differentiation
*   **Optimizers**
    *   Stochastic Gradient Descent (SGD)
    *   Momentum-based SGD
    *   Adaptive Methods (AdaGrad, RMSProp, Adam, AdamW)
*   **Regularization Techniques**
    *   L1 and L2 Regularization (Weight Decay)
    *   Dropout and Spatial Dropout
    *   Early Stopping
*   **Normalization Layers**
    *   Batch Normalization
    *   Layer Normalization
    *   Instance and Group Normalization

## 3. Convolutional Neural Networks (CNN) - Operations
*   **Core Operations**
    *   Convolutions (Kernels, Stride, Padding)
    *   Feature Map Visualization
    *   Pooling (Max Pooling, Average Pooling, Global Average Pooling)
    *   Dilated and Transposed Convolutions
    *   Receptive Field calculation

## 4. Sequence Modeling (RNNs)
*   **Vanilla RNNs**
    *   Hidden State and Temporal Dependencies
    *   Many-to-One, One-to-Many, Many-to-Many mappings
*   **Gated Architectures**
    *   **LSTM (Long Short-Term Memory)**: Forget gates, Input gates, Output gates
    *   **GRU (Gated Recurrent Unit)**: Reset and Update gates
*   **Structural Variations**
    *   Bidirectional RNNs
    *   Stacked/Deep RNNs
