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
    *   [Convolutions (Kernels, Stride, Padding)](notes/07_cnn_convolutions.ipynb)
    *   [Feature Map Visualization & Pooling](notes/08_cnn_feature_maps_pooling.ipynb) (Max, Average, Global Average)
    *   Dilated and Transposed Convolutions
    *   Receptive Field calculation

## 4. Sequence Modeling (RNNs)
*   **Vanilla RNNs**
    *   [RNN Architecture: From Dense to Recurrent](notes/09_rnn_comprehensive.ipynb)
    *   Hidden States, Weight Sharing, Multi-Channel Components
    *   Uni-directional, Bi-directional, Multi-layer variations
    *   Proof: The Vanishing/Exploding Gradient Problem
*   **Gated Architectures**
    *   [LSTM (Long Short-Term Memory) & GRU (Gated Recurrent Unit)](notes/10_lstm_gru.ipynb)
    *   Forget gates, Input gates, Output gates, Reset and Update gates

## 5. Embeddings & Classic NLP
- [ ] Classic Word Embeddings (Word2Vec, GloVe, FastText)
- [ ] Contextual Embeddings vs. Static Embeddings
- [ ] Sentence Transformers (e.g., SBERT)

## 6. Attention & Transformers
- [ ] **Foundations of Attention**
    - [ ] The transition from sequence models (RNNs/LSTMs)
    - [ ] The "Attention Is All You Need" breakthrough
    - [ ] Encoder-Decoder structural paradigm
- [ ] **Types of Attention Mechanisms**
    - [ ] Self-Attention (Query, Key, Value)
    - [ ] Cross-Attention
    - [ ] Sparse Attention
    - [ ] FlashAttention
- [ ] **Core Architecture Components**
    - [ ] Tokenization (BPE, WordPiece, SentencePiece)
    - [ ] Types of Positional Embedding (Absolute, Relative, RoPE, ALiBi)
    - [ ] Multi-Head Attention
    - [ ] Position-wise Feed-Forward Networks
    - [ ] Layer Normalization (Pre-Norm vs. Post-Norm) and Residual Connections
- [ ] **Transformer Architecture Types**
    - [ ] Encoder-only (e.g., BERT, RoBERTa)
    - [ ] Decoder-only (e.g., GPT series, LLaMA)
    - [ ] Encoder-Decoder (e.g., T5, BART)
    - [ ] Vision Transformers (ViT)
