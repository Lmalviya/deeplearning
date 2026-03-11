# Deep Learning Index: Core Architectures & Training

## 1. Neural Building Blocks
*   **The Artificial Neuron**
    *   Weights, Biases, and Summation
    *   **Activation Functions**
        *   Sigmoid & Tanh (The Vanishing Gradient Problem)
        *   ReLU and its variants (Leaky ReLU, ELU, SELU)
        *   Softmax (for Classification)
        *   GELU, Swish, Mish (used in BERT, GPT, modern LLMs)
        *   SwiGLU (gated variant used in LLaMA, PaLM, Gemini)
        *   Dying ReLU problem & solutions (weight init, Leaky ReLU)
*   **Feedforward Mechanism**
    *   Layer-wise transformations
    *   Matrix representation of layers
*   **Residual / Skip Connections**
    *   The math: `y = F(x) + x`
    *   Why they fix vanishing gradients in deep networks
    *   Universal Approximation Theorem — why depth + width matters
*   **Normalisation Layers**
    *   Batch Normalization — mean/var over batch+spatial
    *   Layer Normalization — mean/var over features (used in Transformers)
    *   RMSNorm — simplified LayerNorm without mean subtraction (LLaMA, Mistral)
    *   Instance and Group Normalization
    *   Pre-Norm vs Post-Norm — training stability differences

## 2. Training & Optimization
*   **The Learning Loop**
    *   Backpropagation algorithm (Error signal flow)
    *   Manual computation vs. Automatic Differentiation
    *   Gradient Accumulation — simulate large batches on limited VRAM
    *   Gradient Checkpointing — trade compute for memory in deep networks
*   **Optimizers**
    *   Stochastic Gradient Descent (SGD)
    *   Momentum-based SGD
    *   Adaptive Methods (AdaGrad, RMSProp, Adam, AdamW)
    *   Lion Optimizer (memory-efficient, 2023)
*   **Learning Rate Scheduling**
    *   Step Decay, Exponential Decay
    *   Cosine Annealing & Cosine Annealing with Warm Restarts
    *   Linear Warmup + Cosine Decay (standard for Transformers)
    *   OneCycleLR, ReduceLROnPlateau
*   **Regularization Techniques**
    *   L1 and L2 Regularization (Weight Decay)
    *   Dropout and Spatial Dropout
    *   Stochastic Depth / DropPath (used in ViT, DeiT, modern CNNs)
    *   Early Stopping
    *   Label Smoothing (prevents overconfident predictions)
*   **Precision & Efficiency**
    *   Mixed Precision Training — FP16 vs BF16 vs FP32
    *   Loss Scaling for FP16 (prevents underflow)
    *   Torch.compile and kernel fusion

## 3. Convolutional Neural Networks (CNN)
*   **Core Operations**
    *   [Convolutions: Kernels, Stride, Padding, Output Size Formula](notes/07_cnn_complete.ipynb)
    *   [Feature Maps, Pooling, Conv Variants, Full Training](notes/07_cnn_complete.ipynb)
    *   Dilated (Atrous) Convolutions — expanded receptive field (WaveNet, DeepLab)
    *   Transposed Convolutions — upsampling (GANs, U-Net decoder)
    *   Depthwise Separable Convolution — MobileNet family
    *   1×1 Convolution / Bottleneck blocks — channel projection
    *   Receptive Field calculation across layers
*   **CNN Architecture Evolution**
    *   LeNet-5 (1998) — the original
    *   AlexNet (2012) — ImageNet breakthrough, ReLU, Dropout
    *   VGG (2014) — depth with small 3×3 kernels
    *   GoogLeNet / Inception (2014) — parallel multi-scale branches
    *   ResNet (2015) — skip connections, 152 layers
    *   DenseNet (2017) — connect every layer to every other layer
    *   EfficientNet (2019) — compound scaling (depth + width + resolution)
    *   ConvNeXt (2022) — CNN redesigned to match ViT inductive biases
*   **Attention in CNNs**
    *   Squeeze-and-Excitation (SE) Networks — channel attention
    *   CBAM — Convolutional Block Attention Module
*   **Transfer Learning**
    *   Pretrained ImageNet features as feature extractors
    *   Fine-tuning strategies: full, partial, head-only
*   **Computer Vision Applications (overview)**
    *   Object Detection: YOLO family, Faster R-CNN, DETR
    *   Semantic Segmentation: FCN, U-Net, DeepLab
    *   Instance Segmentation: Mask R-CNN

## 4. Sequence Modeling (RNNs)
*   **Vanilla RNNs**
    *   [RNN: From Dense networks to Recurrent — full deep dive](notes/09_rnn_comprehensive.ipynb)
    *   Hidden States, Weight Sharing, Three Weight Matrices
    *   Architecture types: One-to-One, One-to-Many, Many-to-One, Many-to-Many
    *   Uni-directional, Bi-directional, Stacked/Multi-layer variations
    *   BPTT (Backpropagation Through Time)
    *   Proof: The Vanishing/Exploding Gradient Problem
*   **Gated Architectures**
    *   [LSTM: Cell State, Gates, Constant Error Carousel](notes/10_lstm.ipynb)
    *   [GRU: Update Gate, Reset Gate — simplified LSTM](notes/11_gru.ipynb)
    *   LSTM vs GRU — parameter count, speed, use-case comparison
*   **Encoder-Decoder with Attention (Pre-Transformer)**
    *   [Seq2Seq + Bahdanau/Luong Attention — complete deep dive](notes/12_seq2seq_attention.ipynb)
    *   Seq2Seq architecture (encoder RNN + decoder RNN)
    *   Bahdanau Attention (additive) — alignment scores, context vector, heatmap
    *   Luong Attention (multiplicative / dot-product / general)
    *   The alignment problem this solves; how this leads directly to Transformers
    *   Teacher forcing and exposure bias
*   **Practical Tips**
    *   [Practical Tips — covered in 12_seq2seq_attention.ipynb](notes/12_seq2seq_attention.ipynb)
    *   Packing padded sequences (`pack_padded_sequence` + `pad_packed_sequence`)
    *   CTC Loss — for speech recognition and OCR (variable-length alignment)
    *   Bidirectional inference limitations (cannot use for generation)

## 5. Embeddings & Classic NLP
*   **The Evolution of Text Representations**
    *   Bag of Words → TF-IDF → Count Vectors
    *   One-hot encoding and its limitations
    *   Word embeddings as dense low-dimensional representations
*   **Classic Word Embeddings**
    *   Word2Vec — Skip-gram and CBOW; Negative Sampling math
    *   GloVe — Global vector co-occurrence factorisation
    *   FastText — Subword embeddings for OOV handling
*   **Contextual Embeddings**
    *   ELMo — context-dependent word vectors (BiLSTM based)
    *   BERT sentence representations — [CLS] token embeddings
    *   Contextual vs Static Embeddings trade-offs
*   **Sentence & Document Embeddings**
    *   Sentence Transformers (SBERT) — siamese network for semantic similarity
    *   Mean pooling vs CLS pooling strategies
*   **Multimodal Embeddings**
    *   CLIP (Contrastive Language-Image Pretraining) — joint vision-language space
    *   ALIGN — scaled version of CLIP with noisy data
    *   OpenCLIP — open-source CLIP variants
*   **Subword Tokenisation**
    *   Byte Pair Encoding (BPE) — GPT family
    *   WordPiece — BERT family
    *   SentencePiece / Unigram — LLaMA, T5
    *   Why tokenisation matters for model vocabulary and performance

## 6. Attention & Transformers
*   **Foundations of Attention**
    *   The transition from sequence models (RNNs/LSTMs) and their bottleneck
    *   The "Attention Is All You Need" paper (Vaswani et al., 2017)
    *   Encoder-Decoder structural paradigm
*   **Scaled Dot-Product Attention — Full Math**
    *   Query, Key, Value matrices: `Q = XW_Q`, `K = XW_K`, `V = XW_V`
    *   Formula: `Attention(Q,K,V) = softmax(QK^T / √d_k) V`
    *   Why divide by `√d_k`? — variance control proof
    *   Attention as a soft lookup / weighted retrieval
*   **Types of Attention Mechanisms**
    *   Self-Attention — sequence attends to itself
    *   Cross-Attention — decoder attends to encoder outputs
    *   Causal / Masked Attention — decoder autoregressive mask
    *   Multi-Head Attention (MHA) — parallel attention heads
    *   Multi-Query Attention (MQA) — share KV across heads (faster inference)
    *   Grouped Query Attention (GQA) — balance of MHA and MQA (LLaMA 3, Mistral)
    *   Sparse Attention — local + global patterns (Longformer, BigBird)
    *   FlashAttention / FlashAttention-2 / FlashAttention-3 — IO-aware exact attention
*   **Positional Encoding**
    *   Absolute Sinusoidal Positional Encoding (original Transformer)
    *   Learned Absolute Positional Encoding (BERT, GPT-2)
    *   Relative Positional Encoding (T5, Music Transformer)
    *   RoPE (Rotary Position Embedding) — LLaMA, Mistral, Qwen (internals)
    *   ALiBi (Attention with Linear Biases) — MPT, BLOOM
    *   Context length extension: YaRN, LongRoPE, PoSE
*   **Core Architecture Components**
    *   Tokenization (BPE, WordPiece, SentencePiece)
    *   Position-wise Feed-Forward Networks (FFN)
    *   SwiGLU / GeGLU feed-forward (gated MLP variant in LLaMA)
    *   Layer Normalization (Pre-Norm vs. Post-Norm) and Residual Connections
    *   KV Cache — how inference stores past key/value pairs for efficiency
*   **Mixture of Experts (MoE)**
    *   Dense vs Sparse MoE
    *   Router / Gating mechanism (Top-k routing)
    *   Expert capacity and load balancing loss
    *   Mixtral-8x7B, DeepSeek-MoE, Grok, Gemini architecture
*   **State Space Models (SSM) — 2024 Research**
    *   Structured State Space (S4) — linear recurrent models
    *   Mamba (2023) — selective state spaces, input-dependent SSM
    *   Mamba-2 (2024) — structured state space duality
    *   Hybrid architectures: Jamba (Mamba + Transformer), Zamba
    *   SSM vs Attention: trade-offs in memory, parallelism, long-context
*   **Transformer Architecture Types**
    *   Encoder-only: BERT, RoBERTa, ALBERT, DeBERTa
    *   Decoder-only: GPT series, LLaMA, Mistral, Falcon
    *   Encoder-Decoder: T5, BART, mT5
    *   Vision Transformers: ViT, DeiT, Swin Transformer
    *   Efficient Transformers: Longformer, BigBird, Performer
