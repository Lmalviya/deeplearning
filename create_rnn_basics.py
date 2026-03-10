import nbformat as nbf

nb = nbf.v4.new_notebook()

markdown_1 = r"""# RNN Phase 1: The Foundations of Sequence Modeling

Welcome to Chapter 4. In this section, we move from processing static "snapshots" (like images in CNNs) to processing **Sequences**. Whether it's a sentence, a stock price, or a sound wave, the order of data points is now our most valuable feature.

## 1. Intuition: Why Memory Matters

Imagine you are reading a book. If you treated every word as an independent data point (like a standard Dense network does), you would forget the beginning of a sentence by the time you reached the end. 

**Recurrent Neural Networks (RNNs)** solve this by possessing a "hidden state"—a internal memory that carries information from previous time steps to influence the current one.

### The Analogy
*   **Reading a Map (CNN/Dense):** You look at everything at once. The spatial relationship between points matters, but there's no "beginning" or "end" in time.
*   **Reading a Sentence (RNN):** You process one word at a time. The meaning of "bank" in "I went to the bank..." depends entirely on whether the previous words were about "money" or "rivers".
"""

markdown_2 = r"""## 2. The Mechanics: Hidden State and Recursion

In a standard network, data flows in one direction. In an RNN, it **loops**.

At every time step $t$, the RNN takes two inputs:
1.  The current data point $x_t$ (e.g., the current word).
2.  The **Hidden State** from the *previous* step $h_{t-1}$ (the memory).

It then produces a new hidden state $h_t$, which is passed forward to the next step.

### The Equation
$$ h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h) $$
$$ y_t = W_{hy}h_t + b_y $$

*   $W_{xh}$: Weight matrix for the input to hidden transition.
*   $W_{hh}$: Weight matrix for the hidden to hidden transition (**The Recurring Weight**).
*   $W_{hy}$: Weight matrix for hidden to output.
*   **Crucial Point:** These weights are **shared** across all time steps. The same "logic" is applied whether you are at the 1st word or the 100th.
"""

markdown_3 = r"""## 3. Mapping Architectures: One Size Doesn't Fit All

RNNs are incredibly flexible. Depending on the task, we can hook up inputs and outputs in different ways:

| Type | Structure | Real-World Example |
| :--- | :--- | :--- |
| **One-to-Many** | 1 Input $\rightarrow$ Sequence of Outputs | **Image Captioning:** Input 1 image, output a sentence of 10 words. |
| **Many-to-One** | Sequence of Inputs $\rightarrow$ 1 Output | **Sentiment Analysis:** Input a sentence, output a "Positive/Negative" score. |
| **Many-to-Many (Sync)** | Sequence $\rightarrow$ Sequence (Same length) | **POS Tagging:** For every word in a sentence, output its grammatical tag. |
| **Many-to-Many (Async)** | Sequence $\rightarrow$ Sequence (Different length)| **Machine Translation:** Input an English sentence, output a French translation. |
"""

markdown_4 = r"""## 4. The Deep Dive: The Vanishing Gradient Problem

Why are "Vanilla" RNNs (like the one we just described) rarely used in production today? Because they have **terrible long-term memory**.

### The Mathematical Intuition
Imagine we have a sequence of 50 words. To update the weights for the *first* word based on the error at the *50th* word, we have to calculate the gradient through a chain of 50 multiplications of the same weight matrix $W_{hh}$.

If $W_{hh}$ is even slightly less than 1 (specifically, if its largest eigenvalue is < 1), and we multiply it by itself 50 times:
$$ 0.9^{50} \approx 0.005 $$
The gradient becomes so small that the weights for the early words effectively stop updating. The network "forgets" the start of the sequence.

**This is the Vanishing Gradient Problem.** 
*   **Exploding Gradients:** Conversely, if $W > 1$, the gradient becomes infinitely large (fixed by "Gradient Clipping").
*   **Vanishing Gradients:** Much harder to fix; requires architectural changes like **LSTMs** or **GRUs** (covered in Phase 2).
"""

code_1 = """import torch
import torch.nn as nn

# 1. Define Hyperparameters
input_size = 10   # e.g., 10-dimensional word embedding
hidden_size = 20  # size of the internal memory
seq_len = 5       # length of our sentence
batch_size = 3    # number of sentences processed at once

# 2. Create a Vanilla RNN
# batch_first=True makes the input shape (Batch, Seq, Feature)
rnn = nn.RNN(input_size, hidden_size, batch_first=True)

# 3. Create dummy input data
# (Batch, Seq, Feature)
input_data = torch.randn(batch_size, seq_len, input_size)

# 4. Initialize the hidden state (optional, defaults to zeros)
# (Num_Layers, Batch, Hidden_Size)
h0 = torch.zeros(1, batch_size, hidden_size)

# 5. Forward Pass
output, hn = rnn(input_data, h0)

print(f"Input Shape: {input_data.shape}")
print(f"Output Shape (all time steps): {output.shape}")
print(f"Final Hidden State Shape: {hn.shape}")

# Note: The 'output' contains the hidden states for ALL time steps.
# The 'hn' contains ONLY the hidden state from the very last step.
"""

markdown_5 = r"""## 5. Senior-Level Interview Questions

**Q1: Why do we share the same weight matrices (W_xh, W_hh) across all time steps? Why not have different weights for word 1 vs word 2?**

**Answer:** Two main reasons:
1.  **Generalization:** Sequential patterns often don't care about their absolute position. If the model learns that "He" is usually followed by a verb, it should apply that logic whether "He" appears at the start of the sentence or in the middle. Shared weights allow the model to generalize across positions.
2.  **Parameters & Variable Length:** If we had different weights for every step, we would need to know the maximum sequence length in advance, and the number of parameters would explode with the length of the sequence. Shared weights allow a single model to process a sentence of 5 words or 500 words.

**Q2: Contrast "Vanishing" and "Exploding" gradients in RNNs. Which is easier to solve?**

**Answer:** **Exploding Gradients** are easier to solve. You can use **Gradient Clipping**, which simply caps the magnitude of the gradient at a threshold (e.g., if gradient > 5, set it to 5). 
**Vanishing Gradients** are much more insidious because you can't just "clip" a zero value. They reflect a fundamental loss of information across time steps. Solving them requires complex gating mechanisms (LSTMs/GRUs) that provide a "skip-connection" style path for the gradient to flow through time without constant multiplication.

**Q3: In PyTorch's nn.RNN, what is the difference between the `output` and `hn` returned by the forward pass?**

**Answer:** 
*   **`output`**: Contains the hidden states ($h_t$) for **every** time step in the sequence. If your sequence length is 5, it will have 5 vectors per batch item. This is what you use for "Many-to-Many" tasks (like POS tagging).
*   **`hn`**: Contains only the **final** hidden state from the last time step. This is what you would feed into a classifier for "Many-to-One" tasks (like Sentiment Analysis).
"""

nb.cells = [
    nbf.v4.new_markdown_cell(markdown_1),
    nbf.v4.new_markdown_cell(markdown_2),
    nbf.v4.new_markdown_cell(markdown_3),
    nbf.v4.new_markdown_cell(markdown_4),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(markdown_5)
]

with open('notes/09_rnn_basics.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook 09 generated successfully!")
