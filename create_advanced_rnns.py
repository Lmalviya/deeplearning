import nbformat as nbf

nb = nbf.v4.new_notebook()

markdown_1 = r"""# RNN Phase 3: Structural Variations (Bidirectional & Deep RNNs)

In the previous phases, we mastered the core "Vanilla" RNN and the gated "LSTM/GRU" units. Now, we look at how to arrange these units to build more powerful, context-aware systems.

## 1. Bidirectional RNNs: Seeing the Future

Standard RNNs are "causal"—they only know about the past. But what if you are translating a sentence? To translate the 3rd word, it's often helpful to know the 4th and 5th words.

**Bidirectional RNNs (Bi-RNNs)** solve this by running two separate RNNs on the same input:
1.  **Forward RNN:** Processes from $t=1$ to $t=T$.
2.  **Backward RNN:** Processes from $t=T$ down to $t=1$.

### The Hidden State Concatenation
At each time step $t$, we concatenate the hidden states from both directions:
$$ H_t = [\overrightarrow{h}_t ; \overleftarrow{h}_t] $$

*   **Result:** Vector $H_t$ now contains information about everything that came **before** $t$ AND everything that comes **after** $t$.
"""

markdown_2 = r"""## 2. Stacked (Deep) RNNs: Hierarchical Features

Just like CNNs have multiple layers to learn edges $\rightarrow$ shapes $\rightarrow$ objects, RNNs can be stacked to learn hierarchical temporal patterns.

*   **Lower Layers:** Capture raw, granular transitions (e.g., character-level or phoneme-level patterns).
*   **Higher Layers:** Capture abstract, long-term themes (e.g., semantic meaning or sentence structure).

### The "Return Sequences" Rule
When stacking RNNs, the lower layers **MUST** return their full sequence of hidden states (not just the final one), so the next layer has an input for every time step.
"""

code_1 = """import torch
import torch.nn as nn

# Setup
input_size = 10
hidden_size = 20
num_layers = 3  # Stacked depth
seq_len = 5
batch_size = 2

# 1. Bidirectional LSTM
# Note: hidden_size remains 20, but the output will be 40 (20*2)
bi_lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
input_data = torch.randn(batch_size, seq_len, input_size)

out_bi, _ = bi_lstm(input_data)
print(f"Bi-LSTM Output Shape: {out_bi.shape} (Note the 40 at the end)")

# 2. Stacked (Deep) LSTM
# num_layers=3 creates 3 LSTMs on top of each other
deep_lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)

out_deep, (h_n, c_n) = deep_lstm(input_data)
print(f"\\nDeep LSTM Output Shape: {out_deep.shape}")
print(f"Deep LSTM Hidden State Shape: {h_n.shape} (3 layers, batch, hidden)")
"""

markdown_3 = r"""## 3. Senior-Level Interview Questions

**Q1: Why can't we use Bidirectional RNNs for real-time speech-to-text (live captions)?**

**Answer:** Because Bi-RNNs require the **entire sequence** to be available before they can start the backward pass. In a live streaming scenario, the "future" hasn't happened yet. If you used a Bi-RNN, you would have to wait for the speaker to finish the entire sentence before any text appeared, introducing unacceptable latency. 

**Q2: When stacking RNNs, what is the risk of making the network too deep (e.g., 10+ layers)?**

**Answer:** 
1.  **Vanishing Gradients (Vertical):** Just as gradients vanish through time (horizontal), they also vanish through layers (vertical). Even with LSTMs, extremely deep stacks are hard to train without **Residual Connections** (skip-connections) between layers.
2.  **Computational Cost:** RNNs are inherently sequential and hard to parallelize. Stacking them multiplies the training time significantly.

**Q3: If you have a Bi-LSTM with `hidden_size=128`, what is the dimensionality of the output at each time step?**

**Answer:** The dimensionality will be **256**. This is because the output is the concatenation of the 128-dimensional forward hidden state and the 128-dimensional backward hidden state.
"""

nb.cells = [
    nbf.v4.new_markdown_cell(markdown_1),
    nbf.v4.new_markdown_cell(markdown_2),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(markdown_3)
]

with open('notes/11_advanced_rnns.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook 11 generated successfully!")
