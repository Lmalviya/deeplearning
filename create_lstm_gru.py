import nbformat as nbf

nb = nbf.v4.new_notebook()

markdown_1 = r"""# RNN Phase 2: Gated Architectures (LSTM & GRU)

In Phase 1, we saw how "Vanilla" RNNs struggle with long-term memory due to the **Vanishing Gradient Problem**. Today, we look at the solution: **Gating mechanisms**.

## 1. The Intuition: The Information Highway

Imagine you are a conveyor belt operator in a factory. Items (information) are coming at you fast. You need to decide:
1.  What old junk on the belt should be thrown away? (**Forget Gate**)
2.  What new valuable items should be added? (**Input Gate**)
3.  What is ready to be shipped out to the customer? (**Output Gate**)

**LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)** act exactly like this. They use "gates" (sigmoid layers) to protect and control an internal "Cell State"—a highway of information that can flow through thousands of time steps with very little change.
"""

markdown_2 = r"""## 2. LSTM (Long Short-Term Memory)

The LSTM is the classic gated architecture. It separates the "Long-term memory" (**Cell State**) from the "Short-term/Working memory" (**Hidden State**).

### The Four Steps of an LSTM Unit
1.  **Forget Gate ($f_t$):** Decides what to discard from $C_{t-1}$.
    $$ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) $$
2.  **Input Gate ($i_t$ & $\tilde{C}_t$):** Decides what new info to store.
    $$ i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) $$
    $$ \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) $$
3.  **Update Cell State ($C_t$):** This is the **CRITICAL** step. We add the new info to the old info.
    $$ C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t $$
4.  **Output Gate ($o_t$):** Decides what part of the Cell State becomes the Hidden State.
    $$ o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) $$
    $$ h_t = o_t \odot \tanh(C_t) $$
"""

markdown_3 = r"""## 3. GRU (Gated Recurrent Unit)

The GRU is the "lighter" sibling of the LSTM. It simplifies the architecture by merging the Cell State and Hidden State into one.

### The Two Gates of GRU
1.  **Update Gate ($z_t$):** Combines the roles of the Input and Forget gates. It decides how much of the old memory to keep vs. how much new info to adopt.
2.  **Reset Gate ($r_t$):** Decides how much of the *previous* memory to show to the new candidate state calculation.

**Key Advantage:** Fewer parameters $\rightarrow$ Faster training $\rightarrow$ Less likely to overfit on small datasets.
"""

markdown_4 = r"""## 4. Deep Dive: Why do Gates solve Vanishing Gradients?

In a Vanilla RNN, the gradient is multiplied by the same weight matrix $W$ at every step. This leads to exponential decay (Vanishing).

In an LSTM, the change to the Cell State is **ADDITIVE**:
$$ C_t = f_t \odot C_{t-1} + \dots $$

During backpropagation, the derivative of the cell state $dC_t/dC_{t-1}$ contains the term $f_t$ (the forget gate). If the network learns that certain information is important, it sets $f_t \approx 1$. 

This creates a **linear path** where the gradient can flow through time without being squashed by multiple weight multiplications. It effectively bypasses the vanishing gradient problem!
"""

code_1 = """import torch
import torch.nn as nn

# Setup
input_size = 10
hidden_size = 20
seq_len = 5
batch_size = 3

# 1. Using LSTM
# Output: (output_seq, (last_hidden, last_cell_state))
lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
input_data = torch.randn(batch_size, seq_len, input_size)

out_lstm, (h_n, c_n) = lstm(input_data)
print(f"LSTM Output Seq Shape: {out_lstm.shape}")
print(f"LSTM Cell State Shape: {c_n.shape}")

# 2. Using GRU
# Output: (output_seq, last_hidden) - No separate cell state!
gru = nn.GRU(input_size, hidden_size, batch_first=True)
out_gru, h_n_gru = gru(input_data)
print(f"\\nGRU Output Seq Shape: {out_gru.shape}")
print(f"GRU Last Hidden Shape: {h_n_gru.shape}")
"""

markdown_5 = r"""## 5. Senior-Level Interview Questions

**Q1: What is the most significant architectural difference between LSTM and GRU, and how does it affect performance?**

**Answer:** The most significant difference is that **GRU merges the cell state and hidden state** and uses only two gates (Update and Reset) instead of three. 
*   **Performance Impact:** GRUs have ~33% fewer parameters than LSTMs of the same hidden size. This makes them faster to train and computationally more efficient. However, LSTMs are considered more "expressive" and can sometimes perform better on very complex, long sequences where fine-grained control over what to forget vs. what to output is critical.

**Q2: Explain the "Constant Error Carousel" (CEC) property of LSTMs.**

**Answer:** The CEC refers to the logic that because the cell state update is **additive** rather than multiplicative, the error signal (gradient) can flow back through the cell state without being reduced to zero. As long as the forget gate is close to 1, the gradient is multiplied by ~1 at every step, allowing the error to remain "constant" as it travels back to earlier time steps. This is what allows LSTMs to bridge time gaps of 1000+ steps.

**Q3: When would you use a Tanh activation vs. a Sigmoid activation in an RNN/LSTM gate?**

**Answer:** 
*   **Sigmoid** is used for the **gates** (Forget, Input, Output) because it outputs values between 0 and 1. This acts as a "binary-ish" mask—deciding how much information to let through.
*   **Tanh** is used for the **candidate values** and the **cell state scaling** because it outputs values between -1 and 1. This is mathematically necessary to allow the cell state to both increase and decrease (negative values allowed), and it helps prevent the values in the cell state from growing infinitely as we add more and more info.
"""

nb.cells = [
    nbf.v4.new_markdown_cell(markdown_1),
    nbf.v4.new_markdown_cell(markdown_2),
    nbf.v4.new_markdown_cell(markdown_3),
    nbf.v4.new_markdown_cell(markdown_4),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(markdown_5)
]

with open('notes/10_lstm_gru.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook 10 generated successfully!")
