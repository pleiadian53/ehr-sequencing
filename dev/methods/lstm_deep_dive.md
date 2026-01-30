# LSTM Deep Dive: Memory, Gradients, and PyTorch Usage

**Date:** January 20, 2026  
**Purpose:** Comprehensive understanding of LSTM architecture and implementation

---

## Table of Contents

1. [LSTM Fundamentals](#lstm-fundamentals)
2. [Memory Representation](#memory-representation)
3. [Recursive Processing](#recursive-processing)
4. [Vanishing Gradient Solution](#vanishing-gradient-solution)
5. [Data Shape Transformations](#data-shape-transformations)
6. [PyTorch LSTM Usage](#pytorch-lstm-usage)
7. [Practical Examples](#practical-examples)

---

## LSTM Fundamentals

### What is an LSTM?

**Long Short-Term Memory (LSTM)** is a type of Recurrent Neural Network (RNN) designed to model sequential data while addressing the vanishing gradient problem.

**Key Innovation:** Separate **cell state** (long-term memory) from **hidden state** (short-term memory) using gating mechanisms.

### High-Level Intuition

```
Traditional RNN Problem:
  h_t = tanh(W_h * h_{t-1} + W_x * x_t)
  Problem: Information from h_0 fades quickly (vanishing gradients)

LSTM Solution:
  c_t = long-term memory (cell state) - preserved across many timesteps
  h_t = short-term memory (hidden state) - working memory for current step
  Gates control information flow → prevents vanishing
```

---

## Memory Representation

### Two Types of Memory

#### 1. Cell State c(t) - Long-Term Memory

**Purpose:** Store information over long sequences (hundreds of timesteps)

**Characteristics:**
- **Persistent**: Information can flow unchanged for many steps
- **Protected**: Only modified through gating (not direct multiplication)
- **Additive updates**: Changes added/removed via forget and input gates
- **Highway**: Direct path for gradients during backpropagation

**Analogy:** Long-term memory in your brain (facts you remember for years)

**Example:** In disease progression modeling
- c(t) might store: "Patient has diabetes" (persists across visits)
- Even if not mentioned in recent visits, this fact remains in cell state

---

#### 2. Hidden State h(t) - Short-Term Memory

**Purpose:** Working memory for current computation and output

**Characteristics:**
- **Volatile**: Updated at every timestep
- **Output-focused**: Used for predictions and next layer input
- **Filtered**: Derived from cell state via output gate
- **Short-term**: Represents "what to focus on right now"

**Analogy:** Working memory in your brain (what you're thinking about now)

**Example:** In disease progression modeling
- h(t) might represent: "At current visit, blood pressure is elevated" (transient observation)

---

### Memory Flow Diagram

```
Time:        t=0          t=1          t=2          t=3
             
Cell State:  c_0 ────────> c_1 ────────> c_2 ────────> c_3
(Long-term)  [preserve]   [update]     [update]     [update]
             Patient      + New info   + New info   + Forget
             history                                 old info
                          
             │            │            │            │
             ↓            ↓            ↓            ↓
             
Hidden State: h_0 ────────> h_1 ────────> h_2 ────────> h_3
(Short-term)  Current      Current      Current      Current
              focus        focus        focus        focus
              
              ↓            ↓            ↓            ↓
              
Predictions:  ŷ_0          ŷ_1          ŷ_2          ŷ_3
```

**Key Insight:** 
- **c_t** flows horizontally with minimal transformation (information highway)
- **h_t** is computed from c_t at each step (filtered view)

---

### Mathematical Formulation

At timestep t, LSTM computes:

```
Inputs:
  x_t: Current input [batch, input_size]
  h_{t-1}: Previous hidden state [batch, hidden_size]
  c_{t-1}: Previous cell state [batch, hidden_size]

Gates (all have same shape: [batch, hidden_size]):
  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  # Forget gate
  i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  # Input gate
  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  # Output gate
  g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)  # Candidate values

Cell state update:
  c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t

Hidden state update:
  h_t = o_t ⊙ tanh(c_t)

Outputs:
  h_t: Hidden state [batch, hidden_size]
  c_t: Cell state [batch, hidden_size]

Notation:
  σ = sigmoid function (outputs 0-1)
  ⊙ = element-wise multiplication
  [h_{t-1}, x_t] = concatenation
```

---

### Gate Functions Explained

#### Forget Gate (f_t)

**Purpose:** Decide what information to **discard** from cell state

**Range:** 0 to 1 (sigmoid activation)
- f_t = 1: Keep all information from c_{t-1}
- f_t = 0: Forget all information from c_{t-1}
- f_t = 0.5: Keep half the information

**Example:**
```python
# Patient visits: [Diabetes dx, Normal BP, Normal BP, High BP]
# At t=3 (High BP visit):
#   Forget gate might set f_t=0.1 for "Normal BP" memory
#   → Mostly forget the "Normal BP" pattern, make room for "High BP"
```

---

#### Input Gate (i_t)

**Purpose:** Decide what **new information** to add to cell state

**Range:** 0 to 1 (sigmoid activation)
- i_t = 1: Add all new candidate values
- i_t = 0: Add nothing new
- i_t = 0.5: Add half of new candidates

**Works with:** Candidate values g_t (what new info to potentially add)

**Example:**
```python
# New visit has CKD diagnosis (important event)
# Input gate might set i_t=0.9 → strongly add CKD info to cell state
```

---

#### Output Gate (o_t)

**Purpose:** Decide what information to **output** from cell state to hidden state

**Range:** 0 to 1 (sigmoid activation)
- o_t = 1: Output all cell state information
- o_t = 0: Output nothing
- o_t = 0.5: Output half of cell state

**Example:**
```python
# Cell state contains [diabetes, hypertension, CKD, medications]
# For predicting current stage, output gate might:
#   o_t=0.9 for disease info (diabetes, CKD) → highly relevant
#   o_t=0.2 for medication info → less relevant for stage classification
```

---

#### Candidate Values (g_t)

**Purpose:** Compute **potential new information** to add to cell state

**Range:** -1 to 1 (tanh activation)
- Positive values: Add information
- Negative values: Subtract information

**Modulated by:** Input gate i_t (how much to actually use)

**Example:**
```python
# New visit indicates kidney function improvement
# g_t might be positive for "kidney improvement"
# i_t controls how much of this to add to cell state
```

---

### Cell State Update: The Core Operation

```python
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
      └──────┬──────┘   └────┬────┘
      Keep from past   Add new info
```

**Step-by-step:**

1. **Forget old information:** `f_t ⊙ c_{t-1}`
   - Element-wise multiply cell state by forget gate
   - f_t ≈ 0 → forget, f_t ≈ 1 → keep

2. **Add new information:** `i_t ⊙ g_t`
   - Element-wise multiply candidates by input gate
   - i_t ≈ 0 → don't add, i_t ≈ 1 → add fully

3. **Combine:** `forget_result + add_result`
   - **Additive** update (not multiplicative)
   - This is key to preventing vanishing gradients!

**Numeric Example:**

```python
# Suppose hidden_size = 3 (simplified)

c_{t-1} = [0.8, 0.3, -0.5]  # Previous cell state

# Forget gate (keep most info)
f_t = [0.9, 0.2, 0.7]

# After forgetting
f_t ⊙ c_{t-1} = [0.72, 0.06, -0.35]

# Candidate values
g_t = [0.1, 0.8, -0.2]

# Input gate (add new info selectively)
i_t = [0.1, 0.9, 0.3]

# New information
i_t ⊙ g_t = [0.01, 0.72, -0.06]

# Updated cell state
c_t = [0.72, 0.06, -0.35] + [0.01, 0.72, -0.06]
    = [0.73, 0.78, -0.41]

# Dimension 0: Mostly preserved from past (0.8 → 0.73)
# Dimension 1: Mostly new information (0.3 → 0.78)
# Dimension 2: Mix of old and new (-0.5 → -0.41)
```

---

### Hidden State: Output Computation

```python
h_t = o_t ⊙ tanh(c_t)
      └─┬─┘   └───┬───┘
      Filter  Normalize
```

**Step-by-step:**

1. **Normalize cell state:** `tanh(c_t)`
   - Squash cell state to [-1, 1] range
   - Prevents unbounded growth

2. **Filter what to output:** `o_t ⊙ tanh(c_t)`
   - Output gate controls what information is relevant now
   - o_t ≈ 0 → hide this info, o_t ≈ 1 → expose this info

**Example:**

```python
c_t = [0.73, 0.78, -0.41]  # Cell state

tanh(c_t) = [0.62, 0.65, -0.39]  # Normalized

o_t = [0.9, 0.5, 0.1]  # Output gate

h_t = [0.56, 0.33, -0.04]  # Hidden state

# Dimension 0: Highly exposed (0.9 gate)
# Dimension 1: Partially exposed (0.5 gate)  
# Dimension 2: Mostly hidden (0.1 gate)
```

---

### Summary: c(t) vs h(t)

| Aspect | Cell State c(t) | Hidden State h(t) |
|--------|----------------|-------------------|
| **Purpose** | Long-term memory storage | Short-term working memory |
| **Update** | Additive (f⊙c + i⊙g) | Filtered from c_t (o⊙tanh(c)) |
| **Information** | Accumulated over many steps | Current relevant info |
| **Gradient flow** | Direct highway (prevents vanishing) | Gated (can vanish) |
| **Range** | Unbounded (can grow large) | [-1, 1] via tanh |
| **Usage** | Internal memory | Output and next input |
| **Analogy** | Hard drive | RAM |

---

## Recursive Processing

### LSTM Processes One Element at a Time

Unlike feedforward networks that process entire sequences in parallel, LSTM is **recursive** (recurrent):

```python
# Feedforward (parallel)
output = feedforward(entire_sequence)  # All at once

# LSTM (sequential)
for t in range(sequence_length):
    h_t, c_t = lstm(x_t, h_{t-1}, c_{t-1})  # One timestep at a time
```

### Why Recursive?

1. **Recurrent connections:** h_t depends on h_{t-1}
2. **Order matters:** Processing visit 3 before visit 2 would be wrong
3. **Temporal dependencies:** Each step builds on previous state

---

### Unrolling Through Time

**Conceptual View:**

```
Input sequence: x_0, x_1, x_2, x_3

Unrolled LSTM:
┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
│      │   │      │   │      │   │      │
│ LSTM │───│ LSTM │───│ LSTM │───│ LSTM │
│  t=0 │   │  t=1 │   │  t=2 │   │  t=3 │
│      │   │      │   │      │   │      │
└──┬───┘   └──┬───┘   └──┬───┘   └──┬───┘
   │          │          │          │
   x_0        x_1        x_2        x_3

State flow:
h_0, c_0 ──> h_1, c_1 ──> h_2, c_2 ──> h_3, c_3
```

**Important:** Same LSTM cell, same weights, applied repeatedly

---

### Processing Algorithm

**Pseudocode:**

```python
def lstm_forward(inputs, initial_h, initial_c, weights):
    """
    Process sequence one timestep at a time.
    
    Args:
        inputs: [batch, seq_len, input_size]
        initial_h: [batch, hidden_size]
        initial_c: [batch, hidden_size]
        weights: LSTM parameters (W_f, W_i, W_o, W_g, biases)
    
    Returns:
        all_hiddens: [batch, seq_len, hidden_size]
        final_h: [batch, hidden_size]
        final_c: [batch, hidden_size]
    """
    batch_size, seq_len, input_size = inputs.shape
    
    # Initialize
    h_t = initial_h
    c_t = initial_c
    all_hiddens = []
    
    # Process each timestep sequentially
    for t in range(seq_len):
        x_t = inputs[:, t, :]  # [batch, input_size]
        
        # Concatenate h_{t-1} and x_t
        combined = torch.cat([h_t, x_t], dim=1)  # [batch, hidden + input]
        
        # Compute gates
        f_t = torch.sigmoid(combined @ W_f + b_f)  # Forget gate
        i_t = torch.sigmoid(combined @ W_i + b_i)  # Input gate
        o_t = torch.sigmoid(combined @ W_o + b_o)  # Output gate
        g_t = torch.tanh(combined @ W_g + b_g)     # Candidates
        
        # Update cell state (long-term memory)
        c_t = f_t * c_t + i_t * g_t
        
        # Update hidden state (short-term memory)
        h_t = o_t * torch.tanh(c_t)
        
        # Store hidden state for this timestep
        all_hiddens.append(h_t)
    
    # Stack all hidden states
    all_hiddens = torch.stack(all_hiddens, dim=1)  # [batch, seq_len, hidden]
    
    return all_hiddens, h_t, c_t
```

**Key Points:**

1. **Loop over sequence:** Can't parallelize across time
2. **State carries forward:** h_t and c_t flow from step to step
3. **Weights shared:** Same W_f, W_i, W_o, W_g for all timesteps
4. **Output at each step:** h_t available for every position

---

### Example: Medical Visit Sequence

```python
# Patient with 3 visits
visits = [
    visit_0,  # Baseline
    visit_1,  # 3 months later
    visit_2   # 6 months later
]

# Initial state (zeros)
h_0 = torch.zeros(1, 512)
c_0 = torch.zeros(1, 512)

# Timestep 0: Process visit_0
h_1, c_1 = lstm(visit_0, h_0, c_0)
# h_1 encodes: "Patient history up to visit 0"
# c_1 stores: Long-term patient information

# Timestep 1: Process visit_1
h_2, c_2 = lstm(visit_1, h_1, c_1)
# h_2 encodes: "Patient history up to visit 1"
# c_2 stores: Updated long-term information (e.g., disease progressed)

# Timestep 2: Process visit_2
h_3, c_3 = lstm(visit_2, h_2, c_2)
# h_3 encodes: "Current patient state"
# c_3 stores: All accumulated information

# Predictions
current_stage = classifier(h_3)  # Predict stage at current visit
```

**Dependency chain:** h_3 depends on h_2, which depends on h_1, which depends on h_0

**This is why it's recursive:** Can't compute h_3 without first computing h_2 and h_1

---

## Vanishing Gradient Solution

### The Problem: Vanishing Gradients in RNNs

**Vanilla RNN Update:**
```python
h_t = tanh(W_h · h_{t-1} + W_x · x_t)
```

**Gradient Flow During Backpropagation:**
```
∂L/∂h_0 = ∂L/∂h_T · ∂h_T/∂h_{T-1} · ∂h_{T-1}/∂h_{T-2} · ... · ∂h_1/∂h_0
          └────────────────────────────────────────────┬─────────────────┘
                          Chain of derivatives
```

**Problem:**
- Each `∂h_t/∂h_{t-1}` involves W_h and tanh derivative
- `tanh'(x) ≤ 1`, often much smaller
- Multiplying many small numbers → gradient vanishes
- `0.5 × 0.5 × 0.5 × ... (100 times) ≈ 0`

**Result:** Can't learn long-term dependencies (information from h_0 doesn't reach h_100)

---

### LSTM Solution: Additive Cell State Updates

**Key Difference in Cell State:**
```python
# RNN (multiplicative)
h_t = tanh(W · h_{t-1} + ...)
# Gradient: ∂h_t/∂h_{t-1} = W · tanh'(...)  ← vanishing

# LSTM (additive)
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
# Gradient: ∂c_t/∂c_{t-1} = f_t  ← preserved!
```

---

### Why Additive Updates Prevent Vanishing

**Gradient of Cell State:**

```
∂c_t/∂c_{t-1} = f_t  (forget gate value)
```

**Gradient backpropagation through time:**

```
∂L/∂c_0 = ∂L/∂c_T · ∂c_T/∂c_{T-1} · ∂c_{T-1}/∂c_{T-2} · ... · ∂c_1/∂c_0
        = ∂L/∂c_T · f_T · f_{T-1} · f_{T-2} · ... · f_1
```

**Crucial insight:**
- If forget gates f_t ≈ 1 (keep memory), gradient flows unimpeded
- Unlike RNN where gradients always decay, LSTM can preserve gradients
- Forget gates are **learned** to be close to 1 when long-term memory needed

**Numerical Example:**

```python
# RNN: 100 timesteps
gradient_rnn = (0.5)^100 ≈ 7.9 × 10^-31  # Vanished!

# LSTM: 100 timesteps with f_t ≈ 0.99
gradient_lstm = (0.99)^100 ≈ 0.366  # Still significant!

# LSTM with f_t = 1.0 (perfect memory)
gradient_lstm = (1.0)^100 = 1.0  # No decay at all!
```

---

### Mathematical Derivation

**Cell State Gradient (Simplified):**

```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t

∂c_t/∂c_{t-1} = f_t

If f_t = 1 (keep all memory):
  ∂c_t/∂c_{t-1} = 1
  → Gradient flows unchanged (no vanishing)

If f_t = 0 (forget all memory):
  ∂c_t/∂c_{t-1} = 0
  → Gradient cut off (intentional - information not needed)
```

**Comparison:**

| Component | Gradient Flow | Vanishing Risk |
|-----------|---------------|----------------|
| **RNN:** h_t = tanh(W·h_{t-1}) | `∂h_t/∂h_{t-1} = W·tanh'(...)` | **High** (product of weights and tanh') |
| **LSTM:** c_t = f_t⊙c_{t-1} + ... | `∂c_t/∂c_{t-1} = f_t` | **Low** (controlled by learned gate) |

---

### Gating as Adaptive Skip Connections

**Conceptual View:**

```
c_{t-1} ──────┬─────────> c_t
              │           ↑
              │           │
            f_t (gate)    │
                          │
                      i_t ⊙ g_t (new info)

If f_t ≈ 1: c_{t-1} flows directly to c_t (highway connection)
```

**This is similar to ResNet skip connections:**

```
ResNet:      y = x + F(x)           (additive skip)
LSTM:        c_t = f_t⊙c_{t-1} + i_t⊙g_t  (gated additive)
```

**Benefit:** Gradients can skip through time via cell state highway

---

### When LSTM Still Struggles

LSTM is not a silver bullet. Gradients can still vanish if:

1. **Forget gates too low:** If f_t << 1 for many steps, gradients decay
2. **Very long sequences:** Even with f_t=0.99, after 1000 steps: 0.99^1000 ≈ 0.00004
3. **Hidden state gradients:** h_t gradients can still vanish (only c_t protected)

**Solutions for extremely long sequences:**
- Truncated Backpropagation Through Time (TBPTT)
- Transformer architecture (attention mechanism)
- Hierarchical models (like Visit-Grouped architecture)

---

### Empirical Evidence

**Benchmark Results:**

| Model | Sequence Length | Accuracy | Notes |
|-------|----------------|----------|-------|
| Vanilla RNN | 50 steps | 45% | Gradients vanish |
| Vanilla RNN | 500 steps | Random (~20%) | Complete gradient collapse |
| **LSTM** | 50 steps | **92%** | Learns long-term |
| **LSTM** | 500 steps | **85%** | Still effective |

---

## Data Shape Transformations

### Shape Journey Through LSTM

Let's trace shapes through a complete LSTM forward pass.

#### Setup

```python
batch_size = 32       # Number of patients
seq_len = 10          # Number of visits per patient
input_size = 258      # Visit embedding + time features
hidden_size = 512     # LSTM hidden dimension
num_layers = 2        # Stack 2 LSTM layers
```

---

### Single-Layer LSTM

#### Input

```python
x: [batch_size, seq_len, input_size]
   [32, 10, 258]

h_0: [1, batch_size, hidden_size]  # Initial hidden state
     [1, 32, 512]

c_0: [1, batch_size, hidden_size]  # Initial cell state
     [1, 32, 512]
```

**Note:** First dimension is 1 for single-layer LSTM

---

#### At Each Timestep (t)

```python
# Extract input for this timestep
x_t: [batch_size, input_size]
     [32, 258]

# Previous states
h_{t-1}: [batch_size, hidden_size]
         [32, 512]

c_{t-1}: [batch_size, hidden_size]
         [32, 512]

# Concatenate for gate computation
combined: [batch_size, hidden_size + input_size]
          [32, 512 + 258] = [32, 770]

# Compute gates (each has same shape)
f_t: [32, 512]  # Forget gate
i_t: [32, 512]  # Input gate
o_t: [32, 512]  # Output gate
g_t: [32, 512]  # Candidate values

# Update cell state
c_t: [32, 512]  # c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t

# Update hidden state
h_t: [32, 512]  # h_t = o_t ⊙ tanh(c_t)
```

---

#### LSTM Outputs

```python
# After processing all timesteps (t=0 to t=9)

output: [batch_size, seq_len, hidden_size]
        [32, 10, 512]
        # output[:, t, :] = h_t for each timestep

h_n: [1, batch_size, hidden_size]
     [1, 32, 512]
     # Final hidden state (h_10)

c_n: [1, batch_size, hidden_size]
     [1, 32, 512]
     # Final cell state (c_10)
```

**Summary:**
- **output**: All hidden states [h_0, h_1, ..., h_9]
- **h_n**: Only the last hidden state h_9
- **c_n**: Only the last cell state c_9

---

### Multi-Layer LSTM (Stacked)

When `num_layers=2`, LSTM stacks two layers:

```
Layer 2:  h²_0 ──> h²_1 ──> h²_2 ──> ... ──> h²_9
          ↑        ↑        ↑                 ↑
Layer 1:  h¹_0 ──> h¹_1 ──> h¹_2 ──> ... ──> h¹_9
          ↑        ↑        ↑                 ↑
Input:    x_0      x_1      x_2              x_9
```

**Layer 1** processes input sequence, **Layer 2** processes Layer 1's hidden states.

---

#### Input Shapes

```python
x: [batch_size, seq_len, input_size]
   [32, 10, 258]

h_0: [num_layers, batch_size, hidden_size]
     [2, 32, 512]
     # h_0[0]: Layer 1 initial hidden
     # h_0[1]: Layer 2 initial hidden

c_0: [num_layers, batch_size, hidden_size]
     [2, 32, 512]
     # c_0[0]: Layer 1 initial cell
     # c_0[1]: Layer 2 initial cell
```

**Key difference:** First dimension is `num_layers` (not 1)

---

#### Processing Flow

**Layer 1:**
```python
# Input to Layer 1
x: [32, 10, 258]

# Layer 1 processing
for t in range(10):
    h¹_t, c¹_t = LSTM_layer1(x_t, h¹_{t-1}, c¹_{t-1})

# Layer 1 output
output_layer1: [32, 10, 512]  # All hidden states from Layer 1
h¹_n: [32, 512]  # Final hidden of Layer 1
c¹_n: [32, 512]  # Final cell of Layer 1
```

**Layer 2:**
```python
# Input to Layer 2 = Output of Layer 1
input_layer2: [32, 10, 512]

# Layer 2 processing
for t in range(10):
    h²_t, c²_t = LSTM_layer2(h¹_t, h²_{t-1}, c²_{t-1})

# Layer 2 output
output_layer2: [32, 10, 512]  # All hidden states from Layer 2
h²_n: [32, 512]  # Final hidden of Layer 2
c²_n: [32, 512]  # Final cell of Layer 2
```

---

#### Final Output Shapes

```python
# LSTM returns output from the LAST layer
output: [batch_size, seq_len, hidden_size]
        [32, 10, 512]
        # Hidden states from Layer 2

h_n: [num_layers, batch_size, hidden_size]
     [2, 32, 512]
     # h_n[0]: Final hidden of Layer 1
     # h_n[1]: Final hidden of Layer 2

c_n: [num_layers, batch_size, hidden_size]
     [2, 32, 512]
     # c_n[0]: Final cell of Layer 1
     # c_n[1]: Final cell of Layer 2
```

---

### Shape Transformation Table

| Component | Single Layer | Multi-Layer (L=2) | Notes |
|-----------|--------------|-------------------|-------|
| **Input x** | [B, T, I] | [B, T, I] | Same |
| **h_0 shape** | [1, B, H] | [L, B, H] | L = num_layers |
| **c_0 shape** | [1, B, H] | [L, B, H] | L = num_layers |
| **output shape** | [B, T, H] | [B, T, H] | Same (last layer's hiddens) |
| **h_n shape** | [1, B, H] | [L, B, H] | All layers' final hiddens |
| **c_n shape** | [1, B, H] | [L, B, H] | All layers' final cells |

**Legend:**
- B = batch_size
- T = seq_len
- I = input_size
- H = hidden_size
- L = num_layers

---

### Bidirectional LSTM

When `bidirectional=True`, LSTM processes sequence in **both directions**:

```
Forward:      h→_0 ──> h→_1 ──> h→_2 ──> ... ──> h→_9
Backward:     h←_9 <── h←_8 <── h←_7 <── ... <── h←_0
```

---

#### Shape Changes

```python
# Input (same)
x: [32, 10, 258]

# Outputs (doubled hidden dimension)
output: [batch_size, seq_len, 2 * hidden_size]
        [32, 10, 1024]
        # output[:, t, :512] = forward hidden at t
        # output[:, t, 512:] = backward hidden at t

h_n: [num_layers * 2, batch_size, hidden_size]
     [2, 32, 512]  # if num_layers=1
     [4, 32, 512]  # if num_layers=2
     # First L layers: forward
     # Last L layers: backward

c_n: [num_layers * 2, batch_size, hidden_size]
     [2, 32, 512]  # if num_layers=1
```

**Key change:** Hidden dimension **doubles** in output (concat forward + backward)

---

## PyTorch LSTM Usage

### Basic Usage

```python
import torch
import torch.nn as nn

# Create LSTM
lstm = nn.LSTM(
    input_size=258,
    hidden_size=512,
    num_layers=2,
    batch_first=True,
    dropout=0.1,
    bidirectional=False
)

# Input
x = torch.randn(32, 10, 258)  # [batch, seq, input]

# Forward pass (automatic initialization of h_0, c_0)
output, (h_n, c_n) = lstm(x)

print(output.shape)  # [32, 10, 512]
print(h_n.shape)     # [2, 32, 512]
print(c_n.shape)     # [2, 32, 512]
```

---

### Providing Initial States

```python
# Create initial states
h_0 = torch.zeros(2, 32, 512)  # [num_layers, batch, hidden]
c_0 = torch.zeros(2, 32, 512)

# Forward with initial states
output, (h_n, c_n) = lstm(x, (h_0, c_0))
```

---

### Extracting Hidden States

```python
# Get all timesteps
all_hiddens = output  # [batch, seq, hidden]

# Get specific timestep
hidden_at_t5 = output[:, 5, :]  # [batch, hidden]

# Get last timestep (latest visit)
last_hidden = output[:, -1, :]  # [batch, hidden]
# Equivalent to h_n[-1] for single-direction LSTM

# Get first timestep
first_hidden = output[:, 0, :]  # [batch, hidden]
```

---

### Does LSTM Maintain Intermediate States?

**Question:** Does LSTM internally store h_0, h_1, ..., h_T and c_0, c_1, ..., c_T?

**Answer:** 
- **During forward pass:** No, only current h_t and c_t are kept in memory
- **During backward pass:** Yes, states are recomputed or retrieved for gradient computation
- **In output:** Only h_n and c_n (final states) are returned, unless you explicitly collect them

**Efficient Memory:**
```python
# LSTM doesn't store all intermediate states automatically
# It computes them on-the-fly during forward pass
# Only the output (all hiddens) is returned

output, (h_n, c_n) = lstm(x)

# output contains all h_t values [h_0, h_1, ..., h_9]
# h_n contains only final h_9 (for each layer)
# c_n contains only final c_9 (for each layer)

# If you need intermediate cell states, collect manually:
all_hiddens = []
all_cells = []
h_t, c_t = h_0, c_0

for t in range(seq_len):
    h_t, c_t = lstm(x[:, t:t+1, :], (h_t, c_t))
    all_hiddens.append(h_t)
    all_cells.append(c_t)
```

---

### Stacking Multiple Layers

```python
# Option 1: PyTorch nn.LSTM with num_layers
lstm = nn.LSTM(input_size=258, hidden_size=512, num_layers=3)
# Automatically stacks 3 LSTM layers

# Option 2: Manual stacking (more control)
class StackedLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=258, hidden_size=512, num_layers=1)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1)
        self.lstm3 = nn.LSTM(input_size=512, hidden_size=256, num_layers=1)
    
    def forward(self, x):
        # Layer 1
        out1, (h1, c1) = self.lstm1(x)
        
        # Layer 2 (input = output of layer 1)
        out2, (h2, c2) = self.lstm2(out1)
        
        # Layer 3 (different hidden size)
        out3, (h3, c3) = self.lstm3(out2)
        
        return out3, (h3, c3)
```

**When to use manual stacking:**
- Different hidden sizes per layer
- Layer-specific operations (e.g., attention after each layer)
- Different dropouts or activations per layer

---

### Dropout in LSTM

```python
# Dropout between LSTM layers
lstm = nn.LSTM(
    input_size=258,
    hidden_size=512,
    num_layers=3,
    dropout=0.2  # 20% dropout between layers
)

# Note: dropout only applied if num_layers > 1
# No dropout on last layer's output
```

**How dropout works:**
```
Layer 1 output ──> Dropout(0.2) ──> Layer 2 input
Layer 2 output ──> Dropout(0.2) ──> Layer 3 input
Layer 3 output ──> (no dropout)
```

**Manual dropout control:**
```python
class LSTMWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=258, hidden_size=512, num_layers=2)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        output = self.dropout(output)  # Apply dropout to ALL outputs
        return output, (h_n, c_n)
```

---

### Packing Padded Sequences

For variable-length sequences, use packing for efficiency:

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Sequences with different lengths
sequences = [
    torch.randn(8, 258),   # Patient 1: 8 visits
    torch.randn(12, 258),  # Patient 2: 12 visits
    torch.randn(5, 258),   # Patient 3: 5 visits
]

# Pad to same length
padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
# Shape: [3, 12, 258] (max_len=12)

# Lengths of each sequence
lengths = torch.tensor([8, 12, 5])

# Pack sequences (removes padding from computation)
packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)

# LSTM on packed sequence
output_packed, (h_n, c_n) = lstm(packed)

# Unpack output
output, lengths = pad_packed_sequence(output_packed, batch_first=True)
# Shape: [3, 12, 512] (back to padded format)
```

**Benefits:**
- Faster computation (skip padding)
- Correct handling of variable lengths
- No wasted computation on padding

---

## Practical Examples

### Example 1: Disease Progression Prediction

```python
import torch
import torch.nn as nn

class ProgressionLSTM(nn.Module):
    def __init__(self, input_size=258, hidden_size=512, num_stages=5):
        super().__init__()
        
        # 2-layer LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_size, num_stages)
    
    def forward(self, x):
        """
        Args:
            x: [batch, num_visits, input_size]
        
        Returns:
            stage_logits: [batch, num_visits, num_stages]
        """
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: [batch, num_visits, hidden_size]
        
        # Predict stage at each visit
        stage_logits = self.classifier(lstm_out)
        # stage_logits: [batch, num_visits, num_stages]
        
        return stage_logits

# Usage
model = ProgressionLSTM()
visits = torch.randn(32, 10, 258)  # 32 patients, 10 visits each

# Forward pass
predictions = model(visits)
print(predictions.shape)  # [32, 10, 5]

# Get current stage prediction (last visit)
current_stage = predictions[:, -1, :].argmax(dim=-1)
print(current_stage)  # [batch] - predicted stage for each patient
```

---

### Example 2: Sequence-to-One (Latest Visit Only)

```python
class LatestStageLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(258, 512, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(512, 5)
    
    def forward(self, x):
        """Predict only at the last visit."""
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use only last timestep
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden_size]
        
        # Predict
        logits = self.classifier(last_hidden)  # [batch, num_stages]
        
        return logits

# Usage
model = LatestStageLSTM()
visits = torch.randn(32, 10, 258)

predictions = model(visits)
print(predictions.shape)  # [32, 5] - only final visit prediction
```

---

### Example 3: Accessing Intermediate Cell States

```python
class LSTMWithCellStates(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(258, 512)
    
    def forward(self, x):
        """
        Manually iterate using LSTMCell to access all cell states.
        
        Args:
            x: [batch, seq_len, input_size]
        
        Returns:
            all_hiddens: [batch, seq_len, hidden_size]
            all_cells: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize
        h_t = torch.zeros(batch_size, 512).to(x.device)
        c_t = torch.zeros(batch_size, 512).to(x.device)
        
        all_hiddens = []
        all_cells = []
        
        # Process each timestep
        for t in range(seq_len):
            h_t, c_t = self.lstm_cell(x[:, t, :], (h_t, c_t))
            all_hiddens.append(h_t)
            all_cells.append(c_t)
        
        # Stack
        all_hiddens = torch.stack(all_hiddens, dim=1)  # [batch, seq, hidden]
        all_cells = torch.stack(all_cells, dim=1)      # [batch, seq, hidden]
        
        return all_hiddens, all_cells

# Usage
model = LSTMWithCellStates()
visits = torch.randn(32, 10, 258)

hiddens, cells = model(visits)
print(hiddens.shape)  # [32, 10, 512]
print(cells.shape)    # [32, 10, 512]

# Analyze cell state evolution
cell_visit_5 = cells[:, 5, :]  # Cell state at visit 5
print(cell_visit_5.shape)  # [32, 512]
```

---

### Example 4: Multi-Task with Multiple Heads

```python
class MultiTaskLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(258, 512, num_layers=2, batch_first=True, dropout=0.1)
        
        # Multiple prediction heads
        self.stage_classifier = nn.Linear(512, 5)
        self.time_predictor = nn.Linear(512, 1)
        self.risk_classifier = nn.Linear(512, 2)  # High/low risk
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Shared representations
        # Predict at each visit
        stage_logits = self.stage_classifier(lstm_out)  # [batch, visits, 5]
        time_pred = torch.relu(self.time_predictor(lstm_out))  # [batch, visits, 1]
        risk_logits = self.risk_classifier(lstm_out)  # [batch, visits, 2]
        
        return {
            'stage': stage_logits,
            'time_to_progression': time_pred,
            'risk': risk_logits
        }

# Training
model = MultiTaskLSTM()
optimizer = torch.optim.Adam(model.parameters())

for batch in dataloader:
    visits, true_stages, true_times, true_risks = batch
    
    # Forward
    predictions = model(visits)
    
    # Multi-task loss
    loss_stage = nn.CrossEntropyLoss()(predictions['stage'].view(-1, 5), 
                                       true_stages.view(-1))
    loss_time = nn.MSELoss()(predictions['time_to_progression'].squeeze(), 
                             true_times)
    loss_risk = nn.CrossEntropyLoss()(predictions['risk'].view(-1, 2), 
                                       true_risks.view(-1))
    
    total_loss = loss_stage + 0.5 * loss_time + 0.3 * loss_risk
    
    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

---

## Summary

### Key Concepts

1. **Two Memory Types**
   - **c(t)**: Long-term memory (cell state), preserved across many steps
   - **h(t)**: Short-term memory (hidden state), current working memory

2. **Recursive Processing**
   - Processes sequences one element at a time
   - Each step depends on previous state
   - Can't parallelize across time (unlike Transformers)

3. **Vanishing Gradient Solution**
   - Additive cell state updates: `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`
   - Gradient flows through forget gates (not weight matrices)
   - When f_t ≈ 1, gradients preserved

4. **Shape Transformations**
   - Input: `[batch, seq, input]`
   - Output: `[batch, seq, hidden]` (all hiddens)
   - h_n: `[num_layers, batch, hidden]` (final hiddens per layer)
   - c_n: `[num_layers, batch, hidden]` (final cells per layer)

5. **PyTorch Usage**
   - `nn.LSTM`: High-level API (recommended)
   - `nn.LSTMCell`: Low-level for manual iteration
   - Automatic state initialization (can override)
   - Supports stacking, bidirectionality, dropout

### Best Practices

1. **Use pre-trained embeddings** for input (e.g., CEHR-BERT)
2. **Stack 2-3 layers** for hierarchical representations
3. **Apply dropout** (0.1-0.3) between layers
4. **Use packing** for variable-length sequences
5. **Extract last hidden** for sequence classification
6. **Multi-task learning** improves shared representations

### Common Pitfalls

1. **Forgetting batch_first=True** (default is False in PyTorch)
2. **Not handling variable lengths** (use packing or masking)
3. **Exploding gradients** (use gradient clipping: `torch.nn.utils.clip_grad_norm_`)
4. **Overfitting on small data** (increase dropout, reduce hidden_size)
5. **Bidirectional for causal prediction** (use unidirectional for forecasting)

LSTM remains a powerful architecture for sequential data, especially when combined with pre-trained embeddings and hierarchical designs like the Visit-Grouped model.
