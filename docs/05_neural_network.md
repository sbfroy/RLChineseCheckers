# 5. The Neural Network

## What the network does

The network takes a board state as input and produces two outputs:

1. **Policy:** "How good is each possible move?" — a probability distribution over all 1210 actions
2. **Value:** "How good is this position for me?" — a single number from -1 (losing) to +1 (winning)

This is called a **policy-value network** because it predicts both what to do and how well we're doing, from a single forward pass.

## Architecture

### The big picture

```
Input: spatial (10×17×17) + scalars (10)
         │
         v
  ┌──────────────┐
  │  Input Conv   │  10 channels → 128 channels, 3×3 convolution
  │  + BatchNorm  │
  │  + ReLU       │
  └──────┬───────┘
         │
         v
  ┌──────────────┐
  │  Res Block 1  │  Each block: conv→BN→ReLU→conv→BN + skip connection → ReLU
  │  Res Block 2  │
  │  Res Block 3  │  (6 blocks by default)
  │  Res Block 4  │
  │  Res Block 5  │
  │  Res Block 6  │
  └──────┬───────┘
         │
    ┌────┴────┐
    │         │
    v         v
┌────────┐ ┌────────┐
│ Policy │ │ Value  │
│  Head  │ │  Head  │
└───┬────┘ └───┬────┘
    │          │
    v          v
 1210 logits  1 scalar
 (actions)   (-1 to +1)
```

### Component by component

#### Input convolution
Takes the 10-channel spatial input and expands it to 128 channels using a 3×3 convolution. This is like "feature extraction" — each of the 128 channels will learn to detect a different spatial pattern.

#### Residual blocks (the trunk)
The core of the network. Each **residual block** has:
1. A 3×3 convolution (finds patterns)
2. Batch normalization (stabilizes training)
3. ReLU activation (adds nonlinearity)
4. Another 3×3 convolution + batch norm
5. A **skip connection** — the input is added back to the output

The skip connection is the key innovation from ResNets. Without it, deep networks struggle to train because gradients vanish through many layers. The skip connection lets gradients flow directly back, making deep networks trainable.

We use 6 blocks by default. More blocks = more capacity to learn complex patterns, but slower training and inference.

#### Policy head
Reduces the trunk output to a probability over actions:
1. 1×1 convolution → 32 channels (compression)
2. Flatten to vector
3. Concatenate scalar features (this is where global info enters)
4. Two fully-connected layers → 1210 output values (one per possible action)
5. Action masking (illegal actions set to -infinity)
6. Softmax → probabilities that sum to 1.0

#### Value head
Reduces the trunk output to a single position evaluation:
1. 1×1 convolution → 1 channel (heavy compression)
2. Flatten to vector
3. Concatenate scalar features
4. Two fully-connected layers → 1 output
5. Tanh activation → squashes to range [-1, +1]

### Why two heads sharing a trunk?

The policy ("what to do") and value ("how good is this") share a lot of the same understanding. Both need to recognize piece formations, distances to goal, mobility, etc. By sharing the trunk, they share this computational work and learn shared representations. The heads only diverge for their specific tasks.

This is also how AlphaZero works — a single network with two heads.

## How the network is used

### During training
```python
# Batch of states
policy_logits, values = model(spatial_batch, scalars_batch, legal_masks)
# policy_logits: (batch_size, 1210) — raw scores, masked
# values: (batch_size, 1) — position estimates

# Losses are computed against training targets:
# - Policy loss: how close are the logits to the MCTS policy targets?
# - Value loss: how close are the values to the actual game outcomes?
```

### During play (inference)
```python
# Single state
probs, value = model.predict(spatial, scalars, legal_mask)
# probs: (1210,) — probability distribution over legal actions
# value: float — position estimate

# The agent picks the action with highest probability (or samples)
```

## Model size

With default settings (`num_res_blocks=6, trunk_channels=128`):
- **~7.2 million parameters**
- Forward pass takes ~2-5ms on CPU
- This is moderate — small enough for CPU competition play, large enough to learn complex strategies

For comparison:
- AlphaZero (chess): 80 res blocks, 256 channels (~80M params) — much larger, but also a much harder game
- Our model is sized for practical CPU inference within competition time limits

## Key implementation details

### Batch normalization
Used after every convolution. Normalizes the activations within each batch, which:
- Makes training faster and more stable
- Reduces sensitivity to weight initialization
- Acts as slight regularization

During training it uses batch statistics; during inference (`model.eval()`) it uses accumulated running statistics.

### Gradient clipping
We clip gradient norms to 1.0 during training. This prevents exploding gradients from destabilizing learning, especially in early training when the network outputs are essentially random.

### The predict() method
Wraps the forward pass for single-state inference with `torch.no_grad()` (no gradient computation needed during play). This is faster and uses less memory.
