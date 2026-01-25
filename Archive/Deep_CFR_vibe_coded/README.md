# Deep CFR Implementation for Poker

Complete implementation of Deep Counterfactual Regret Minimization for the MIT Pokerbots discard variant.

## Overview

This implementation uses neural networks to approximate the regret tables in traditional CFR, allowing it to scale to large game trees.

### Key Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Deep CFR Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Game State → Encoding → Regret Network → Strategy          │
│      ↓           ↓            ↓               ↓             │
│   RoundState   [32]         [8]          Sample Action      │
│                                                              │
│  Training Loop:                                              │
│    CFR Traversal → Collect Regrets → Train Network          │
│         ↓                  ↓                 ↓               │
│    Percolate values    Store samples    Update weights      │
└─────────────────────────────────────────────────────────────┘
```

## Files

### Core Components

1. **`deep_cfr_encoding.py`** - State Encoding
   - Converts game states to 32-dimensional feature vectors
   - Handles cards, stacks, pot odds, betting history
   - All features normalized to [0,1]

2. **`regret_network.py`** - Neural Network
   - Architecture: 32 → 256 → 128 → 8
   - Predicts regret values for each action
   - Includes regret matching function

3. **`action_mapping.py`** - Action Utilities
   - Maps network outputs (0-7) to game actions
   - Creates legal action masks
   - Handles discard/betting street differences

4. **`cfr_trainer.py`** - CFR Traversal
   - Recursive game tree traversal
   - Regret percolation from terminal states
   - Training data collection

5. **`train.py`** - Main Training Loop
   - Alternates data collection and network training
   - Checkpointing and logging
   - Command-line interface

### Testing

- `test_encoding.py` - Test state encoding
- `test_encoding_comprehensive.py` - Extended encoding tests
- `test_regret_network.py` - Test network and regret matching

## Installation

```bash
cd /Users/nikhileshbelulkar/Documents/mit-poker-2026

# Add PyTorch to project
uv add torch

# Sync dependencies
uv sync
```

## Usage

### Training

Basic training (100 epochs, default settings):
```bash
cd Deep_CFR
uv run python train.py
```

Advanced training with custom parameters:
```bash
uv run python train.py \
    --epochs 200 \
    --traversals-per-epoch 200 \
    --train-steps-per-epoch 150 \
    --learning-rate 0.001 \
    --hidden-sizes 256 128 \
    --save-interval 10 \
    --device cpu
```

### Testing

Test individual components:
```bash
# Test encoding
uv run python test_encoding_comprehensive.py

# Test regret network
uv run python test_regret_network.py

# Test CFR traversal
uv run python cfr_trainer.py
```

## How It Works

### 1. State Encoding (32 features)

```python
encode_state(game_state, round_state, player_id) → [32]

Features:
  [0]      Hand strength
  [1-6]    My cards (rank, suit pairs)
  [7-20]   Board cards (rank, suit pairs)  
  [21-22]  Game state (street, button)
  [23-27]  Stacks/pot
  [28]     Pot odds
  [29-31]  Betting history
```

### 2. Regret Prediction

```python
regrets = regret_net(state_encoding)  # [8]

Actions:
  0: Fold
  1: Check/Call
  2: Raise Small
  3: Raise Medium
  4: Raise Large
  5: Discard Card 0
  6: Discard Card 1
  7: Discard Card 2
```

### 3. Regret Matching

```python
# Convert regrets to strategy
strategy = regret_matching(regrets, legal_mask)

# Positive regrets → higher probability
# Negative regrets → zero probability
# Illegal actions → zero probability
```

### 4. CFR Traversal (The Magic!)

```python
def cfr_traverse(state, player):
    if terminal:
        return payoff  # BASE CASE
    
    # Get strategy from network
    strategy = regret_matching(regret_net(state), legal_mask)
    
    # RECURSE for each action
    for action in legal_actions:
        action_values[action] = cfr_traverse(next_state, player)
    
    # Compute node value
    node_value = Σ(strategy[a] * action_values[a])
    
    # Compute regrets
    regrets = action_values - node_value
    
    # Store training data
    buffer.append((state, regrets))
    
    # PERCOLATE value up
    return node_value
```

### 5. Training Loop

```
For each epoch:
  Phase 1: Data Collection
    - Run 100 CFR traversals
    - Collect (state, true_regrets) pairs
    
  Phase 2: Network Training
    - Sample batches from buffer
    - Train: predicted_regrets ← true_regrets
    - Update weights via SGD
```

## Training Progress

Expected training output:
```
======================================================================
Epoch 1/100
======================================================================

Phase 1: Collecting data (100 traversals)...
  Traversal 20/100 | Avg P0 value: +0.52
  Traversal 40/100 | Avg P0 value: +0.31
  ...

Phase 2: Training network (100 steps)...

Epoch 1 Summary:
  Avg P0 value: +0.42
  Avg P1 value: -0.42
  Training loss: 0.234567
  Buffer size: 5000
  Total traversals: 100
  Epoch time: 12.3s
  ✓ Saved checkpoint: checkpoints/deep_cfr_epoch1.pt
```

## Model Checkpoints

Saved checkpoints contain:
- Network weights
- Optimizer state
- Training statistics
- Hyperparameters

Load a checkpoint:
```python
from train import load_checkpoint

regret_net, checkpoint = load_checkpoint('checkpoints/deep_cfr_final.pt')

# Use for inference
state_encoding = encode_state(game_state, round_state, player_id)
regrets = regret_net(state_encoding)
strategy = regret_matching(regrets, legal_mask)
action = sample_action(strategy)
```

## Integration with Player

To use the trained network in actual games:

```python
# In player.py

from deep_cfr_encoding import encode_state
from regret_network import regret_matching, sample_action  
from action_mapping import get_legal_mask, action_index_to_engine_action
from train import load_checkpoint

class Player(Bot):
    def __init__(self):
        # Load trained model
        self.regret_net, _ = load_checkpoint('checkpoints/deep_cfr_final.pt')
    
    def get_action(self, game_state, round_state, active):
        # Encode state
        state = encode_state(game_state, round_state, active)
        
        # Predict regrets
        regrets = self.regret_net(state)
        
        # Get strategy
        legal_mask = get_legal_mask(round_state, active)
        strategy = regret_matching(regrets, legal_mask)
        
        # Sample action
        action_idx = sample_action(strategy)
        return action_index_to_engine_action(action_idx, round_state)
```

## Performance Tips

1. **Start small**: Train on 10-20 epochs first to verify everything works
2. **Monitor loss**: Should generally decrease over time
3. **Buffer size**: Larger = more diverse data, but slower
4. **Learning rate**: 1e-3 is a good default, reduce if loss oscillates
5. **GPU**: Use `--device cuda` if available for 5-10x speedup

## Troubleshooting

### Loss is nan or exploding
- Reduce learning rate (try 1e-4)
- Check gradients: `torch.nn.utils.clip_grad_norm_` (already included)

### Not learning (loss plateaus)
- Increase traversals per epoch
- Increase network size (e.g., `--hidden-sizes 512 256 128`)
- Check buffer isn't too small

### Too slow
- Reduce traversals per epoch
- Use GPU (`--device cuda`)
- Reduce network size

## Next Steps

After training:

1. **Evaluate**: Test against baseline bots
2. **Iterate**: Train longer or tune hyperparameters
3. **Deploy**: Integrate into player.py
4. **Compare**: Test against your NFSP implementation

## References

- Deep CFR paper: Brown et al. (2019)
- Original CFR: Zinkevich et al. (2007)
- Neural Fictitious Self-Play: Heinrich & Silver (2016)

## Architecture Diagram

```
                    ┌──────────────────┐
                    │   Game Engine    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   RoundState     │
                    │  (hands, board,  │
                    │   stacks, etc.)  │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  State Encoding  │
                    │      [32]        │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Regret Network   │
                    │  32→256→128→8    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Regret Values   │
                    │      [8]         │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Regret Matching  │
                    │ + Legal Mask     │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   Strategy [8]   │
                    │  (probabilities) │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Sample Action   │
                    │     (0-7)        │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Engine Action   │
                    │ (Fold/Call/etc.) │
                    └──────────────────┘
```

## License

MIT - For educational use in MIT Pokerbots competition

