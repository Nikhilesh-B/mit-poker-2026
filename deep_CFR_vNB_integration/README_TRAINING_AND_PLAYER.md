# Training and Using Deep CFR Model

## Quick Start

### 1. Train a Model

```bash
# Basic training (500 iterations, default settings)
python train_model.py

# Custom training
python train_model.py --iterations 1000 --network-dim 512 --output my_model.pt

# Quiet mode (less output)
python train_model.py --iterations 500 --quiet
```

### 2. Use the Model in Player

```bash
# Default: looks for 'deep_cfr_model.pt' in current directory
python player.py <port>

# Specify custom model path
DEEP_CFR_MODEL_PATH=my_model.pt python player.py <port>
```

## Training Script (`train_model.py`)

### Usage

```bash
python train_model.py [OPTIONS]
```

### Options

- `--iterations N` - Number of Deep CFR iterations (default: 500)
- `--network-dim N` - Network hidden dimension (default: 256)
- `--learning-rate FLOAT` - Learning rate (default: 0.001)
- `--use-network-after N` - Start using network after N iterations (default: 100)
- `--train-every N` - Train network every N iterations (default: 10)
- `--train-epochs N` - Epochs per training session (default: 5)
- `--output PATH` - Output model path (default: deep_cfr_model.pt)
- `--quiet` - Suppress progress output

### Example Training Runs

**Quick test (50 iterations):**
```bash
python train_model.py --iterations 50 --output test_model.pt
```

**Production training (2000 iterations):**
```bash
python train_model.py --iterations 2000 --network-dim 256 --output deep_cfr_production.pt
```

**High-capacity model:**
```bash
python train_model.py --iterations 1000 --network-dim 512 --learning-rate 0.0005 --output deep_cfr_large.pt
```

### Training Output

The script will:
1. Train the Deep CFR model
2. Print training statistics
3. Save the model to the specified path

Example output:
```
======================================================================
TRAINING DEEP CFR MODEL
======================================================================

Configuration:
  Iterations: 500
  Network dim: 256
  Learning rate: 0.001
  Use network after: 100 iterations
  Train every: 10 iterations
  Train epochs: 5
  Output: deep_cfr_model.pt

Starting training...
Iteration 10 | Loss: 147M | Samples: 38
Iteration 20 | Loss: 56M | Samples: 79
...

======================================================================
TRAINING COMPLETE
======================================================================

Final Statistics:
  Total iterations: 500
  Network usages: 400
  Training sessions: 50
  Initial loss: 120,480,173
  Final loss: 2,345,123
  Loss improvement: 98.1%
  Training samples: 1,234

Saving model to deep_cfr_model.pt...
✓ Model saved successfully!
  File size: 6.23 MB
```

## Player Script (`player.py`)

### Usage

```bash
python player.py <port> [--host HOST]
```

### Model Loading

The player automatically looks for `deep_cfr_model.pt` in the same directory. To use a different model:

```bash
# Via environment variable
DEEP_CFR_MODEL_PATH=my_model.pt python player.py 1337

# Or modify player.py to set default path
```

### What the Player Does

1. **Loads the trained model** on initialization
2. **Uses network predictions** to select actions
3. **Falls back to random** if model fails to load
4. **Implements Bot interface** for game engine

### Example Output

```
Loading Deep CFR model from deep_cfr_model.pt...
✓ Deep CFR Model Loaded!
  Network dim: 256
  Training loss: 2,345,123
  Training samples: 1,234
  Ready to play with Nash equilibrium strategy!
```

## Model File Format

The saved model (`.pt` file) contains:

```python
{
    'network_state_dict': {...},  # PyTorch model weights
    'network_dim': 256,            # Network configuration
    'iterations': 500,              # Training iterations
    'final_loss': 2345123.0,       # Final training loss
    'training_samples': 1234,      # Number of training samples
    'stats': {...}                 # Full training statistics
}
```

## Training Recommendations

### For Quick Testing
- **Iterations**: 50-100
- **Network dim**: 128
- **Time**: ~1-2 minutes

### For Development
- **Iterations**: 200-500
- **Network dim**: 256
- **Time**: ~5-10 minutes

### For Production
- **Iterations**: 1000-5000
- **Network dim**: 256-512
- **Time**: ~30 minutes - 2 hours

### Hyperparameter Guidelines

**Network Dimension:**
- 128: Fast, less capacity (good for testing)
- 256: Balanced (recommended default)
- 512: High capacity, slower (for strong bots)

**Learning Rate:**
- 0.001: Default, works well
- 0.0005: Slower but more stable
- 0.002: Faster but may be unstable

**Training Schedule:**
- `train_every=10`: Good balance
- `train_every=5`: More frequent updates
- `train_every=20`: Less frequent, faster iterations

## Troubleshooting

### Model Not Found
```
✗ Model file not found: deep_cfr_model.pt
Train a model first with: python train_model.py
```

**Solution**: Train a model first!

### Model Load Error
```
✗ Failed to load model: ...
```

**Solution**: 
- Check file exists and is not corrupted
- Ensure PyTorch version matches training version
- Try retraining the model

### Player Falls Back to Random
If the player prints "Falling back to random play", check:
1. Model file exists
2. Model file is not corrupted
3. All dependencies are installed
4. Check error messages in output

## Integration with Game Engine

The player follows the standard Bot interface:

```python
class Player(Bot):
    def handle_new_round(self, game_state, round_state, active):
        # Called at start of each round
        pass
    
    def handle_round_over(self, game_state, terminal_state, active):
        # Called at end of each round
        pass
    
    def get_action(self, game_state, round_state, active):
        # Called when action is needed
        # Returns: FoldAction, CallAction, CheckAction, RaiseAction, or DiscardAction
        return action
```

## Next Steps

1. **Train a model**: `python train_model.py --iterations 500`
2. **Test the player**: Run against the game engine
3. **Evaluate performance**: Compare against baseline bots
4. **Iterate**: Train longer, tune hyperparameters, improve architecture

## Files

- `train_model.py` - Training script
- `player.py` - Player bot using trained model
- `deep_cfr_model.pt` - Trained model (created by training)
