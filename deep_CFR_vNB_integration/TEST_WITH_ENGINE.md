# Testing Deep CFR Player with Engine

## Quick Setup

### 1. Make sure you have a trained model

```bash
# If you don't have one yet, train a quick model:
cd deep_CFR_vNB_integration
python train_model.py --iterations 50 --output deep_cfr_model.pt
```

### 2. Update config.py

Edit `/Users/nikhileshbelulkar/Documents/mit-poker-2026/config.py`:

```python
PLAYER_1_NAME = "SkeletonBot"
PLAYER_1_PATH = "./python_skeleton_OG"

PLAYER_2_NAME = "DeepCFR"
PLAYER_2_PATH = "./deep_CFR_vNB_integration"
```

Or test Deep CFR as Player 1:
```python
PLAYER_1_NAME = "DeepCFR"
PLAYER_1_PATH = "./deep_CFR_vNB_integration"

PLAYER_2_NAME = "SkeletonBot"
PLAYER_2_PATH = "./python_skeleton_OG"
```

### 3. Run the engine

```bash
cd /Users/nikhileshbelulkar/Documents/mit-poker-2026
python engine.py
```

The engine will:
- Start both bots
- Run 1000 rounds
- Show progress
- Save game logs to `gamelogs/` folder

### 4. Check the results

After the game completes, check:
- Final bankrolls in the output
- Game logs in `gamelogs/` folder
- Plots showing bankroll over time

## What to Expect

- **First few rounds**: Deep CFR might play randomly if model isn't loaded
- **Model loading**: You should see "✓ Deep CFR Model Loaded!" in the output
- **Gameplay**: The bot will use network predictions to select actions
- **Results**: Even with a lightly trained model (50 iterations), it should make reasonable decisions

## Troubleshooting

### Model not found
If you see "✗ Model file not found", make sure:
- `deep_cfr_model.pt` exists in `deep_CFR_vNB_integration/` folder
- Or train one with `python train_model.py`

### Import errors
If there are import errors, make sure:
- All dependencies are installed
- The `skeleton/` folder exists in `deep_CFR_vNB_integration/`
- Parent directory is in Python path (should be automatic)

### Player falls back to random
If the player falls back to random:
- Check the error messages in the engine output
- Verify the model file is not corrupted
- Try retraining the model

## Testing Against Different Bots

You can test against any bot by updating `config.py`:

```python
# Test against MCCFR
PLAYER_2_NAME = "MCCFR"
PLAYER_2_PATH = "./MCCFR"

# Test against another bot
PLAYER_2_NAME = "OtherBot"
PLAYER_2_PATH = "./other_bot_folder"
```

## Quick Test (Fewer Rounds)

To test faster, temporarily change in `config.py`:
```python
NUM_ROUNDS = 10  # Instead of 1000
```

This will run just 10 rounds for quick testing.
