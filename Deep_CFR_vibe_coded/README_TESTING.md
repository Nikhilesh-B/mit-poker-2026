# Testing Deep CFR Encoding

## Setup

First, you need to install PyTorch. Add it to your `pyproject.toml`:

```toml
[project]
dependencies = [
    "torch>=2.0.0",
    # ... other dependencies
]
```

Then run:
```bash
uv sync
```

## Running the Tests

```bash
cd /Users/nikhileshbelulkar/Documents/mit-poker-2026/Deep_CFR
uv run python test_encoding.py
```

## What the Test Does

The test script (`test_encoding.py`) creates mock poker game states and verifies:

1. **Preflop encoding** - Tests with 3-card hands and empty board
2. **Postflop encoding** - Tests after discards with board cards
3. **Edge cases** - Tests with minimal cards

## Expected Output

You should see:
- ✓ Shape matches (torch.Size([33]))
- ✓ All features properly normalized (values in 0-1 range)
- Feature breakdown showing all 33 features

## Features Encoded (33 total)

1. Hand strength: 1
2. My cards (3×2): 6  
3. Board cards (7×2): 14
4. Game state: 3
5. Stacks/pot: 5
6. Pot odds: 1
7. Betting history: 3

## Next Steps After Testing

Once encoding tests pass:

1. Build the `RegretNetwork` (neural network architecture)
2. Implement `regret_matching` function
3. Create CFR traversal loop
4. Build training loop

## Troubleshooting

If you get import errors, make sure:
- PyTorch is installed (`uv add torch`)
- pkrbot is installed (should be in project dependencies)
- You're running from the correct directory

