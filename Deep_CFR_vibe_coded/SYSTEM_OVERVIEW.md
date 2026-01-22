# Deep CFR System - Consolidated Overview

**Purpose**: Clear mental model of how everything fits together

---

## 🎯 The Big Picture (Read This First!)

Deep CFR learns poker by playing against itself millions of times, using a neural network to remember which actions led to regret.

**The Core Idea in 3 Steps:**
```
1. PLAY: Simulate poker games (self-play)
2. LEARN: Train neural network on which actions were good/bad
3. USE: Trained network plays poker for you
```

---

## 📁 File Organization: What's What?

### ⭐ **CORE FILES** (7 files - These are the actual system)

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: Game State                                          │
│  ↓                                                           │
│  deep_cfr_encoding.py     → Convert state to numbers        │
│  ↓                                                           │
│  regret_network.py        → Neural network predicts regrets │
│  ↓                                                           │
│  action_mapping.py        → Convert numbers to actions      │
│  ↓                                                           │
│  OUTPUT: Poker Action                                       │
└─────────────────────────────────────────────────────────────┘

TRAINING LOOP:
  cfr_trainer.py          → Self-play + regret computation
  train.py                → Main training script
  training_monitor.py     → Track progress

DEPLOYMENT:
  player.py               → Your competition bot
```

### 🧪 **TEST FILES** (5 files - Optional, for debugging)

```
test_encoding.py              → Test state encoding works
test_regret_network.py        → Test network works
test_player_integration.py    → Test bot loads model
test_trained_model.py         → Quick model test
check_progress.py             → Check training status
plot_training.py              → Generate plots
```

**Reality**: You rarely need these. They were useful during development.

### 🏗️ **SKELETON** (4 files - Game engine, don't touch)

```
skeleton/
  actions.py    → Action definitions (Fold, Call, Raise, etc.)
  states.py     → Game state (cards, pot, stacks, etc.)
  bot.py        → Bot base class
  runner.py     → Bot runner infrastructure
```

**Reality**: Pre-written by MIT, just use them.

---

## 🔄 The Complete Data Flow

### **During Training:**

```
START
  ↓
train.py
  ↓
FOR EACH EPOCH:
  ├─→ PHASE 1: DATA COLLECTION
  │     ↓
  │   cfr_trainer.traverse_episode()
  │     ├─→ Create random initial game
  │     ├─→ For each game state:
  │     │     ├─→ deep_cfr_encoding.encode_state()      [state → 32 numbers]
  │     │     ├─→ regret_network.forward()              [numbers → regrets]
  │     │     ├─→ regret_network.regret_matching()      [regrets → strategy]
  │     │     ├─→ action_mapping.sample_action()        [strategy → action idx]
  │     │     └─→ action_mapping.index_to_action()      [idx → actual action]
  │     ├─→ Recurse to terminal state (game end)
  │     ├─→ Compute regrets (counterfactual analysis)
  │     └─→ Store (state, regret) pairs in buffer
  │
  └─→ PHASE 2: NETWORK TRAINING
        ↓
      Train regret_network on collected data
        ↓
      Save checkpoint every N epochs
        ↓
DONE → checkpoints/deep_cfr_final.pt
```

### **During Competition (Inference):**

```
START GAME
  ↓
player.py loads trained model
  ↓
FOR EACH DECISION:
  ├─→ Get current game state
  ├─→ deep_cfr_encoding.encode_state()      [state → 32 numbers]
  ├─→ regret_network.forward()              [numbers → regrets]
  ├─→ action_mapping.get_legal_mask()       [which actions are legal?]
  ├─→ regret_network.regret_matching()      [regrets → strategy]
  ├─→ action_mapping.sample_action()        [strategy → action idx]
  └─→ action_mapping.index_to_action()      [idx → actual action]
  ↓
EXECUTE ACTION
```

---

## 🧠 Mental Model: 3 Layers

Think of the system as 3 layers:

### **Layer 1: REPRESENTATION** (State → Numbers)
```python
# deep_cfr_encoding.py
game_state → [0.75, 0.3, 0.1, 0.8, ...] (32 numbers)
             ↑
             Hand strength, cards, pot size, betting history, etc.
```

**Why**: Neural networks need numbers, not poker concepts.

### **Layer 2: DECISION** (Numbers → Strategy)
```python
# regret_network.py
[32 numbers] → Neural Network → [8 regrets]
                                  ↓
                            regret_matching()
                                  ↓
                            [8 probabilities]
             
Example: [0.0, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0]
         ↑
         Fold 0%, Call 40%, Raise-S 30%, Raise-M 20%, ...
```

**Why**: This is the learned policy. Network predicts regrets → convert to action probabilities.

### **Layer 3: EXECUTION** (Strategy → Action)
```python
# action_mapping.py
[8 probabilities] → sample_action() → 2 (index)
                                      ↓
                    index_to_action() → RaiseAction(50)
                                        ↓
                                   Game engine executes
```

**Why**: Convert abstract strategy to concrete poker action.

---

## 🎮 How To Use The System

### **1. Train a Model**

```bash
cd /Users/nikhileshbelulkar/Documents/mit-poker-2026

# Basic training (no monitoring)
uv run python Deep_CFR/train.py --epochs 25

# Better: WITH monitoring
uv run python Deep_CFR/train.py \
    --epochs 50 \
    --eval-interval 5 \
    --eval-games 100 \
    --device cpu
```

**What happens:**
- Creates checkpoints every 2 epochs
- Logs metrics to `Deep_CFR/logs/`
- Saves best model to `checkpoints/`

### **2. Test a Model**

```bash
# Quick inference test
uv run python Deep_CFR/test_trained_model.py \
    --checkpoint checkpoints/deep_cfr_epoch16.pt

# Full 1000-hand match vs Henry
python engine.py  # (uses Deep_CFR/player.py)
```

### **3. Deploy for Competition**

Your bot is already configured! Just make sure:
```python
# Deep_CFR/player.py (line ~42)
checkpoint_paths = [
    Path("checkpoints/deep_cfr_epoch16.pt"),  # ← Your best model
    ...
]
```

Then run:
```bash
python engine.py
```

---

## 📊 Key Parameters (What Actually Matters)

### **In Training** (`train.py`):

| Parameter | What It Does | Current | Recommended |
|-----------|--------------|---------|-------------|
| `--epochs` | Training iterations | 25 | 30-50 |
| `--traversals-per-epoch` | Games per epoch | 100 | 100-200 |
| `--train-steps-per-epoch` | NN updates per epoch | 100 | 100-200 |
| `--learning-rate` | How fast to learn | 0.001 | 0.0005-0.001 |
| `--eval-interval` | Test every N epochs | 5 | 2-5 |

**Don't touch** (unless you know why):
- `--buffer-size` (100k is fine)
- `--batch-size` (128 is fine)
- `--hidden-sizes` (256, 128 is fine)

### **In CFR Traversal** (`cfr_trainer.py`):

| Parameter | What It Does | Current |
|-----------|--------------|---------|
| `MAX_DEPTH` | Tree depth limit | 15 |
| Monte Carlo sampling | Actions sampled after depth 5 | 2 actions |

**Critical**: These prevent infinite recursion in huge poker game tree.

### **In State Encoding** (`deep_cfr_encoding.py`):

- 32 features total (hand strength, cards, pot, history)
- All normalized to [0, 1] range
- Fixed size (required for neural network)

---

## 🔧 What To Change If Performance Is Bad

### **Problem 1: Bot plays too tight (folds 80%+)**

**Diagnosis**: Policy collapse
**Solution**:
```python
# In train.py, add entropy bonus (prevents too-tight play)
# OR stop training earlier (before collapse)
# OR use the monitoring system to catch it early
```

### **Problem 2: Loss not decreasing**

**Diagnosis**: Learning rate too low or network too small
**Solution**:
```bash
python Deep_CFR/train.py --learning-rate 0.005  # 5x higher
# OR
python Deep_CFR/train.py --hidden-sizes 512 256  # Bigger network
```

### **Problem 3: Training too slow**

**Diagnosis**: Monte Carlo sampling not aggressive enough
**Solution**:
```python
# In cfr_trainer.py, line ~178:
if len(legal_action_indices) > 2 and depth > 3:  # Sample earlier (was depth > 5)
    sample_size = 2  # OR 1 for even faster
```

### **Problem 4: Performance varies wildly**

**Diagnosis**: Poker has high variance
**Solution**:
- Test with 10,000 hands instead of 1,000
- Run multiple tests and average results
- Use the monitoring system's evaluation feature

---

## 📈 Monitoring & Debugging

### **While Training:**

```bash
# Watch live logs
tail -f Deep_CFR/logs/training_metrics.csv

# Generate plots anytime
uv run python Deep_CFR/plot_training.py
```

### **Key Metrics to Watch:**

1. **Loss**: Should decrease steadily (0.5 → 0.2 over 25 epochs)
2. **Action distribution**: Should stabilize
   - Good: 30-40% fold, 30-40% call, 20-30% raise
   - Bad: 80%+ fold (too tight) or 80%+ call (too loose)
3. **Win rate vs baseline**: Should increase (if eval enabled)

---

## 💡 The Simplest Possible Mental Model

**If you only remember ONE thing:**

```
Deep CFR = Neural Network that predicts "how much regret 
           would I have if I don't do this action?"

High regret → Do this action more
Low regret  → Do this action less

Train by: Playing millions of hands against yourself
Use by:   Let the network choose your actions
```

**That's it!** Everything else is engineering to make this work.

---

## 🎯 Quick Reference Card

### I want to...

| Task | Command |
|------|---------|
| **Train from scratch** | `uv run python Deep_CFR/train.py --epochs 30 --eval-interval 5` |
| **Continue training** | `uv run python Deep_CFR/train.py --epochs 50` (loads latest checkpoint) |
| **Test a checkpoint** | Edit `player.py` line 42, then `python engine.py` |
| **See training progress** | `uv run python Deep_CFR/plot_training.py` |
| **Check what epoch trained to** | `ls -lt checkpoints/` |
| **Deploy for competition** | Already done! `player.py` loads best model |

---

## 🚀 Next Steps for Better Performance

### Option A: Retrain with Monitoring (6-8 hours)
```bash
uv run python Deep_CFR/train.py \
    --epochs 50 \
    --eval-interval 2 \
    --eval-games 500 \
    --learning-rate 0.0005
```

**Why**: 
- See exactly when bot improves
- Catch policy collapse early
- Find true best checkpoint (not lucky variance)

### Option B: Improve Features (2-3 hours)
Edit `deep_cfr_encoding.py`:
- Add more betting history features
- Add opponent modeling (track opponent's actions)
- Better pot odds calculations

### Option C: Add Strategy Network (4-5 hours)
Implement proper CFR theory:
- Second network for average strategy
- More stable than single checkpoint
- Theoretically guaranteed Nash convergence

---

## ✅ Summary: What You Need to Know

**The 7 Core Files:**
1. `deep_cfr_encoding.py` - State → Numbers
2. `regret_network.py` - Neural network
3. `action_mapping.py` - Numbers → Actions
4. `cfr_trainer.py` - Self-play algorithm
5. `train.py` - Main training loop
6. `training_monitor.py` - Progress tracking
7. `player.py` - Your competition bot

**The Flow:**
```
State → Encoding → Network → Strategy → Action
         ↑                                ↓
         └────── Training Loop ───────────┘
```

**How to Use:**
- Train: `uv run python Deep_CFR/train.py --epochs 30 --eval-interval 5`
- Test: `python engine.py`
- Monitor: `uv run python Deep_CFR/plot_training.py`

**That's all you need!** Everything else is optional testing/debugging files.

---

Want me to explain any specific part in more detail? Or shall we start a better training run now? 🚀
