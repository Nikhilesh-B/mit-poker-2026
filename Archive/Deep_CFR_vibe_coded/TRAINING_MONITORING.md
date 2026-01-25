# Training Monitoring Guide 📊

**Problem:** You were training blind - no idea if the bot was improving!

**Solution:** Comprehensive training diagnostics to see exactly what's happening.

---

## 🎯 What's New

### 1. **Training Loss Tracking**
- See if the regret network is learning
- Loss should decrease over epochs
- Logged to CSV after each epoch

### 2. **Online Evaluation**
- Bot plays vs random baseline every N epochs
- Track win rate improvement over time
- See actual poker performance (BB/hand)

### 3. **Action Distribution**
- Are we too tight? (folding too much)
- Are we too loose? (calling everything)
- Track fold/call/raise/discard percentages

### 4. **Visual Progress Plots**
- 4-panel visualization showing all metrics
- Auto-generated at end of training
- Can generate manually anytime

---

## 🚀 How to Use

### Training with Monitoring (Recommended)

```bash
# Train with evaluation every 5 epochs
python Deep_CFR/train.py \
    --epochs 50 \
    --eval-interval 5 \
    --eval-games 100 \
    --device cpu
```

**What this does:**
- ✅ Logs training loss every epoch
- ✅ Plays 100 hands vs random bot every 5 epochs
- ✅ Tracks action distributions
- ✅ Generates plots at end

### Check Progress During Training

```bash
# View current logs (live updating)
tail -f Deep_CFR/logs/training_metrics.csv

# Generate plots from current data
python Deep_CFR/plot_training.py

# Auto-refresh plots every 10 seconds
watch -n 10 python Deep_CFR/plot_training.py
```

---

## 📁 Output Files

All logs saved to `Deep_CFR/logs/`:

### `training_metrics.csv`
Logged every epoch:
```csv
epoch,loss,avg_p0_value,avg_p1_value,buffer_size,fold_pct,call_pct,raise_pct,discard_pct
1,0.523,-42.3,38.7,10000,45.2,30.1,20.5,4.2
2,0.401,-25.1,22.8,20000,40.5,32.3,22.8,4.4
3,0.298,-10.4,8.2,30000,38.2,35.1,23.5,3.2
...
```

### `evaluation_results.csv`
Logged every N epochs (if `--eval-interval` > 0):
```csv
epoch,num_games,wins,losses,avg_chips_won,win_rate,bb_per_hand
5,100,58,42,145.2,58.0,7.26
10,100,65,35,287.5,65.0,14.38
15,100,71,29,401.3,71.0,20.07
...
```

### `training_progress.png`
4-panel visualization:
1. **Training Loss** - Is the network learning?
2. **Expected Values** - P0 vs P1 value trends
3. **Action Distribution** - Fold/call/raise % over time
4. **Win Rate vs Baseline** - Performance improvement

---

## 📊 Interpreting Results

### 🔍 What to Look For

#### ✅ **Good Training**
- Loss **decreasing** over epochs
- Win rate **increasing** vs baseline
- Action distribution **stabilizing** (not 100% fold)
- BB/hand **positive and growing**

#### ❌ **Problems to Watch For**

**Problem 1: Loss not decreasing**
```
Epoch 1: Loss 0.523
Epoch 2: Loss 0.521
Epoch 3: Loss 0.519
Epoch 4: Loss 0.518  ← Too slow!
```
**Fix:** Increase learning rate (`--learning-rate 0.01`)

**Problem 2: Too tight (losing to blinds)**
```
Action dist: Fold 87%, Call 8%, Raise 5%
Result: -500 chips (paying blinds!)
```
**Fix:** Train longer, or increase exploration

**Problem 3: Too loose (calling everything)**
```
Action dist: Fold 10%, Call 80%, Raise 10%
Result: -800 chips (losing big pots!)
```
**Fix:** May need more training data, or CFR traversals

**Problem 4: Win rate plateauing**
```
Epoch 10: 62% win rate
Epoch 20: 63% win rate  ← Not improving
Epoch 30: 62% win rate
```
**Fix:** 
- Random baseline too weak, test vs better bots
- May need deeper network or more data

---

## 🎮 Example Training Session

```bash
# Start training with monitoring
python Deep_CFR/train.py --epochs 25 --eval-interval 5 --device cpu

# Output:
# ==============================================================================
# Epoch 1/25
# ==============================================================================
# Phase 1: Collecting data (100 traversals)...
# Phase 2: Training network (100 steps)...
# 
# Epoch 1 Summary:
#   Avg P0 value: -42.30
#   Avg P1 value: +38.70
#   Training loss: 0.5234
#   Buffer size: 10000
#   Total traversals: 100
#   Action dist: Fold 45.2%, Call 30.1%, Raise 20.5%, Discard 4.2%
#   Epoch time: 120.3s
#   Total time: 2.0m
# 
# → Running evaluation (100 hands vs Random Bot)...
# → Results: 52W-48L (52.0% win rate)
# → Avg: +85.4 chips/hand (+4.27 BB/hand)
# 
# ✓ Saved checkpoint: checkpoints/deep_cfr_epoch5.pt
# 
# ==============================================================================
# Epoch 25/25
# ==============================================================================
# ...
# 
# Generating training visualizations...
# ✓ Training plots saved to: Deep_CFR/logs/training_progress.png
# 
# Next steps:
#   1. Check training plots: Deep_CFR/logs/training_progress.png
#   2. Review metrics: Deep_CFR/logs/training_metrics.csv
#   3. Test the model using test_trained_model.py
```

---

## 💡 Tips

1. **Start with evaluation enabled**
   - Use `--eval-interval 5` to see improvement
   - Without evaluation, you're flying blind!

2. **Check plots frequently**
   - Run `python Deep_CFR/plot_training.py` anytime
   - Visual feedback is much clearer than numbers

3. **Compare checkpoints**
   - If Epoch 20 is worse than Epoch 15, use Epoch 15!
   - Training can temporarily worsen (exploration)

4. **Action distribution matters**
   - If 80%+ folding → too tight → bleeding blinds
   - If 80%+ calling → too loose → losing big pots
   - Aim for balanced mix (30-40% fold, 30-40% call, 20-30% raise)

5. **BB/hand is the key metric**
   - Positive BB/hand = profitable bot
   - Negative BB/hand = losing bot
   - +5 BB/hand vs random = good
   - +10 BB/hand vs random = very good

---

## 🔧 Advanced Options

```bash
# Detailed help
python Deep_CFR/train.py --help

# Key monitoring options:
--eval-interval N      # Evaluate every N epochs (default: 5, 0=disable)
--eval-games N         # Play N games for evaluation (default: 100)
--save-interval N      # Save checkpoint every N epochs (default: 10)
```

---

## 📈 Next Steps

1. **Run a monitored training session:**
   ```bash
   python Deep_CFR/train.py --epochs 50 --eval-interval 5
   ```

2. **Watch the magic happen:**
   - See loss decrease
   - See win rate increase
   - See action distribution stabilize

3. **Find the best checkpoint:**
   - Check `evaluation_results.csv`
   - Find epoch with highest win rate
   - Use that checkpoint in player.py

4. **Test against real opponents:**
   - Load best checkpoint
   - Play vs Henry, A, etc.
   - Iterate and improve!

---

## 🎉 You're No Longer Blind!

Before: "Is it learning? Who knows! 🤷"
After: "Loss down 60%, win rate up to 68%, let's go! 📈"

Happy training! 🚀
