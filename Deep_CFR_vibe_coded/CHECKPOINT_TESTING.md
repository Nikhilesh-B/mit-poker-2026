# Testing Checkpoints During Training

## 🎯 Checkpoint Schedule (Every 2 Epochs)

Your training will save **12 checkpoints** over 3 hours:

| Checkpoint | Time | What to Expect |
|------------|------|----------------|
| **Epoch 2** | **~15 min** | **Very weak, learning basics** |
| **Epoch 4** | **~30 min** | **Starting to understand poker** |
| Epoch 6 | ~45 min | Learning hand strength |
| Epoch 8 | ~60 min | Basic strategy forming |
| **Epoch 10** | **~1.5 hrs** | **Good test point** ✓ |
| Epoch 12 | ~1.8 hrs | Improving |
| Epoch 14 | ~2.1 hrs | Getting better |
| **Epoch 16** | **~2.4 hrs** | **Solid strategy** ✓ |
| Epoch 18 | ~2.7 hrs | Refined play |
| Epoch 20 | ~2.8 hrs | Strong play |
| **Epoch 22** | **~2.9 hrs** | **Near final** ✓ |
| Epoch 24 | ~3 hrs | Almost done |
| **Epoch 25 (final)** | **~3 hrs** | **COMPLETE!** 🎉 |

## 🧪 How to Test Any Checkpoint

### Quick Test (See Strategy)
```bash
cd Deep_CFR
bash test_checkpoint.sh 10    # Test epoch 10
```

This will:
- Load the checkpoint
- Show the learned strategy on test hands
- Display action probabilities

### Full Game Test (Play Against Engine)
```bash
# 1. Copy checkpoint you want to test
cp ../checkpoints/deep_cfr_epoch10.pt ../checkpoints/deep_cfr_final.pt

# 2. Start engine (Terminal 1)
python ../engine.py

# 3. Run your bot (Terminal 2)
cd Deep_CFR
uv run python player.py
```

## 📊 Expected Performance by Checkpoint

| Epochs | vs Random | vs Henry | Strategy Quality |
|--------|-----------|----------|------------------|
| 2-4 | 50% (coin flip) | -4 BB/h | ❌ Terrible |
| 6-8 | 55% | -2 BB/h | 🟡 Weak |
| 10-12 | 60% | -1 BB/h | 🟢 Improving |
| 14-16 | 65% | ~0 BB/h | ✅ Decent |
| 18-20 | 68% | +0.5 BB/h | 💪 Good |
| 22-25 | 70% | +0.5-1 BB/h | 🔥 Strong |

## 🔍 Monitor Checkpoint Creation

**Watch for new checkpoints:**
```bash
watch -n 10 ls -lht ../checkpoints/
```

**Check training progress:**
```bash
python3 check_progress.py
```

## 💡 Recommended Testing Points

### Test #1: Epoch 10 (~1.5 hours)
**Why:** First meaningful checkpoint
```bash
bash test_checkpoint.sh 10
```

**Expected:** 
- Still loses to Henry but much better than epoch 3
- Shows some poker understanding
- Loss should be ~-1 to -2 BB/hand (vs -4.4 before)

### Test #2: Epoch 16 (~2.4 hours)  
**Why:** Solid strategy should be forming
```bash
bash test_checkpoint.sh 16
```

**Expected:**
- Competitive with Henry
- Reasonable betting patterns
- ~Break even or small win

### Test #3: Epoch 25 (Final)
**Why:** Best version!
```bash
bash test_checkpoint.sh 25
# or just use test_trained_model.py
```

**Expected:**
- Best performance in 3 hours
- Should beat or tie Henry
- Good exploitation of mistakes

## 🎮 Full Testing Example

```bash
# 1. Check what's available
ls -lht ../checkpoints/

# 2. Test epoch 10 strategy
cd Deep_CFR
bash test_checkpoint.sh 10

# 3. Play 100 hands vs engine
cp ../checkpoints/deep_cfr_epoch10.pt ../checkpoints/deep_cfr_final.pt
# Run engine in Terminal 1
# Run player.py in Terminal 2

# 4. Compare with epoch 16
bash test_checkpoint.sh 16
# Run another game

# 5. See improvement!
```

## 📈 Tracking Improvement

Create a simple log:
```bash
echo "Epoch,Result,Notes" > test_results.csv
echo "10,-123,Better than epoch 3!" >> test_results.csv
echo "16,-45,Much improved!" >> test_results.csv
echo "25,+78,Beat Henry!" >> test_results.csv
```

## ⚡ Quick Commands

**List all checkpoints:**
```bash
ls -lht ../checkpoints/ | grep epoch
```

**Test latest checkpoint:**
```bash
LATEST=$(ls -t ../checkpoints/deep_cfr_epoch*.pt | head -1)
echo "Testing: $LATEST"
cp "$LATEST" ../checkpoints/deep_cfr_final.pt
uv run python test_trained_model.py
```

**Compare two checkpoints:**
```bash
# Test epoch 10
bash test_checkpoint.sh 10 > results_epoch10.txt

# Test epoch 20
bash test_checkpoint.sh 20 > results_epoch20.txt

# Compare
diff results_epoch10.txt results_epoch20.txt
```

## 🎯 What You're Looking For

As epochs increase, you should see:

1. **Action Distribution Changes:**
   - Early: Random-looking (33% fold, 33% call, 33% raise)
   - Later: Smart (0% fold with good hands, strategic raises)

2. **Betting Patterns:**
   - Early: Min bets everywhere
   - Later: Varied bet sizes based on situation

3. **Hand Reading:**
   - Early: Ignores opponent actions
   - Later: Adjusts to opponent's betting

4. **Win Rate:**
   - Epoch 3: -4.4 BB/hand ❌
   - Epoch 10: -1 to -2 BB/hand 📈
   - Epoch 25: 0 to +1 BB/hand ✅

## 🔥 Pro Tip

**Test the same scenario at different epochs:**

```bash
# Save a specific game state for testing
# Then load epoch 2, 10, 20, 25 and see different decisions!
```

This shows how the strategy evolves!

---

**Current Status:** Training running with checkpoints every 2 epochs. First checkpoint in ~15 minutes! 🚀

