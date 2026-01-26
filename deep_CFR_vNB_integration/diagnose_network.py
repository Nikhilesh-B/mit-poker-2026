"""
Diagnostic script to see what the network is outputting.
"""
import sys
import os
import torch
import torch.nn.functional as F

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

from network.model import DeepCFRModule
from utils.infoset_parser import parse_infoset_to_network_input

# Load model
model_path = "output/models/mini_test.pt"
if len(sys.argv) > 1:
    model_path = sys.argv[1]

print(f"Loading model from: {model_path}")
model_data = torch.load(model_path, map_location='cpu', weights_only=False)

network_dim = model_data.get('network_dim', 256)
# Network args: nhandcards=3, nboardcards=5, n_action_history=20, nresponses=9
# (actions_layer1 has shape [256, 120] = 256 x (20*6))
network = DeepCFRModule(nhandcards=3, nboardcards=5, n_action_history=20, nresponses=9, dim=network_dim)
network.load_state_dict(model_data['strategy_network_state_dict'])
network.eval()

print(f"Network loaded. Hidden dim: {network_dim}")
print(f"Training iterations: {model_data.get('iterations', 'unknown')}")
print()

# Test with some sample infosets using actual format:
# "S{street}|H:{hand_str}|B:{board_str}|A:{history_str}"
# Hand format uses canonical cards like "14s0" (Ace spades)
test_infosets = [
    # Preflop (street=0), mediocre hand, no actions yet
    "S0|H:6s0,8s0,7s0|B:|A:",
    # Preflop, good hand, no actions
    "S0|H:14s0,13s0,12s0|B:|A:",
    # Preflop, facing a raise (action R)
    "S0|H:2s0,3s0,4s0|B:|A:R",
    # After call action
    "S0|H:14s0,13s0,12s0|B:|A:C",
    # With some board cards (street 1)
    "S1|H:14s0,13s0,12s0|B:10s0,9s0,8s0|A:CC",
]

print("=" * 60)
print("Network outputs (raw logits) and softmax probabilities")
print("=" * 60)
print()
print("Action order: DISCARD_0, DISCARD_1, DISCARD_2, CHECK, CALL, FOLD, RAISE_S, RAISE_M, RAISE_L")
print()

for infoset in test_infosets:
    print(f"Infoset: {infoset}")
    
    try:
        cc, ah = parse_infoset_to_network_input(infoset)
        
        with torch.no_grad():
            output = network(cc, ah)
        
        logits = output[0]  # Remove batch dim
        probs = F.softmax(logits, dim=0)
        
        print(f"  Raw logits:     [{', '.join([f'{x:.2f}' for x in logits.tolist()])}]")
        print(f"  Softmax probs:  [{', '.join([f'{x:.3f}' for x in probs.tolist()])}]")
        
        # Highlight dominant action
        max_idx = torch.argmax(probs).item()
        action_names = ['DISCARD_0', 'DISCARD_1', 'DISCARD_2', 'CHECK', 'CALL', 'FOLD', 'RAISE_S', 'RAISE_M', 'RAISE_L']
        print(f"  Dominant action: {action_names[max_idx]} (prob={probs[max_idx]:.3f})")
        print()
        
    except Exception as e:
        print(f"  Error: {e}")
        print()

# Also check the value networks
print("=" * 60)
print("Checking value networks (V0 and V1)")
print("=" * 60)

for player in [0, 1]:
    key = f'network_p{player}_state_dict'
    if key in model_data:
        v_network = DeepCFRModule(nhandcards=3, nboardcards=5, n_action_history=20, nresponses=9, dim=network_dim)
        v_network.load_state_dict(model_data[key])
        v_network.eval()
        
        # Test with first infoset
        infoset = test_infosets[0]
        cc, ah = parse_infoset_to_network_input(infoset)
        
        with torch.no_grad():
            output = v_network(cc, ah)
        
        logits = output[0]
        print(f"V{player} logits for '{infoset}':")
        print(f"  [{', '.join([f'{x:.2f}' for x in logits.tolist()])}]")
        print()
