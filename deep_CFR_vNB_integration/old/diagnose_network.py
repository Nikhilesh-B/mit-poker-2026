"""
Comprehensive diagnostic script to analyze network outputs across all possible infoset types.
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

ACTION_NAMES = ['DISCARD_0', 'DISCARD_1', 'DISCARD_2', 'CHECK', 'CALL', 'FOLD', 'RAISE_S', 'RAISE_M', 'RAISE_L']

# Comprehensive test cases organized by category
# Format: "S{street}|H:{hand_str}|B:{board_str}|A:{history_str}"
# Cards: {rank}s{suit} where rank=2-14 (14=Ace), suit=0-3
# Actions: C=Check/Call, R=Raise, F=Fold, D=Discard

TEST_CASES = {
    # ========================================
    # STREET 0 (PREFLOP) - Discard decisions
    # ========================================
    "Street 0 - Initial (no actions)": [
        ("Weak hand (2-3-4)", "S0|H:2s0,3s0,4s0|B:|A:"),
        ("Medium hand (7-8-9)", "S0|H:7s0,8s0,9s0|B:|A:"),
        ("Strong hand (Q-K-A)", "S0|H:12s0,13s0,14s0|B:|A:"),
        ("Pair hand (7-7-K)", "S0|H:7s0,7s1,13s0|B:|A:"),
        ("Suited connectors (8-9-10 same suit)", "S0|H:8s0,9s0,10s0|B:|A:"),
        ("Random weak (2-7-J offsuit)", "S0|H:2s0,7s1,11s2|B:|A:"),
    ],
    
    "Street 0 - After Check": [
        ("Weak after check", "S0|H:2s0,3s0,4s0|B:|A:C"),
        ("Strong after check", "S0|H:12s0,13s0,14s0|B:|A:C"),
    ],
    
    "Street 0 - Facing Raise": [
        ("Weak facing raise", "S0|H:2s0,3s0,4s0|B:|A:R"),
        ("Medium facing raise", "S0|H:7s0,8s0,9s0|B:|A:R"),
        ("Strong facing raise", "S0|H:12s0,13s0,14s0|B:|A:R"),
        ("Pair facing raise", "S0|H:10s0,10s1,14s0|B:|A:R"),
    ],
    
    "Street 0 - Check-Raise Sequence": [
        ("Weak in check-raise", "S0|H:2s0,3s0,4s0|B:|A:CR"),
        ("Strong in check-raise", "S0|H:12s0,13s0,14s0|B:|A:CR"),
    ],
    
    "Street 0 - After Discard": [
        ("After discard, weak", "S0|H:2s0,3s0,4s0|B:|A:D"),
        ("After discard, strong", "S0|H:12s0,13s0,14s0|B:|A:D"),
        ("After opp discard", "S0|H:7s0,8s0,9s0|B:|A:CD"),
    ],
    
    # ========================================
    # STREET 1 (FLOP) - Board interaction
    # ========================================
    "Street 1 - Hit the board": [
        ("Top pair (K on K-7-2)", "S1|H:13s0,8s0,4s0|B:13s1,7s0,2s0|A:CC"),
        ("Two pair (K-7 on K-7-2)", "S1|H:13s0,7s1,4s0|B:13s1,7s0,2s0|A:CC"),
        ("Set (777 on 7-J-2)", "S1|H:7s0,7s1,7s2|B:7s3,11s0,2s0|A:CC"),
        ("Overpair (AA on T-7-2)", "S1|H:14s0,14s1,5s0|B:10s0,7s0,2s0|A:CC"),
    ],
    
    "Street 1 - Missed the board": [
        ("Total air (2-3-4 on K-Q-J)", "S1|H:2s0,3s0,4s0|B:13s0,12s0,11s0|A:CC"),
        ("Low cards on high board", "S1|H:5s0,6s0,7s0|B:14s0,13s0,12s0|A:CC"),
    ],
    
    "Street 1 - Drawing hands": [
        ("Flush draw (2 hearts + heart board)", "S1|H:8s0,9s0,4s0|B:10s0,7s0,2s0|A:CC"),
        ("Straight draw (7-8 on 9-T-2)", "S1|H:7s0,8s0,3s0|B:9s0,10s0,2s0|A:CC"),
        ("Open-ended (6-7 on 8-9-K)", "S1|H:6s0,7s0,2s0|B:8s0,9s0,13s0|A:CC"),
    ],
    
    "Street 1 - Facing aggression": [
        ("Top pair facing raise", "S1|H:13s0,8s0,4s0|B:13s1,7s0,2s0|A:CCR"),
        ("Air facing raise", "S1|H:2s0,3s0,4s0|B:13s0,12s0,11s0|A:CCR"),
        ("Draw facing raise", "S1|H:6s0,7s0,2s0|B:8s0,9s0,13s0|A:CCR"),
    ],
    
    "Street 1 - As aggressor": [
        ("Strong hand, we raised", "S1|H:13s0,13s1,5s0|B:13s2,7s0,2s0|A:CR"),
        ("Bluffing with air", "S1|H:2s0,3s0,4s0|B:13s0,12s0,11s0|A:CR"),
    ],
    
    # ========================================
    # STREET 2 (TURN) - Deeper streets
    # ========================================
    "Street 2 - Made hands": [
        ("Top pair, turn brick", "S2|H:13s0,8s0,4s0|B:13s1,7s0,2s0,5s0|A:CCCC"),
        ("Two pair on turn", "S2|H:13s0,7s1,4s0|B:13s1,7s0,2s0,9s0|A:CCCC"),
        ("Set on turn", "S2|H:7s0,7s1,3s0|B:7s2,11s0,2s0,5s0|A:CCCC"),
    ],
    
    "Street 2 - Draws improving": [
        ("Made straight on turn", "S2|H:6s0,7s0,2s0|B:8s0,9s0,13s0,10s0|A:CCCC"),
        ("Still drawing on turn", "S2|H:6s0,7s0,2s0|B:8s0,9s0,13s0,4s0|A:CCCC"),
    ],
    
    "Street 2 - Big pots (raised flop)": [
        ("Strong after flop raise", "S2|H:13s0,13s1,5s0|B:13s2,7s0,2s0,4s0|A:CRCC"),
        ("Weak after flop raise", "S2|H:2s0,3s0,5s0|B:13s0,12s0,11s0,4s0|A:CRCC"),
    ],
    
    # ========================================
    # STREET 3 (RIVER) - Final decisions
    # ========================================
    "Street 3 - Value hands": [
        ("Top pair river", "S3|H:13s0,8s0,4s0|B:13s1,7s0,2s0,5s0,3s0|A:CCCCCC"),
        ("Two pair river", "S3|H:13s0,7s1,4s0|B:13s1,7s0,2s0,9s0,3s0|A:CCCCCC"),
        ("Set river", "S3|H:7s0,7s1,3s0|B:7s2,11s0,2s0,5s0,4s0|A:CCCCCC"),
        ("Straight river", "S3|H:6s0,7s0,2s0|B:8s0,9s0,13s0,10s0,4s0|A:CCCCCC"),
    ],
    
    "Street 3 - Bluff catchers": [
        ("Medium pair facing bet", "S3|H:9s0,9s1,3s0|B:13s0,12s0,7s0,5s0,2s0|A:CCCCCCR"),
        ("Weak pair facing bet", "S3|H:7s0,7s1,3s0|B:14s0,13s0,12s0,5s0,2s0|A:CCCCCCR"),
    ],
    
    "Street 3 - Bluffing spots": [
        ("Total air on river", "S3|H:2s0,3s0,4s0|B:14s0,13s0,10s0,8s0,6s0|A:CCCCCC"),
        ("Missed draw on river", "S3|H:6s0,7s0,2s0|B:8s0,9s0,13s0,4s0,3s0|A:CCCCCC"),
    ],
    
    "Street 3 - All-in situations (long action history)": [
        ("River after multiple raises", "S3|H:14s0,14s1,5s0|B:14s2,7s0,2s0,4s0,3s0|A:CRRCRC"),
        ("River facing 3-bet pot", "S3|H:13s0,13s1,5s0|B:13s2,7s0,2s0,4s0,3s0|A:RRRCC"),
    ],
    
    # ========================================
    # EDGE CASES
    # ========================================
    "Edge Cases": [
        ("Empty action history street 0", "S0|H:7s0,8s0,9s0|B:|A:"),
        ("Single action street 0", "S0|H:7s0,8s0,9s0|B:|A:C"),
        ("Long action history", "S3|H:14s0,13s0,12s0|B:11s0,10s0,9s0,8s0,7s0|A:CRCRCRCC"),
        ("All same suit hand", "S0|H:2s0,5s0,9s0|B:|A:"),
        ("All different suits", "S0|H:2s0,5s1,9s2|B:|A:"),
        ("Low pocket pair preflop", "S0|H:2s0,2s1,5s0|B:|A:"),
        ("High pocket pair preflop", "S0|H:14s0,14s1,5s0|B:|A:"),
    ],
}


def analyze_network(network, infosets, verbose=True):
    """Analyze network outputs for a list of infosets."""
    results = []
    
    for description, infoset in infosets:
        try:
            cc, ah = parse_infoset_to_network_input(infoset)
            
            with torch.no_grad():
                output = network(cc, ah)
            
            logits = output[0]  # Remove batch dim
            probs = F.softmax(logits, dim=0)
            
            max_idx = torch.argmax(probs).item()
            max_prob = probs[max_idx].item()
            
            results.append({
                'description': description,
                'infoset': infoset,
                'logits': logits.tolist(),
                'probs': probs.tolist(),
                'dominant_action': ACTION_NAMES[max_idx],
                'dominant_prob': max_prob,
                'error': None
            })
            
        except Exception as e:
            results.append({
                'description': description,
                'infoset': infoset,
                'error': str(e)
            })
    
    return results


def print_results(category, results, compact=False):
    """Print analysis results for a category."""
    print(f"\n{'='*70}")
    print(f"  {category}")
    print(f"{'='*70}")
    
    if compact:
        # Compact format: one line per infoset
        print(f"{'Description':<35} {'Dominant':<12} {'Prob':<8} {'Other high probs'}")
        print("-" * 70)
        
        for r in results:
            if r.get('error'):
                print(f"{r['description']:<35} ERROR: {r['error']}")
                continue
            
            # Find other significant actions (prob > 0.10)
            other_actions = []
            for i, p in enumerate(r['probs']):
                if p > 0.10 and i != r['probs'].index(max(r['probs'])):
                    other_actions.append(f"{ACTION_NAMES[i]}:{p:.2f}")
            
            other_str = ", ".join(other_actions[:3]) if other_actions else "-"
            
            print(f"{r['description']:<35} {r['dominant_action']:<12} {r['dominant_prob']:.3f}    {other_str}")
    else:
        # Verbose format
        for r in results:
            print(f"\n  {r['description']}")
            print(f"  Infoset: {r['infoset']}")
            
            if r.get('error'):
                print(f"  ERROR: {r['error']}")
                continue
            
            # Format logits and probs
            logits_str = ", ".join([f"{x:+.2f}" for x in r['logits']])
            probs_str = ", ".join([f"{x:.3f}" for x in r['probs']])
            
            print(f"  Logits: [{logits_str}]")
            print(f"  Probs:  [{probs_str}]")
            print(f"  >> Dominant: {r['dominant_action']} (prob={r['dominant_prob']:.3f})")


def print_summary_stats(all_results):
    """Print summary statistics across all test cases."""
    print(f"\n{'='*70}")
    print("  SUMMARY STATISTICS")
    print(f"{'='*70}")
    
    action_counts = {name: 0 for name in ACTION_NAMES}
    total = 0
    
    for results in all_results.values():
        for r in results:
            if not r.get('error'):
                action_counts[r['dominant_action']] += 1
                total += 1
    
    print(f"\nDominant action distribution across {total} test cases:")
    print("-" * 40)
    
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {action:<12} {count:>3} ({pct:>5.1f}%) {bar}")
    
    # Check for potential issues
    print("\n" + "-" * 40)
    print("Potential issues to watch:")
    
    # Check if one action dominates too much
    max_action = max(action_counts.items(), key=lambda x: x[1])
    if max_action[1] > 0.5 * total:
        print(f"  WARNING: {max_action[0]} dominates {100*max_action[1]/total:.0f}% of cases")
    else:
        print(f"  OK: No single action dominates (max is {max_action[0]} at {100*max_action[1]/total:.0f}%)")
    
    # Check if discards are being used
    discard_total = sum(action_counts.get(f'DISCARD_{i}', 0) for i in range(3))
    print(f"  Discards used: {discard_total} times ({100*discard_total/total:.1f}% of cases)")
    
    # Check if raises are being used
    raise_total = sum(action_counts.get(f'RAISE_{s}', 0) for s in ['S', 'M', 'L'])
    print(f"  Raises used: {raise_total} times ({100*raise_total/total:.1f}% of cases)")


def main():
    # Load model
    model_path = "output/models/mini_test.pt"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    # Parse optional flags
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    compact = "--compact" in sys.argv or "-c" in sys.argv
    
    print(f"Loading model from: {model_path}")
    
    try:
        model_data = torch.load(model_path, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print(f"ERROR: Model file not found: {model_path}")
        print("\nUsage: python diagnose_network.py <model_path> [--compact|-c] [--verbose|-v]")
        sys.exit(1)
    
    network_dim = model_data.get('network_dim', 256)
    network = DeepCFRModule(nhandcards=3, nboardcards=6, n_action_history=20, nresponses=19, dim=network_dim)
    network.load_state_dict(model_data['strategy_network_state_dict'])
    network.eval()
    
    print(f"Network loaded. Hidden dim: {network_dim}")
    print(f"Training iterations: {model_data.get('iterations', 'unknown')}")
    
    print("\n" + "="*70)
    print("  NETWORK OUTPUT ANALYSIS")
    print("  Action order: " + ", ".join(ACTION_NAMES))
    print("="*70)
    
    # Run analysis for each category
    all_results = {}
    for category, infosets in TEST_CASES.items():
        results = analyze_network(network, infosets)
        all_results[category] = results
        print_results(category, results, compact=not verbose)
    
    # Print summary
    print_summary_stats(all_results)
    
    # Also show value networks if requested
    if verbose:
        print(f"\n{'='*70}")
        print("  VALUE NETWORKS (V0 and V1)")
        print(f"{'='*70}")
        
        sample_infoset = ("Sample preflop", "S0|H:7s0,8s0,9s0|B:|A:")
        
        for player in [0, 1]:
            key = f'network_p{player}_state_dict'
            if key in model_data:
                v_network = DeepCFRModule(nhandcards=3, nboardcards=6, n_action_history=20, nresponses=19, dim=network_dim)
                v_network.load_state_dict(model_data[key])
                v_network.eval()
                
                results = analyze_network(v_network, [sample_infoset])
                if results and not results[0].get('error'):
                    logits_str = ", ".join([f"{x:+.2f}" for x in results[0]['logits']])
                    print(f"\n  V{player} for '{sample_infoset[1]}':")
                    print(f"  Logits: [{logits_str}]")


if __name__ == "__main__":
    main()
