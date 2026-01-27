"""
Diagnostic script with hardcoded realistic scenarios for each game phase.
These scenarios are fixed, allowing comparison across different models.

Per "Toss or Hold'em" rules:
- Preflop: 3 hole cards, no board, betting only
- Discard round: After 2-card flop dealt, choose which card to discard
- Flop/Turn/River: 2 hole cards + board cards, betting only

Usage:
    python diagnose_scenarios.py <model_path>
    python diagnose_scenarios.py model1.pt model2.pt  # Compare two models
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

# =============================================================================
# HARDCODED REALISTIC SCENARIOS
# =============================================================================
# Format: "S{street}|H:{hand}|B:{board}|A:{history}"
# Hand is sorted descending by rank (highest first): D0=highest, D2=lowest
# Cards: {rank}s{suit} where rank=2-14, suit=0-3 (canonicalized)

SCENARIOS = {
    # =========================================================================
    # PREFLOP BETTING (S0, no board, 3 hole cards)
    # =========================================================================
    'PREFLOP BETTING': {
        'legal_indices': [3, 4, 5, 6, 7, 8],  # CHECK, CALL, FOLD, RAISES
        'scenarios': [
            # Premium hands
            ('Premium pair (AA-x)', 'S0|H:14s0,14s1,8s2|B:|A:'),
            ('High pair (KK-x)', 'S0|H:13s0,13s1,5s2|B:|A:'),
            ('Broadway (AKQ)', 'S0|H:14s0,13s0,12s0|B:|A:'),
            
            # Medium hands  
            ('Medium pair (99-x)', 'S0|H:9s0,9s1,6s2|B:|A:'),
            ('Connected (JT9)', 'S0|H:11s0,10s0,9s0|B:|A:'),
            ('One high (A-7-3)', 'S0|H:14s0,7s1,3s2|B:|A:'),
            
            # Weak hands
            ('Low cards (7-5-3)', 'S0|H:7s0,5s1,3s2|B:|A:'),
            ('Garbage (8-4-2)', 'S0|H:8s0,4s1,2s2|B:|A:'),
            
            # Facing raise
            ('Premium facing raise', 'S0|H:14s0,14s1,10s2|B:|A:R'),
            ('Weak facing raise', 'S0|H:7s0,5s1,2s2|B:|A:R'),
        ]
    },
    
    # =========================================================================
    # DISCARD ROUND (after 2-card flop, before flop betting)
    # Must choose which of 3 cards to discard
    # =========================================================================
    'DISCARD ROUND': {
        'legal_indices': [0, 1, 2],  # DISCARD_0, DISCARD_1, DISCARD_2
        'scenarios': [
            # Clear discard (low card obvious choice) - D2 should be preferred
            ('Clear low (A-K-3, board 9-7)', 'S0|H:14s0,13s0,3s1|B:9s2,7s2|A:CC'),
            ('Clear low (Q-J-2, board T-8)', 'S0|H:12s0,11s0,2s1|B:10s2,8s2|A:CC'),
            
            # Pair in hand (keep the pair, discard kicker)
            ('Pair + kicker (K-K-5, board A-9)', 'S0|H:13s0,13s1,5s2|B:14s2,9s2|A:CC'),
            ('Pair + kicker (9-9-3, board J-7)', 'S0|H:9s0,9s1,3s2|B:11s2,7s2|A:CC'),
            
            # Flush draw (keep suited cards)
            ('Flush draw (A-T-4 suited, board K-7)', 'S0|H:14s0,10s0,4s0|B:13s0,7s1|A:CC'),
            
            # Straight draw (keep connectors)
            ('Straight draw (J-T-5, board 9-8)', 'S0|H:11s0,10s0,5s1|B:9s2,8s2|A:CC'),
            
            # Tough decisions (all cards similar value)
            ('Close decision (K-Q-J, board A-9)', 'S0|H:13s0,12s0,11s0|B:14s1,9s1|A:CC'),
            ('Close decision (8-7-6, board T-5)', 'S0|H:8s0,7s0,6s0|B:10s1,5s1|A:CC'),
            
            # After opponent discarded
            ('After opp discard (A-K-2, board 9-7-Qopp)', 'S0|H:14s0,13s0,2s1|B:9s2,7s2,12s3|A:CCD'),
            ('After opp discard low', 'S0|H:10s0,8s0,3s1|B:14s2,9s2,5s3|A:CCD'),
        ]
    },
    
    # =========================================================================
    # FLOP BETTING (S1, 2 hole cards, 4 board cards after discards)
    # =========================================================================
    'FLOP BETTING': {
        'legal_indices': [3, 4, 5, 6, 7, 8],
        'scenarios': [
            # Strong made hands
            ('Top pair (K-Q, board K-9-7-5)', 'S1|H:13s0,12s0|B:13s1,9s2,7s2,5s2|A:CCDD'),
            ('Two pair (K-9, board K-9-7-5)', 'S1|H:13s0,9s0|B:13s1,9s1,7s2,5s2|A:CCDD'),
            ('Set (9-9, board K-9-7-5)', 'S1|H:9s0,9s1|B:13s2,9s2,7s2,5s2|A:CCDD'),
            
            # Drawing hands
            ('Flush draw (As-Ts, board Ks-7s-9h-5h)', 'S1|H:14s0,10s0|B:13s0,7s0,9s1,5s1|A:CCDD'),
            ('Open-ended (J-T, board 9-8-K-3)', 'S1|H:11s0,10s0|B:9s1,8s1,13s2,3s2|A:CCDD'),
            
            # Weak/missed
            ('Missed (A-Q, board K-9-7-5)', 'S1|H:14s0,12s0|B:13s1,9s2,7s2,5s2|A:CCDD'),
            ('Bottom pair (5-4, board K-9-7-5)', 'S1|H:5s0,4s0|B:13s1,9s2,7s2,5s1|A:CCDD'),
            
            # Facing aggression
            ('Top pair facing bet', 'S1|H:13s0,12s0|B:13s1,9s2,7s2,5s2|A:CCDDR'),
            ('Draw facing bet', 'S1|H:14s0,10s0|B:13s0,7s0,9s1,5s1|A:CCDDR'),
            ('Air facing bet', 'S1|H:6s0,4s0|B:13s1,9s2,7s2,5s2|A:CCDDR'),
        ]
    },
    
    # =========================================================================
    # TURN BETTING (S2, 2 hole cards, 5 board cards)
    # =========================================================================
    'TURN BETTING': {
        'legal_indices': [3, 4, 5, 6, 7, 8],
        'scenarios': [
            # Strong hands
            ('Top pair good kicker', 'S2|H:14s0,13s0|B:14s1,9s2,7s2,5s2,3s2|A:CCDDCC'),
            ('Two pair', 'S2|H:14s0,9s0|B:14s1,9s1,7s2,5s2,3s2|A:CCDDCC'),
            ('Set', 'S2|H:9s0,9s1|B:14s2,9s2,7s2,5s2,3s2|A:CCDDCC'),
            
            # Draws
            ('Flush draw turn', 'S2|H:14s0,10s0|B:13s0,7s0,9s1,5s1,2s1|A:CCDDCC'),
            ('Missed draw', 'S2|H:11s0,10s0|B:9s1,8s1,14s2,3s2,2s2|A:CCDDCC'),
            
            # Medium/weak
            ('Second pair', 'S2|H:9s0,8s0|B:14s1,9s1,7s2,5s2,3s2|A:CCDDCC'),
            ('Weak pair', 'S2|H:5s0,4s0|B:14s1,9s2,7s2,5s1,3s2|A:CCDDCC'),
            
            # Bigger pots
            ('Strong in raised pot', 'S2|H:14s0,14s1|B:14s2,9s2,7s2,5s2,3s2|A:CRDDRCC'),
            ('Draw in raised pot', 'S2|H:14s0,10s0|B:13s0,7s0,9s1,5s1,2s1|A:CRDDRCC'),
            ('Bluff catcher', 'S2|H:9s0,8s0|B:14s1,13s1,12s1,5s2,3s2|A:CCDDCCR'),
        ]
    },
    
    # =========================================================================
    # RIVER BETTING (S3, 2 hole cards, 6 board cards)
    # =========================================================================
    'RIVER BETTING': {
        'legal_indices': [3, 4, 5, 6, 7, 8],
        'scenarios': [
            # Value hands
            ('Top pair river', 'S3|H:14s0,13s0|B:14s1,9s2,7s2,5s2,3s2,2s2|A:CCDDCCCC'),
            ('Two pair river', 'S3|H:14s0,9s0|B:14s1,9s1,7s2,5s2,3s2,2s2|A:CCDDCCCC'),
            ('Set river', 'S3|H:9s0,9s1|B:14s2,9s2,7s2,5s2,3s2,2s2|A:CCDDCCCC'),
            ('Straight river', 'S3|H:11s0,10s0|B:9s1,8s1,7s2,5s2,3s2,2s2|A:CCDDCCCC'),
            
            # Bluff catchers
            ('Medium pair vs bet', 'S3|H:9s0,8s0|B:14s1,13s1,12s1,5s2,3s2,2s2|A:CCDDCCCCR'),
            ('Weak pair vs bet', 'S3|H:5s0,4s0|B:14s1,13s1,12s1,9s2,5s1,2s2|A:CCDDCCCCR'),
            
            # Bluffing spots
            ('Missed draw river', 'S3|H:14s0,10s0|B:13s1,7s1,9s2,5s2,3s2,2s2|A:CCDDCCCC'),
            ('Total air river', 'S3|H:6s0,4s0|B:14s1,13s1,12s1,9s2,7s2,2s2|A:CCDDCCCC'),
            
            # Big pots
            ('Nuts in big pot', 'S3|H:14s0,14s1|B:14s2,9s2,7s2,5s2,3s2,2s2|A:CRDDRCRC'),
            ('Bluff catcher big pot', 'S3|H:9s0,9s1|B:14s1,13s1,12s1,10s2,5s2,2s2|A:CRDDRCRCR'),
        ]
    },
}


def analyze_scenario(network, infoset_str, legal_indices):
    """Get network output for a scenario, filtered to legal actions."""
    try:
        cc, ah = parse_infoset_to_network_input(infoset_str)
        
        with torch.no_grad():
            output = network(cc, ah)
        
        logits = output[0]
        
        # Get logits for legal actions only
        legal_logits = torch.tensor([logits[i].item() for i in legal_indices])
        legal_probs = F.softmax(legal_logits, dim=0)
        
        # Build result
        probs_dict = {}
        for i, idx in enumerate(legal_indices):
            probs_dict[ACTION_NAMES[idx]] = legal_probs[i].item()
        
        dominant_idx = legal_indices[torch.argmax(legal_probs).item()]
        
        return {
            'probs': probs_dict,
            'dominant': ACTION_NAMES[dominant_idx],
            'dominant_prob': torch.max(legal_probs).item(),
            'error': None
        }
    except Exception as e:
        return {'error': str(e)}


def print_phase_results(phase_name, phase_data, results):
    """Print results for a phase with nice formatting."""
    legal_indices = phase_data['legal_indices']
    action_names = [ACTION_NAMES[i] for i in legal_indices]
    
    # Header
    print(f"\n{'─' * 90}")
    print(f"  {phase_name}")
    print(f"{'─' * 90}")
    
    # Column header
    if len(legal_indices) == 3:  # Discard phase
        print(f"  {'Scenario':<40} │ {'D0':>6} │ {'D1':>6} │ {'D2':>6} │ Dominant")
    else:  # Betting phase
        print(f"  {'Scenario':<40} │ {'CHK':>5} │{'CALL':>5} │{'FOLD':>5} │{'R_S':>5} │{'R_M':>5} │{'R_L':>5} │ Dominant")
    print(f"  {'─' * 40}─┼{'─' * (7 if len(legal_indices) == 3 else 6) * len(legal_indices)}─┼─────────")
    
    # Results
    avg_probs = {name: 0.0 for name in action_names}
    valid_count = 0
    
    for (desc, _), result in zip(phase_data['scenarios'], results):
        if result.get('error'):
            print(f"  {desc:<40} │ ERROR: {result['error'][:30]}")
            continue
        
        valid_count += 1
        probs = result['probs']
        
        for name in action_names:
            avg_probs[name] += probs.get(name, 0)
        
        if len(legal_indices) == 3:
            row = f"  {desc:<40} │ {probs.get('DISCARD_0', 0):>5.1%} │ {probs.get('DISCARD_1', 0):>5.1%} │ {probs.get('DISCARD_2', 0):>5.1%} │ {result['dominant']}"
        else:
            row = f"  {desc:<40} │{probs.get('CHECK', 0):>5.1%}│{probs.get('CALL', 0):>5.1%}│{probs.get('FOLD', 0):>5.1%}│{probs.get('RAISE_S', 0):>5.1%}│{probs.get('RAISE_M', 0):>5.1%}│{probs.get('RAISE_L', 0):>5.1%}│ {result['dominant']}"
        print(row)
    
    # Average row
    if valid_count > 0:
        for name in avg_probs:
            avg_probs[name] /= valid_count
        
        print(f"  {'─' * 40}─┼{'─' * (7 if len(legal_indices) == 3 else 6) * len(legal_indices)}─┼─────────")
        if len(legal_indices) == 3:
            print(f"  {'AVERAGE':<40} │ {avg_probs.get('DISCARD_0', 0):>5.1%} │ {avg_probs.get('DISCARD_1', 0):>5.1%} │ {avg_probs.get('DISCARD_2', 0):>5.1%} │")
        else:
            print(f"  {'AVERAGE':<40} │{avg_probs.get('CHECK', 0):>5.1%}│{avg_probs.get('CALL', 0):>5.1%}│{avg_probs.get('FOLD', 0):>5.1%}│{avg_probs.get('RAISE_S', 0):>5.1%}│{avg_probs.get('RAISE_M', 0):>5.1%}│{avg_probs.get('RAISE_L', 0):>5.1%}│")
    
    return avg_probs


def analyze_model(model_path):
    """Analyze a single model."""
    print(f"\nLoading model: {model_path}")
    
    try:
        model_data = torch.load(model_path, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print(f"ERROR: Model not found: {model_path}")
        return None
    
    network_dim = model_data.get('network_dim', 256)
    iterations = model_data.get('iterations', '?')
    
    network = DeepCFRModule(nhandcards=3, nboardcards=5, n_action_history=20, nresponses=9, dim=network_dim)
    network.load_state_dict(model_data['strategy_network_state_dict'])
    network.eval()
    
    print(f"  Network dim: {network_dim}, Iterations: {iterations}")
    
    print("\n" + "═" * 90)
    print(f"  MODEL: {os.path.basename(model_path)} ({iterations} iterations)")
    print("═" * 90)
    
    all_avgs = {}
    
    for phase_name, phase_data in SCENARIOS.items():
        results = []
        for desc, infoset in phase_data['scenarios']:
            result = analyze_scenario(network, infoset, phase_data['legal_indices'])
            results.append(result)
        
        avg_probs = print_phase_results(phase_name, phase_data, results)
        all_avgs[phase_name] = avg_probs
    
    return all_avgs


def print_summary_comparison(models_data):
    """Print summary comparison of multiple models."""
    if len(models_data) < 2:
        return
    
    print("\n" + "═" * 90)
    print("  SUMMARY COMPARISON")
    print("═" * 90)
    
    model_names = list(models_data.keys())
    
    # Betting phases comparison
    print("\n  BETTING PHASES (Average CHECK probability):")
    print(f"  {'Phase':<20}", end="")
    for name in model_names:
        print(f" │ {name[:15]:<15}", end="")
    print()
    
    betting_phases = ['PREFLOP BETTING', 'FLOP BETTING', 'TURN BETTING', 'RIVER BETTING']
    for phase in betting_phases:
        print(f"  {phase:<20}", end="")
        for name in model_names:
            if phase in models_data[name]:
                check_prob = models_data[name][phase].get('CHECK', 0)
                print(f" │ {check_prob:>14.1%}", end="")
            else:
                print(f" │ {'N/A':>14}", end="")
        print()
    
    # Discard phase comparison
    print("\n  DISCARD PHASE:")
    print(f"  {'Action':<20}", end="")
    for name in model_names:
        print(f" │ {name[:15]:<15}", end="")
    print()
    
    for action in ['DISCARD_0', 'DISCARD_1', 'DISCARD_2']:
        label = f"{action} (D0=highest)"
        print(f"  {label:<20}", end="")
        for name in model_names:
            if 'DISCARD ROUND' in models_data[name]:
                prob = models_data[name]['DISCARD ROUND'].get(action, 0)
                print(f" │ {prob:>14.1%}", end="")
            else:
                print(f" │ {'N/A':>14}", end="")
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_scenarios.py <model_path> [model2_path ...]")
        print("\nExamples:")
        print("  python diagnose_scenarios.py output/models/model.pt")
        print("  python diagnose_scenarios.py model_v1.pt model_v2.pt  # Compare models")
        sys.exit(1)
    
    model_paths = [arg for arg in sys.argv[1:] if not arg.startswith('-')]
    
    print("=" * 90)
    print("  SCENARIO-BASED MODEL DIAGNOSTICS")
    print("  Fixed scenarios for comparing models")
    print("=" * 90)
    print(f"\n  D0 = HIGHEST rank card, D1 = middle, D2 = LOWEST rank card")
    print(f"  (In poker, you usually want to KEEP high cards → D2 should be preferred)")
    
    models_data = {}
    
    for model_path in model_paths:
        result = analyze_model(model_path)
        if result:
            models_data[os.path.basename(model_path)] = result
    
    if len(models_data) > 1:
        print_summary_comparison(models_data)
    
    print("\n" + "═" * 90)
    print("  DIAGNOSTICS COMPLETE")
    print("═" * 90)


if __name__ == "__main__":
    main()
