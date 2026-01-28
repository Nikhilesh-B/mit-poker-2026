"""
Diagnostic script that generates random valid infosets for each game phase
and shows probability distributions.

Per "Toss or Hold'em" rules:
- Preflop betting: betting only (no discards)
- Discard round: discards only (after flop dealt, before flop betting)
- Flop/Turn/River betting: betting only

Usage:
    python diagnose_by_street.py <model_path> [--samples N]
"""
import sys
import os
import random
import torch
import torch.nn.functional as F
from collections import defaultdict

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

from network.model import DeepCFRModule
from utils.infoset_parser import parse_infoset_to_network_input

ACTION_NAMES = ['DISCARD_0', 'DISCARD_1', 'DISCARD_2', 'CHECK', 'CALL', 'FOLD', 'RAISE_S', 'RAISE_M', 'RAISE_L']
ACTION_SHORT = ['D0', 'D1', 'D2', 'CHK', 'CALL', 'FOLD', 'R_S', 'R_M', 'R_L']

# Game phases per "Toss or Hold'em" rules
PHASES = [
    {
        'name': 'PREFLOP BETTING',
        'description': 'Preflop betting (3 hole cards, no board)',
        'street': 0,
        'board_cards': 0,
        'is_discard': False,
        'legal_indices': [3, 4, 5, 6, 7, 8],  # CHECK, CALL, FOLD, RAISES
    },
    {
        'name': 'DISCARD ROUND',
        'description': 'Discard phase (after flop dealt, choose card to discard)',
        'street': 0,  # Encoded as S0 in infoset but with board
        'board_cards': 2,  # 2 flop cards before our discard
        'is_discard': True,
        'legal_indices': [0, 1, 2],  # DISCARD_0, DISCARD_1, DISCARD_2
    },
    {
        'name': 'FLOP BETTING',
        'description': 'Flop betting (2 hole cards, 4 board cards after discards)',
        'street': 1,
        'board_cards': 4,  # 2 flop + 2 discards
        'is_discard': False,
        'legal_indices': [3, 4, 5, 6, 7, 8],
    },
    {
        'name': 'TURN BETTING',
        'description': 'Turn betting (2 hole cards, 5 board cards)',
        'street': 2,
        'board_cards': 5,
        'is_discard': False,
        'legal_indices': [3, 4, 5, 6, 7, 8],
    },
    {
        'name': 'RIVER BETTING',
        'description': 'River betting (2 hole cards, 6 board cards)',
        'street': 3,
        'board_cards': 6,
        'is_discard': False,
        'legal_indices': [3, 4, 5, 6, 7, 8],
    },
]


def random_card(exclude=None):
    """Generate a random card in format {rank}s{suit}."""
    exclude = exclude or set()
    while True:
        rank = random.randint(2, 14)  # 2-14 (14=Ace)
        suit = random.randint(0, 3)
        card = f"{rank}s{suit}"
        if card not in exclude:
            return card


def random_hand(n_cards, exclude=None):
    """Generate n random cards for a hand."""
    exclude = exclude or set()
    cards = []
    for _ in range(n_cards):
        card = random_card(exclude)
        cards.append(card)
        exclude.add(card)
    return cards, exclude


def random_board(n_cards, exclude=None):
    """Generate n board cards."""
    exclude = exclude or set()
    cards = []
    for _ in range(n_cards):
        card = random_card(exclude)
        cards.append(card)
        exclude.add(card)
    return cards, exclude


def random_action_history(phase):
    """Generate a plausible action history for a given phase."""
    street = phase['street']
    is_discard = phase['is_discard']
    
    if street == 0 and not is_discard:
        # Preflop betting: 0-3 actions (C=check/call, R=raise)
        length = random.randint(0, 3)
        actions = [random.choices(['C', 'R'], weights=[0.7, 0.3])[0] for _ in range(length)]
    elif is_discard:
        # Discard phase: preflop betting completed, maybe opponent discard
        preflop_actions = random.randint(1, 3)
        actions = [random.choices(['C', 'R'], weights=[0.7, 0.3])[0] for _ in range(preflop_actions)]
        # Maybe opponent already discarded
        if random.random() < 0.5:
            actions.append('D')
    else:
        # Post-flop betting: need history from preflop + discards + current street
        # Simplified: just generate some betting actions
        length = 2 + street * 2 + random.randint(0, 2)
        actions = [random.choices(['C', 'R'], weights=[0.7, 0.3])[0] for _ in range(length)]
    
    return ''.join(actions)


def categorize_hand_strength(hand_cards, board_cards):
    """Simple hand strength categorization."""
    ranks = [int(c.split('s')[0]) for c in hand_cards]
    
    # Check for pairs
    has_pair = len(ranks) != len(set(ranks))
    
    # Check for high cards (J+)
    high_count = sum(1 for r in ranks if r >= 11)
    
    # Avg rank
    avg_rank = sum(ranks) / len(ranks)
    
    if has_pair and avg_rank >= 10:
        return "strong"
    elif has_pair or high_count >= 2:
        return "medium"
    elif avg_rank >= 10:
        return "medium"
    else:
        return "weak"


def generate_random_infosets(phase, n_samples=10):
    """Generate n random valid infosets for a given game phase."""
    infosets = []
    
    for _ in range(n_samples):
        exclude = set()
        
        # Generate hand (3 cards pre-discard, 2 cards post-discard)
        if phase['is_discard'] or phase['street'] == 0:
            n_hand = 3  # Pre-discard
        else:
            n_hand = 2  # Post-discard (for simplicity, though network sees 3)
        
        # Actually, the network always expects 3 hand cards in the encoding
        hand, exclude = random_hand(3, exclude)
        
        # Generate board
        board, exclude = random_board(phase['board_cards'], exclude)
        
        # Generate action history
        history = random_action_history(phase)
        
        # Build infoset string
        hand_str = ','.join(hand)
        board_str = ','.join(board) if board else ''
        
        infoset = f"S{phase['street']}|H:{hand_str}|B:{board_str}|A:{history}"
        
        # Categorize hand strength
        strength = categorize_hand_strength(hand, board)
        
        infosets.append({
            'infoset': infoset,
            'hand': hand,
            'board': board,
            'history': history,
            'strength': strength,
            'phase': phase['name']
        })
    
    return infosets


def analyze_infoset(network, infoset_str, legal_indices):
    """Get network output for an infoset, filtered to legal actions."""
    try:
        cc, ah = parse_infoset_to_network_input(infoset_str)
        
        with torch.no_grad():
            output = network(cc, ah)
        
        logits = output[0]
        
        # Get logits for legal actions only
        legal_logits = torch.tensor([logits[i].item() for i in legal_indices])
        legal_probs_tensor = F.softmax(legal_logits, dim=0)
        
        # Build full probability array with 0 for illegal actions
        legal_probs = [0.0] * 9
        for i, idx in enumerate(legal_indices):
            legal_probs[idx] = legal_probs_tensor[i].item()
        
        # Find dominant among LEGAL actions only
        dominant_legal_idx = legal_indices[torch.argmax(legal_probs_tensor).item()]
        
        return {
            'logits': logits.tolist(),
            'probs': legal_probs,
            'dominant_idx': dominant_legal_idx,
            'legal_indices': legal_indices,
            'error': None
        }
    except Exception as e:
        return {'error': str(e)}


def print_phase_table(phase, results):
    """Print a formatted table of results for a phase."""
    
    # Determine which columns to show based on legal actions
    legal_indices = phase['legal_indices']
    
    # Calculate average probabilities
    avg_probs = [0.0] * 9
    valid_count = 0
    
    for r in results:
        if r.get('error'):
            continue
        valid_count += 1
        for i, p in enumerate(r['probs']):
            avg_probs[i] += p
    
    if valid_count > 0:
        avg_probs = [p / valid_count for p in avg_probs]
    
    # Build header with only legal actions
    legal_names = [ACTION_SHORT[i] for i in legal_indices]
    header = "  " + " | ".join([f"{name:>5}" for name in legal_names])
    print(header)
    print("  " + "-" * len(header))
    
    # Print each sample
    for i, r in enumerate(results[:10]):  # Show max 10
        if r.get('error'):
            print(f"  ERROR: {r['error'][:40]}")
            continue
        
        probs = r['probs']
        legal_probs = [probs[idx] for idx in legal_indices]
        row = "  " + " | ".join([f"{p:>5.1%}" for p in legal_probs])
        
        # Mark dominant action
        dom_idx = r['dominant_idx']
        strength = r.get('strength', '?')[0].upper()
        print(f"{row}  [{strength}] {ACTION_NAMES[dom_idx]}")
    
    # Print averages
    print("  " + "-" * len(header))
    legal_avg = [avg_probs[idx] for idx in legal_indices]
    avg_row = "  " + " | ".join([f"{p:>5.1%}" for p in legal_avg])
    print(f"{avg_row}  [AVG]")
    
    return avg_probs


def main():
    # Parse args
    model_path = "output/models/mini_test_v4.pt"
    n_samples = 10
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    for i, arg in enumerate(sys.argv):
        if arg in ['--samples', '-n'] and i + 1 < len(sys.argv):
            n_samples = int(sys.argv[i + 1])
    
    print(f"Loading model: {model_path}")
    
    try:
        model_data = torch.load(model_path, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print(f"ERROR: Model not found: {model_path}")
        print("\nUsage: python diagnose_by_street.py <model_path> [--samples N]")
        sys.exit(1)
    
    network_dim = model_data.get('network_dim', 256)
    network = DeepCFRModule(nhandcards=3, nboardcards=6, n_action_history=20, nresponses=19, dim=network_dim)
    network.load_state_dict(model_data['strategy_network_state_dict'])
    network.eval()
    
    print(f"Network dim: {network_dim}, Iterations: {model_data.get('iterations', '?')}")
    print(f"Generating {n_samples} random infosets per phase...")
    print()
    print("Per 'Toss or Hold'em' rules: betting and discards are NEVER simultaneous!")
    print()
    
    # Store all results for final summary
    all_phase_avgs = {}
    
    for phase in PHASES:
        print("=" * 80)
        print(f"  {phase['name']}: {phase['description']}")
        print("=" * 80)
        
        # Generate random infosets
        infosets = generate_random_infosets(phase, n_samples)
        
        # Analyze each
        results = []
        for info in infosets:
            analysis = analyze_infoset(network, info['infoset'], phase['legal_indices'])
            analysis.update(info)
            results.append(analysis)
        
        # Print results
        print("\n  ALL HANDS:")
        avg_probs = print_phase_table(phase, results)
        all_phase_avgs[phase['name']] = {
            'avg_probs': avg_probs,
            'legal_indices': phase['legal_indices']
        }
        
        # Count by strength
        strength_counts = defaultdict(int)
        for r in results:
            strength_counts[r.get('strength', 'unknown')] += 1
        
        print(f"\n  Hand strength distribution: {dict(strength_counts)}")
        print()
    
    # Final summary tables
    print()
    print("=" * 80)
    print("  SUMMARY TABLES")
    print("=" * 80)
    
    # Betting phases table
    print()
    print("  ┌─────────────────────────────────────────────────────────────────────────┐")
    print("  │                        BETTING PHASES                                   │")
    print("  ├──────────────────┬────────┬────────┬────────┬────────┬────────┬────────┤")
    print("  │ Phase            │  CHECK │  CALL  │  FOLD  │ RAISE_S│ RAISE_M│ RAISE_L│")
    print("  ├──────────────────┼────────┼────────┼────────┼────────┼────────┼────────┤")
    
    betting_phases = ['PREFLOP BETTING', 'FLOP BETTING', 'TURN BETTING', 'RIVER BETTING']
    for phase_name in betting_phases:
        if phase_name in all_phase_avgs:
            data = all_phase_avgs[phase_name]
            probs = [data['avg_probs'][i] for i in data['legal_indices']]
            row = " │ ".join([f"{p:>5.1%}" for p in probs])
            print(f"  │ {phase_name:<16} │ {row} │")
    
    print("  └──────────────────┴────────┴────────┴────────┴────────┴────────┴────────┘")
    
    # Discard phase table
    print()
    print("  ┌───────────────────────────────────────────┐")
    print("  │              DISCARD PHASE                │")
    print("  ├──────────────────┬────────┬────────┬──────┤")
    print("  │ Phase            │  D0    │  D1    │  D2  │")
    print("  ├──────────────────┼────────┼────────┼──────┤")
    
    if 'DISCARD ROUND' in all_phase_avgs:
        data = all_phase_avgs['DISCARD ROUND']
        probs = [data['avg_probs'][i] for i in data['legal_indices']]
        print(f"  │ {'DISCARD ROUND':<16} │ {probs[0]:>5.1%} │ {probs[1]:>5.1%} │{probs[2]:>5.1%}│")
    
    print("  └──────────────────┴────────┴────────┴──────┘")
    
    # Dominant action summary table
    print()
    print("  ┌─────────────────────────────────────────────┐")
    print("  │           DOMINANT ACTIONS                  │")
    print("  ├──────────────────┬─────────────┬────────────┤")
    print("  │ Phase            │ Action      │ Probability│")
    print("  ├──────────────────┼─────────────┼────────────┤")
    
    for phase in PHASES:
        if phase['name'] in all_phase_avgs:
            data = all_phase_avgs[phase['name']]
            legal_probs = [data['avg_probs'][i] for i in data['legal_indices']]
            dom_local_idx = legal_probs.index(max(legal_probs))
            dom_global_idx = data['legal_indices'][dom_local_idx]
            dom_prob = max(legal_probs)
            print(f"  │ {phase['name']:<16} │ {ACTION_NAMES[dom_global_idx]:<11} │ {dom_prob:>9.1%} │")
    
    print("  └──────────────────┴─────────────┴────────────┘")
    
    # Model info footer
    print()
    print(f"  Model: {model_path}")
    print(f"  Iterations: {model_data.get('iterations', '?')}, Network dim: {network_dim}")


if __name__ == "__main__":
    main()
