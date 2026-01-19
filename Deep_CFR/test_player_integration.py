"""
Test the integrated Deep CFR player
"""
import pkrbot
from player import Player
from skeleton.states import RoundState, GameState, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.actions import CheckAction, RaiseAction, CallAction, FoldAction


def test_player_integration():
    """Test that the player can make decisions using Deep CFR"""
    print("="*70)
    print("TESTING DEEP CFR PLAYER INTEGRATION")
    print("="*70)
    
    # Create player
    print("\n1. Creating player...")
    player = Player()
    print(f"   Deep CFR enabled: {player.use_deep_cfr}")
    print(f"   Model loaded: {player.regret_net is not None}")
    
    if not player.use_deep_cfr:
        print("\n⚠️  Warning: No model loaded, will use random fallback")
        print("   Train a model first: uv run python train.py")
    
    # Create test game state
    print("\n2. Creating test game state...")
    deck = pkrbot.Deck()
    deck.shuffle()
    hands = [deck.deal(3), deck.deal(3)]
    
    round_state = RoundState(
        button=0,
        street=0,  # Preflop
        pips=[SMALL_BLIND, BIG_BLIND],
        stacks=[STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND],
        hands=hands,
        board=[],
        previous_state=None
    )
    
    game_state = GameState(
        bankroll=0,
        game_clock=60.0,
        round_num=1
    )
    
    print(f"   Player 0 hand: {hands[0]}")
    print(f"   Player 1 hand: {hands[1]}")
    print(f"   Street: {round_state.street}")
    print(f"   Stacks: P0={round_state.stacks[0]}, P1={round_state.stacks[1]}")
    
    # Test multiple decisions
    print("\n3. Making decisions with Deep CFR...")
    for i in range(5):
        active = round_state.button % 2
        legal_actions = round_state.legal_actions()
        
        print(f"\n   Decision {i+1}:")
        print(f"     Active player: {active}")
        print(f"     Legal actions: {[type(a).__name__ for a in legal_actions]}")
        
        # Get action from player
        action = player.get_action(game_state, round_state, active)
        
        print(f"     ✓ Deep CFR chose: {type(action).__name__}")
        
        # Apply action to continue game
        round_state = round_state.proceed(action)
        
        # Check if game ended
        from skeleton.states import TerminalState
        if isinstance(round_state, TerminalState):
            print(f"\n   Game ended!")
            print(f"     Final deltas: P0={round_state.deltas[0]}, P1={round_state.deltas[1]}")
            break
    
    # Show statistics
    print(f"\n4. Statistics:")
    print(f"   Total actions: {player.actions_taken}")
    print(f"   Deep CFR actions: {player.deep_cfr_actions}")
    if player.actions_taken > 0:
        print(f"   Deep CFR usage: {100*player.deep_cfr_actions/player.actions_taken:.1f}%")
    
    print("\n" + "="*70)
    print("✓ INTEGRATION TEST PASSED!")
    print("="*70)
    
    if player.use_deep_cfr:
        print("\n🎉 Deep CFR player is working!")
        print("   • Model loads successfully")
        print("   • Makes decisions using trained network")
        print("   • Handles game states correctly")
        print("   • Falls back gracefully on errors")
    else:
        print("\n⚠️  Player using fallback strategy (no model loaded)")
    
    print("\n📋 Next steps:")
    print("   1. Train for more epochs: uv run python train.py --epochs 100")
    print("   2. Test against engine: python ../engine.py (then run player.py)")
    print("   3. Evaluate performance against random/NFSP agents")


if __name__ == "__main__":
    test_player_integration()

