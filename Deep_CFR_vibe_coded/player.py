'''
Deep CFR Poker Bot - Uses trained neural network to play near-optimal poker
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import torch
import random
from pathlib import Path

# Deep CFR imports
try:
    from deep_cfr_encoding import encode_state
    from regret_network import regret_matching, sample_action
    from action_mapping import get_legal_mask, action_index_to_engine_action
    from train import load_checkpoint
    DEEP_CFR_AVAILABLE = True
except ImportError:
    DEEP_CFR_AVAILABLE = False
    print("Warning: Deep CFR modules not found, falling back to random play")


class Player(Bot):
    '''
    A Deep CFR poker bot that uses trained neural networks.
    '''

    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.
        Loads the best trained Deep CFR model (Epoch 16).
        
        Note: Tested ensemble averaging (multiple epochs) but single
        Epoch 16 outperformed all ensembles. See ENSEMBLE_RESULTS.txt
        '''
        self.regret_net = None
        self.use_deep_cfr = False

        if DEEP_CFR_AVAILABLE:
            # Load BEST SINGLE model: Epoch 16
            # Tested ensembles (8,12,16) and (14,16,18) but they performed worse
            # Epoch 16 is a unique peak: -430 chips vs Henry
            checkpoint_paths = [
                Path("checkpoints/deep_cfr_epoch16.pt"),
                Path("Deep_CFR/checkpoints/deep_cfr_epoch16.pt"),
                Path("../checkpoints/deep_cfr_epoch16.pt"),
            ]
            
            for checkpoint_path in checkpoint_paths:
                if checkpoint_path.exists():
                    try:
                        print(f"Loading Deep CFR model from {checkpoint_path}...")
                        self.regret_net, checkpoint = load_checkpoint(
                            str(checkpoint_path))
                        self.regret_net.eval()
                        self.use_deep_cfr = True
                        print(f"✓ Model loaded! Epoch {checkpoint['epoch']} (BEST model)")
                        print(f"  Performance: -0.43 BB/hand vs Henry")
                        break
                    except Exception as e:
                        print(f"Failed to load {checkpoint_path}: {e}")
            
            if not self.use_deep_cfr:
                print("Warning: No trained model found, falling back to random play")
                print("Train a model first: uv run python Deep_CFR/train.py")

        # Statistics
        self.actions_taken = 0
        self.deep_cfr_actions = 0

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.
        '''
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Called NUM_ROUNDS times.
        '''
        pass

    def get_action(self, game_state, round_state, active):
        '''
        Uses Deep CFR to select optimal actions.
        Falls back to random play if model not available.
        '''
        legal_actions = round_state.legal_actions()
        self.actions_taken += 1

        # === DEEP CFR STRATEGY ===
        if self.use_deep_cfr and self.regret_net is not None:
            try:
                return self._get_deep_cfr_action(game_state, round_state, active)
            except Exception as e:
                print(f"Deep CFR failed: {e}, falling back to random")
                # Fall through to random strategy

        # === FALLBACK: RANDOM STRATEGY ===
        return self._get_random_action(round_state, active, legal_actions)

    def _get_deep_cfr_action(self, game_state, round_state, active):
        '''
        Use trained Deep CFR model to select action.
        
        Uses Epoch 16 - the best single checkpoint from training.
        (Tested ensemble averaging but single model performed best!)
        '''
        # Encode the current state
        state_encoding = encode_state(game_state, round_state, active)

        # Get legal action mask
        legal_mask = get_legal_mask(round_state, active)

        # Predict regrets with the network
        with torch.no_grad():
            predicted_regrets = self.regret_net(state_encoding)

        # Convert regrets to strategy via regret matching
        strategy = regret_matching(predicted_regrets, legal_mask)

        # Sample action from strategy
        action_idx = sample_action(strategy)

        # Convert to engine action
        action = action_index_to_engine_action(action_idx, round_state)

        self.deep_cfr_actions += 1

        # Debug output (optional - comment out for performance)
        if self.actions_taken % 100 == 0:
            print(f"[Deep CFR] Actions: {self.deep_cfr_actions}/{self.actions_taken} "
                  f"({100*self.deep_cfr_actions/self.actions_taken:.1f}%)")

        return action

    def _get_random_action(self, round_state, active, legal_actions):
        '''
        Fallback: Simple random strategy.
        '''
        my_cards = round_state.hands[active]
        my_pip = round_state.pips[active]

        # Discard weakest card
        if DiscardAction in legal_actions:
            if len(my_cards) > 0:
                rank_order = {r: i for i, r in enumerate("23456789TJQKA")}
                weakest_idx = min(
                    range(len(my_cards)),
                    key=lambda i: rank_order.get(my_cards[i][0], -1)
                )
                return DiscardAction(weakest_idx)

        # Random raise
        if RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()
            if random.random() < 0.5:
                return RaiseAction(min_raise)

        # Check if possible
        if CheckAction in legal_actions:
            return CheckAction()

        # Random fold
        if random.random() < 0.25:
            return FoldAction()

        # Otherwise call
        return CallAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
