"""
Deep CFR Player Bot

This bot loads a trained Deep CFR model and uses it to make decisions.
The model predicts regrets for any game state, enabling Nash equilibrium play.
"""

from core.integration import NetworkMCCFRIntegration
from core.mccfr import MCCFR
from network.model import DeepCFRModule
from skeleton.states import RoundState
from skeleton.runner import parse_args, run_bot
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.bot import Bot
import numpy as np
import random
import torch
import sys
import os
from pathlib import Path

# Add current directory to path (for local imports)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


class Player(Bot):
    """
    Deep CFR Player Bot - Uses trained neural network to predict regrets.
    """

    def __init__(self, model_path: str = None):
        """
        Initialize the bot by loading the trained model.

        Args:
            model_path: Path to the trained model file. If None, looks for 'deep_cfr_model.pt'
        """
        self.network = None
        self.mccfr = None
        self.integration = None
        self.model_loaded = False

        # Default model path (look in output/models/ folder)
        if model_path is None:
            raise Exception("Model path is not provided cannot run")
        else:
            model_path = Path(model_path)

        # Try to load the model - fail hard if it doesn't work
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Train a model first with: python train_model.py\n"
                f"Or specify a model with: --model <filename>"
            )
        
        try:
            print(f"Loading Deep CFR model from {model_path}...")
            model_data = torch.load(model_path, map_location='cpu')

            # Get network configuration
            network_dim = model_data.get('network_dim', 256)

            # Create network
            self.network = DeepCFRModule(
                nhandcards=3,
                nboardcards=6,  # 2 flop + 2 discards + turn + river = 6 max
                n_action_history=20,
                nresponses=19,  # 3 discards + 3 basic + 13 pot-relative raises (25%-500% + all-in)
                dim=network_dim
            )

            # Load weights - prefer strategy network (average strategy for play)
            if 'strategy_network_state_dict' in model_data:
                self.network.load_state_dict(
                    model_data['strategy_network_state_dict'])
            elif 'network_state_dict' in model_data:
                self.network.load_state_dict(
                    model_data['network_state_dict'])
            else:
                raise ValueError(
                    "No valid network weights found in model file")
            self.network.eval()  # Set to evaluation mode

            # Create MCCFR instance (needed for integration)
            self.mccfr = MCCFR()

            # Create integration
            # Player uses strategy network (average strategy), so outputs are logits → softmax
            self.integration = NetworkMCCFRIntegration(
                network=self.network,
                mccfr=self.mccfr,
                is_strategy_network=True  # Strategy network outputs are logits (per paper Section 5.1)
            )

            self.model_loaded = True

            print(f"✓ Deep CFR Model Loaded!")
            print(f"  Network dim: {network_dim}")
            if 'final_loss' in model_data and model_data['final_loss'] is not None:
                print(f"  Training loss: {model_data['final_loss']:,.0f}")
            if 'training_samples' in model_data:
                print(
                    f"  Training samples: {model_data['training_samples']}")
            print(f"  Ready to play with Nash equilibrium strategy!")

        except Exception as e:
            # Re-raise with more context - don't silently fail
            raise RuntimeError(
                f"Failed to load Deep CFR model from {model_path}: {e}\n"
                f"Make sure the model file is valid and contains the required weights."
            ) from e

    def handle_new_round(self, game_state, round_state, active):
        """
        Called when a new round starts.
        """
        pass  # Deep CFR is stateless, nothing to do here

    def handle_round_over(self, game_state, terminal_state, active):
        """
        Called when a round ends.
        """
        pass  # Deep CFR doesn't need to learn during play

    def get_action(self, game_state, round_state, active):
        """
        Query the trained network for the current game state.

        Args:
            game_state: The game state object
            round_state: The current round state
            active: The active player (0 or 1)

        Returns:
            Action to take
        """
        # Get legal actions
        legal_action_types = round_state.legal_actions()

        if not legal_action_types:
            # Should never happen, but handle gracefully
            return CheckAction()

        # Model must be loaded - fail hard if not
        if not self.model_loaded or self.integration is None:
            raise RuntimeError(
                "Deep CFR model not loaded! Cannot play without a trained model.\n"
                "Make sure the model was successfully loaded during initialization."
            )

        # Use network to select action
        try:
            action = self.integration.select_network_action(
                round_state, active)
            return action
        except Exception as e:
            # Re-raise with context - don't silently fall back to random
            raise RuntimeError(
                f"Error selecting action with Deep CFR model: {e}\n"
                f"This indicates a problem with the model or integration."
            ) from e

    def _random_action(self, round_state, legal_action_types):
        """Fallback to random legal action."""
        legal_actions = []
        active = round_state.button % 2

        for action_type in legal_action_types:
            if action_type == RaiseAction:
                min_raise, max_raise = round_state.raise_bounds()
                raise_sizes = [
                    min_raise,
                    (min_raise + max_raise) // 2,
                    max_raise
                ]
                raise_sizes = sorted(list(set(raise_sizes)))
                for size in raise_sizes:
                    legal_actions.append(RaiseAction(size))
            elif action_type == DiscardAction:
                for card_idx in range(len(round_state.hands[active])):
                    legal_actions.append(DiscardAction(card_idx))
            else:
                legal_actions.append(action_type())

        if not legal_actions:
            return CheckAction()

        return random.choice(legal_actions)


if __name__ == '__main__':
    # Allow model path to be specified via environment variable
    model_path = os.environ.get('DEEP_CFR_MODEL_PATH', None)
    player = Player(model_path=model_path)
    run_bot(player, parse_args())
