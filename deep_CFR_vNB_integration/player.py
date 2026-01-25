"""
Deep CFR Player Bot

This bot loads a trained Deep CFR model and uses it to make decisions.
The model predicts regrets for any game state, enabling Nash equilibrium play.
"""

import sys
import os
from pathlib import Path

# Add current directory to path (for local imports)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import torch
import random
import numpy as np

from skeleton.bot import Bot
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.runner import parse_args, run_bot
from skeleton.states import RoundState

from network.model import DeepCFRModule
from core.mccfr import MCCFR
from core.integration import NetworkMCCFRIntegration


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
            model_path = Path(__file__).parent / 'output' / 'models' / 'deep_cfr_model.pt'
        else:
            model_path = Path(model_path)
        
        # Try to load the model
        if model_path.exists():
            try:
                print(f"Loading Deep CFR model from {model_path}...")
                model_data = torch.load(model_path, map_location='cpu')
                
                # Get network configuration
                network_dim = model_data.get('network_dim', 256)
                
                # Create network
                self.network = DeepCFRModule(
                    nhandcards=3,
                    nboardcards=5,
                    n_action_history=20,
                    nresponses=9,
                    dim=network_dim
                )
                
                # Load weights - prefer strategy network (average strategy for play)
                if 'strategy_network_state_dict' in model_data:
                    self.network.load_state_dict(model_data['strategy_network_state_dict'])
                elif 'network_state_dict' in model_data:
                    self.network.load_state_dict(model_data['network_state_dict'])
                else:
                    raise ValueError("No valid network weights found in model file")
                self.network.eval()  # Set to evaluation mode
                
                # Create MCCFR instance (needed for integration)
                self.mccfr = MCCFR()
                
                # Create integration
                self.integration = NetworkMCCFRIntegration(
                    network=self.network,
                    mccfr=self.mccfr
                )
                
                self.model_loaded = True
                
                print(f"✓ Deep CFR Model Loaded!")
                print(f"  Network dim: {network_dim}")
                if 'final_loss' in model_data and model_data['final_loss'] is not None:
                    print(f"  Training loss: {model_data['final_loss']:,.0f}")
                if 'training_samples' in model_data:
                    print(f"  Training samples: {model_data['training_samples']}")
                print(f"  Ready to play with Nash equilibrium strategy!")
                
            except Exception as e:
                print(f"✗ Failed to load model: {e}")
                import traceback
                traceback.print_exc()
                print("Falling back to random play")
        else:
            print(f"✗ Model file not found: {model_path}")
            print("Train a model first with: python train_model.py")
            print("Falling back to random play")

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
        try:
            # Get legal actions
            legal_action_types = round_state.legal_actions()
            
            if not legal_action_types:
                # Should never happen, but handle gracefully
                return CheckAction()
            
            # If model not loaded, fall back to random
            if not self.model_loaded or self.integration is None:
                return self._random_action(round_state, legal_action_types)
            
            # Use network to select action
            action = self.integration.select_network_action(round_state, active)
            
            return action
            
        except Exception as e:
            # If anything fails, fall back to random play
            print(f"Error in get_action: {e}")
            import traceback
            traceback.print_exc()
            
            # Safe fallback
            return self._random_action(round_state, round_state.legal_actions())
    
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
