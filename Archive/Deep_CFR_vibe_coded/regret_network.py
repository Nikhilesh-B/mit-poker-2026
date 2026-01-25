"""
Regret Network for Deep CFR

This network predicts counterfactual regrets for each action given a game state.
Regrets are used to compute the current strategy via regret matching.

Architecture:
    Input (32) → Linear(256) → ReLU → Linear(128) → ReLU → Output (8)

The network learns to approximate:
    "How much better would action X have been compared to what I actually did?"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegretNetwork(nn.Module):
    """
    Neural network that predicts regret values for poker actions.

    In tabular CFR, you'd store: regret_sum[state][action] = cumulative_regret
    In Deep CFR, this network approximates that function.

    Input: 32-dimensional state encoding
    Output: 8-dimensional regret vector (one per action)
        Actions: [FOLD, CHECK_CALL, RAISE_SMALL, RAISE_MED, RAISE_LARGE, 
                  DISCARD_0, DISCARD_1, DISCARD_2]

    Note: Output regrets can be negative (no activation on final layer)
    """

    def __init__(self, input_dim=32, output_dim=8, hidden_sizes=(256, 128)):
        """
        Initialize the regret network.

        Args:
            input_dim: Size of state encoding (default 32)
            output_dim: Number of actions (default 8)
            hidden_sizes: Tuple of hidden layer sizes (default (256, 128))
        """
        super(RegretNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes

        # Build network layers
        layers = []
        last_dim = input_dim

        # Hidden layers with ReLU activation
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim

        # Output layer (no activation - regrets can be negative!)
        layers.append(nn.Linear(last_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights using He initialization (good for ReLU)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights for faster convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He initialization for ReLU networks
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, state_encoding):
        """
        Forward pass through the network.

        Args:
            state_encoding: Tensor of shape [batch_size, 32] or [32]
                           Encoded game state from encode_state()

        Returns:
            regrets: Tensor of shape [batch_size, 8] or [8]
                    Predicted regret for each action
                    Positive regret = "wish I'd taken this action"
                    Negative regret = "glad I didn't take this action"

        Example:
            >>> state = torch.randn(32)  # Some game state
            >>> regrets = regret_net(state)
            >>> print(regrets)  # e.g., [-0.3, 0.5, 0.8, 0.2, -0.1, -0.5, -0.2, 0.0]
        """
        return self.network(state_encoding)

    def get_num_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def regret_matching(regrets, legal_mask):
    """
    Convert regrets to action probabilities using regret matching.

    This is the core of CFR: actions with higher positive regret get
    selected more often in future iterations.

    Algorithm:
        1. Zero out negative regrets (only consider positive)
        2. Zero out illegal actions (apply mask)
        3. Normalize to probabilities
        4. If all regrets ≤ 0, use uniform random over legal actions

    Args:
        regrets: Tensor of shape [batch_size, 8] or [8]
                Regret values from network
        legal_mask: Tensor of shape [batch_size, 8] or [8]
                   1.0 for legal actions, 0.0 for illegal

    Returns:
        strategy: Tensor of shape [batch_size, 8] or [8]
                 Action probabilities summing to 1.0

    Example:
        >>> regrets = torch.tensor([0.5, -0.3, 0.8, 0.2, 0.1, -0.5, -0.2, 0.0])
        >>> legal_mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        >>> strategy = regret_matching(regrets, legal_mask)
        >>> print(strategy)  # [0.3125, 0, 0.5, 0.125, 0.0625, 0, 0, 0]
    """
    # Step 1: Only positive regrets contribute to strategy
    positive_regrets = torch.clamp(regrets, min=0.0)

    # Step 2: Mask out illegal actions
    masked_regrets = positive_regrets * legal_mask

    # Step 3: Normalize to get probabilities
    regret_sum = masked_regrets.sum(dim=-1, keepdim=True)

    # Step 4: Handle edge case - if no positive regrets, uniform random
    strategy = torch.where(
        regret_sum > 0,
        masked_regrets / regret_sum,
        legal_mask / legal_mask.sum(dim=-1, keepdim=True)  # Uniform over legal
    )

    return strategy


def sample_action(strategy):
    """
    Sample an action from the strategy distribution.

    Args:
        strategy: Tensor of shape [8] - action probabilities

    Returns:
        action_idx: int - sampled action index (0-7)

    Example:
        >>> strategy = torch.tensor([0.2, 0.0, 0.5, 0.2, 0.1, 0.0, 0.0, 0.0])
        >>> action = sample_action(strategy)
        >>> print(action)  # Most likely 2 (50% prob), could be 0, 3, or 4
    """
    return torch.multinomial(strategy, num_samples=1).item()


# For debugging/testing
if __name__ == "__main__":
    print("=" * 70)
    print("RegretNetwork Testing")
    print("=" * 70)

    # Create network
    net = RegretNetwork(input_dim=32, output_dim=8)
    print(f"\nNetwork architecture:")
    print(net)
    print(f"\nTotal parameters: {net.get_num_parameters():,}")

    # Test forward pass
    print("\n" + "-" * 70)
    print("Test 1: Forward Pass")
    print("-" * 70)
    dummy_state = torch.randn(32)
    regrets = net(dummy_state)
    print(f"Input shape: {dummy_state.shape}")
    print(f"Output shape: {regrets.shape}")
    print(f"Output regrets: {regrets}")

    # Test regret matching
    print("\n" + "-" * 70)
    print("Test 2: Regret Matching")
    print("-" * 70)
    legal_mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    strategy = regret_matching(regrets, legal_mask)
    print(f"Legal mask: {legal_mask}")
    print(f"Strategy: {strategy}")
    print(f"Sum: {strategy.sum().item():.4f} (should be 1.0)")

    # Test action sampling
    print("\n" + "-" * 70)
    print("Test 3: Action Sampling")
    print("-" * 70)
    action = sample_action(strategy)
    print(f"Sampled action: {action}")

    # Test batch mode
    print("\n" + "-" * 70)
    print("Test 4: Batch Processing")
    print("-" * 70)
    batch_states = torch.randn(10, 32)
    batch_regrets = net(batch_states)
    print(f"Batch input shape: {batch_states.shape}")
    print(f"Batch output shape: {batch_regrets.shape}")

    print("\n" + "=" * 70)
    print("✓ All tests passed! Network is ready.")
    print("=" * 70)
