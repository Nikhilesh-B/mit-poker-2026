import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
from network.card_embedding import CardEmbedding
from utils.canon_cards import canon_cards


def normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize tensor to zero mean and unit variance (per sample).
    From Deep CFR paper: applied to last-layer features before action head.
    """
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + eps)


class DeepCFRModule(nn.Module):
    """
    Deep CFR Network following the paper's architecture:
    - Separate branches for cards and action history
    - Skip connections on layers with matching dimensions
    - Normalization before final output
    """
    
    def __init__(self, nhandcards: int, nboardcards: int, n_action_history: int, nresponses: int, dim=256) -> None:
        super(DeepCFRModule, self).__init__()
        
        self.dim = dim

        # Hand card embeddings and layers
        self.hand_embeddings = nn.ModuleList(
            [CardEmbedding(dim) for _ in range(nhandcards)])
        self.hand_layer1 = nn.Linear(dim*nhandcards, dim)
        self.hand_layer2 = nn.Linear(dim, dim)
        self.hand_layer3 = nn.Linear(dim, dim)

        # Board card embeddings and layers
        self.board_embeddings = nn.ModuleList(
            [CardEmbedding(dim) for _ in range(nboardcards)])
        self.board_layer1 = nn.Linear(dim*nboardcards, dim)
        self.board_layer2 = nn.Linear(dim, dim)
        self.board_layer3 = nn.Linear(dim, dim)

        # Action history layers (analogous to "bet branch" in paper)
        # Paper: bet1 and bet2 with skip on bet2
        self.actions_layer1 = nn.Linear(n_action_history*6, dim)
        self.actions_layer2 = nn.Linear(dim, dim)  # Skip connection here

        # Combined trunk layers (paper: comb1, comb2, comb3 with skips on comb2, comb3)
        self.comb_layer1 = nn.Linear(3*dim, dim)
        self.comb_layer2 = nn.Linear(dim, dim)  # Skip connection here
        self.comb_layer3 = nn.Linear(dim, dim)  # Skip connection here

        # Output head
        # nactions => discard0, discard1, discard2, check, call, fold, raise_small, raise_medium, raise_large
        self.action_head = nn.Linear(dim, nresponses)
        self.n_action_history = n_action_history
        
        # Initialize output layer to return 0 for all inputs (paper requirement)
        # "Initialize each player's advantage network...so that it returns 0 for all inputs"
        self._init_output_to_zero()
    
    def _init_output_to_zero(self):
        """
        Initialize the output layer so the network returns 0 for all inputs.
        
        From the paper: "Initialize each player's advantage network V(I,a|θp) 
        with parameters θp so that it returns 0 for all inputs."
        
        This is important because at iteration 0, all advantages should be 0,
        leading to uniform strategy via regret matching.
        """
        # Set output layer weights and bias to zero
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

    def _encode_action_history(self, action_history_str: str, device: torch.device = None) -> torch.Tensor:
        """
        Encode action history string into numerical features.

        Args:
            action_history_str: String representation of action history
            device: Device to create tensor on (defaults to CPU)

        Each action is one-hot encoded as 6 features:
        [is_check, is_call, is_fold, is_discard, is_raise_small, is_raise_medium]
        (raise_large is mapped to raise_medium to fit 6 features)
        """
        # Map action characters to indices
        action_map = {
            'X': 0,  # check
            'C': 1,  # call
            'F': 2,  # fold
            'D': 3,  # discard
            'r': 4,  # raise small
            'R': 5,  # raise medium
            'B': 5,  # raise large (map to same as medium)
        }

        # Convert string to list of characters
        all_actions = list(action_history_str)

        # Truncate or pad to n_action_history
        all_actions = all_actions[:self.n_action_history]
        while len(all_actions) < self.n_action_history:
            all_actions.append('')  # Empty action (all zeros)

        # One-hot encode each action as 6 features
        features = []
        for action_char in all_actions:
            one_hot = [0.0] * 6
            if action_char in action_map:
                idx = action_map[action_char]
                one_hot[idx] = 1.0
            features.extend(one_hot)

        # Convert to tensor and add batch dimension if needed
        return torch.tensor(features, dtype=torch.float32, device=device if device else torch.device('cpu'))

    @property
    def device(self) -> torch.device:
        """Get the device of the network parameters."""
        return next(self.parameters()).device

    def forward(self, canon_cards: List[canon_cards], action_history: List[str]):
        # Handle single sample vs batch
        if not isinstance(canon_cards, list):
            canon_cards = [canon_cards]
        if not isinstance(action_history, list):
            action_history = [action_history]

        batch_size = len(canon_cards)
        device = self.device  # Get device for tensor creation

        # Get canonical hand and board tensors for each sample
        canon_hands = [c.get_canonical_hand_tensor() for c in canon_cards]
        canon_boards = [c.get_canonical_board_tensor() for c in canon_cards]

        # Process hand cards
        # Each card position has its own embedding
        hand_embeds_list = []
        for i in range(len(self.hand_embeddings)):
            # Get the i-th card from each sample in the batch
            # Pad with -1 if card doesn't exist
            card_tensors = []
            for hand_tensor in canon_hands:
                if i < len(hand_tensor):
                    card_tensors.append(hand_tensor[i].item())
                else:
                    card_tensors.append(-1)

            # Create batch tensor on the correct device: [batch_size, 1]
            card_batch = torch.tensor(
                card_tensors, dtype=torch.long, device=device).unsqueeze(1)

            # Embed through the i-th embedding
            card_embed = self.hand_embeddings[i](
                card_batch)  # [batch_size, dim]
            hand_embeds_list.append(card_embed)

        # Concatenate all hand card embeddings
        # [batch_size, dim * nhandcards]
        hand_embeds = torch.cat(hand_embeds_list, dim=1)

        # Pass through hand layers (NO skip connections - following paper)
        hand_feat = F.relu(self.hand_layer1(hand_embeds))
        hand_feat = F.relu(self.hand_layer2(hand_feat))
        hand_feat = F.relu(self.hand_layer3(hand_feat))

        # Process board cards (same as hand)
        board_embeds_list = []
        for i in range(len(self.board_embeddings)):
            card_tensors = []
            for board_tensor in canon_boards:
                if i < len(board_tensor):
                    card_tensors.append(board_tensor[i].item())
                else:
                    card_tensors.append(-1)

            card_batch = torch.tensor(
                card_tensors, dtype=torch.long, device=device).unsqueeze(1)
            card_embed = self.board_embeddings[i](
                card_batch)  # [batch_size, dim]
            board_embeds_list.append(card_embed)

        # Concatenate all board card embeddings
        # [batch_size, dim * nboardcards]
        board_embeds = torch.cat(board_embeds_list, dim=1)

        # Pass through board layers (NO skip connections - following paper)
        board_feat = F.relu(self.board_layer1(board_embeds))
        board_feat = F.relu(self.board_layer2(board_feat))
        board_feat = F.relu(self.board_layer3(board_feat))

        # Encode action history
        # For batch processing, we need to handle each sample's history
        action_features_list = []
        for hist in action_history:
            action_feat = self._encode_action_history(hist, device)  # [n_action_history * 6]
            action_features_list.append(action_feat)

        # Stack into batch tensor
        # [batch_size, n_action_history * 6]
        action_features = torch.stack(action_features_list, dim=0)

        # Pass through action layers (paper: 2 layers with skip on layer 2)
        action_feat = F.relu(self.actions_layer1(action_features))
        action_feat = F.relu(self.actions_layer2(action_feat) + action_feat)  # Skip connection

        # Combine hand, board, and action features
        # [batch_size, 3*dim]
        combined = torch.cat([hand_feat, board_feat, action_feat], dim=1)

        # Pass through combination layers (with skip connections on layers 2 and 3)
        combined_feat = F.relu(self.comb_layer1(combined))
        combined_feat = F.relu(self.comb_layer2(combined_feat) + combined_feat)  # Skip connection
        combined_feat = F.relu(self.comb_layer3(combined_feat) + combined_feat)  # Skip connection
        
        # Normalize to zero mean and unit variance (from paper)
        combined_feat = normalize(combined_feat)

        # Final output (raw regrets/logits, no activation)
        output = self.action_head(combined_feat)  # [batch_size, nresponses]

        return output
