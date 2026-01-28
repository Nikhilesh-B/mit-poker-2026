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

        # Board card embedding - SINGLE shared embedding for permutation invariance
        # All board cards use the same embedding and are SUMMED (order doesn't matter)
        self.board_embedding = CardEmbedding(dim)
        self.nboardcards = nboardcards  # Store for forward pass
        # Input is summed embedding (dim), not concat
        self.board_layer1 = nn.Linear(dim, dim)
        self.board_layer2 = nn.Linear(dim, dim)
        self.board_layer3 = nn.Linear(dim, dim)

        # Action history layers (analogous to "bet branch" in paper)
        # Paper: bet1 and bet2 with skip on bet2
        # 17 features per action: check, call, fold, discard, + 13 raise sizes (25%-500% + all-in)
        self.actions_layer1 = nn.Linear(n_action_history*17, dim)
        self.actions_layer2 = nn.Linear(dim, dim)  # Skip connection here

        # Combined trunk layers (paper: comb1, comb2, comb3 with skips on comb2, comb3)
        self.comb_layer1 = nn.Linear(3*dim, dim)
        self.comb_layer2 = nn.Linear(dim, dim)  # Skip connection here
        self.comb_layer3 = nn.Linear(dim, dim)  # Skip connection here

        # Output head
        # nactions => discard0, discard1, discard2, check, call, fold,
        #             raise_tiny (<15), raise_small (15-50), raise_medium (50-125),
        #             raise_large (125-250), raise_all_in (250+)
        #             (11 total with absolute raise buckets)
        self.action_head = nn.Linear(dim, nresponses)
        self.n_action_history = n_action_history

        # Initialize output layer to return 0 for all inputs (paper requirement)
        # "Initialize each player's advantage network...so that it returns 0 for all inputs"
        self._init_output_to_zero()

    def _init_output_to_zero(self):
        """
        Initialize the output layer so outputs are near-zero at start.

        From the paper: "Initialize each player's advantage network V(I,a|θp) 
        with parameters θp so that it returns 0 for all inputs."

        IMPORTANT: The paper also says "trained from scratch each CFR iteration, 
        starting from a random initialization" (Section 5.2). Setting weights to
        exactly zero BLOCKS GRADIENT FLOW to earlier layers!

        Solution: Use very small random weights (Xavier with small gain) so:
        1. Initial outputs are near-zero (giving ~uniform strategy via regret matching)
        2. Gradients can still flow back through the network
        """
        # Small random weights - allows gradients to flow while keeping outputs near zero
        nn.init.xavier_uniform_(self.action_head.weight, gain=0.01)
        # Zero bias - centers outputs around zero
        nn.init.zeros_(self.action_head.bias)

    def _encode_action_history(self, action_history_str: str, device: torch.device = None) -> torch.Tensor:
        """
        Encode action history string into numerical features.

        Args:
            action_history_str: String representation of action history
            device: Device to create tensor on (defaults to CPU)

        Each action is one-hot encoded as 17 features:
        [check, call, fold, discard, raise_25, raise_50, raise_75, raise_100, raise_150,
         raise_200, raise_250, raise_300, raise_350, raise_400, raise_450, raise_500, raise_all_in]

        Raise characters: '1'-'9' for 25%-350%, 'T'=400%, 'E'=450%, 'W'=500%, 'Z'=all-in
        """
        # Map action characters to indices (17 features total: 4 base + 13 raises)
        action_map = {
            'X': 0,   # check
            'C': 1,   # call
            'F': 2,   # fold
            'D': 3,   # discard
            '1': 4,   # raise 25% pot
            '2': 5,   # raise 50% pot
            '3': 6,   # raise 75% pot
            '4': 7,   # raise 100% pot
            '5': 8,   # raise 150% pot
            '6': 9,   # raise 200% pot
            '7': 10,  # raise 250% pot
            '8': 11,  # raise 300% pot
            '9': 12,  # raise 350% pot
            'T': 13,  # raise 400% pot
            'E': 14,  # raise 450% pot
            'W': 15,  # raise 500% pot
            'Z': 16,  # raise all-in
        }

        # Convert string to list of characters
        all_actions = list(action_history_str)

        # Truncate or pad to n_action_history
        all_actions = all_actions[:self.n_action_history]
        while len(all_actions) < self.n_action_history:
            all_actions.append('')  # Empty action (all zeros)

        # One-hot encode each action as 17 features
        features = []
        for action_char in all_actions:
            one_hot = [0.0] * 17
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
        assert (batch_size == len(action_history))
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

        # Process board cards with PERMUTATION INVARIANCE
        # All board cards use the same embedding and are SUMMED
        # This means [card1, card2, card3] == [card3, card1, card2] (order doesn't matter)
        board_embeds_sum = torch.zeros(batch_size, self.dim, device=device)

        for i in range(self.nboardcards):
            card_tensors = []
            for board_tensor in canon_boards:
                if i < len(board_tensor):
                    card_tensors.append(board_tensor[i].item())
                else:
                    card_tensors.append(-1)  # No card at this position

            card_batch = torch.tensor(
                card_tensors, dtype=torch.long, device=device).unsqueeze(1)

            # Use the SAME embedding for all board card positions
            card_embed = self.board_embedding(card_batch)  # [batch_size, dim]

            # SUM instead of concatenate (permutation invariance)
            board_embeds_sum = board_embeds_sum + card_embed

        # Pass through board layers (NO skip connections - following paper)
        # Input is now [batch_size, dim] instead of [batch_size, dim*nboardcards]
        board_feat = F.relu(self.board_layer1(board_embeds_sum))
        board_feat = F.relu(self.board_layer2(board_feat))
        board_feat = F.relu(self.board_layer3(board_feat))

        # Encode action history
        # For batch processing, we need to handle each sample's history
        action_features_list = []
        for hist in action_history:
            action_feat = self._encode_action_history(
                hist, device)  # [n_action_history * 7]
            action_features_list.append(action_feat)

        # Stack into batch tensor
        # [batch_size, n_action_history * 7]
        action_features = torch.stack(action_features_list, dim=0)

        # Pass through action layers (paper: 2 layers with skip on layer 2)
        action_feat = F.relu(self.actions_layer1(action_features))
        action_feat = F.relu(self.actions_layer2(
            action_feat) + action_feat)  # Skip connection

        # Combine hand, board, and action features
        # [batch_size, 3*dim]
        combined = torch.cat([hand_feat, board_feat, action_feat], dim=1)

        # Pass through combination layers (with skip connections on layers 2 and 3)
        combined_feat = F.relu(self.comb_layer1(combined))
        combined_feat = F.relu(self.comb_layer2(
            combined_feat) + combined_feat)  # Skip connection
        combined_feat = F.relu(self.comb_layer3(
            combined_feat) + combined_feat)  # Skip connection

        # Normalize to zero mean and unit variance (from paper)
        combined_feat = normalize(combined_feat)

        # Final output (raw regrets/logits, no activation)
        output = self.action_head(combined_feat)  # [batch_size, nresponses]

        return output
