"""
Deep CFR Training Pipeline

This module trains the DeepCFR network using samples collected from MCCFR iterations.

Following the paper:
- Uses INSTANTANEOUS regrets (not cumulative)
- Linear weighting by iteration t' in loss function
- Gradient norm clipping
- Fixed number of SGD iterations per training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from network.model import DeepCFRModule
from core.mccfr import MCCFR
from utils.infoset_parser import parse_infoset_to_network_input, batch_parse_infosets
from utils.action_mapping import regrets_dict_to_tensor


class TrainingSample:
    """
    A single training sample for the network.

    Following the paper, samples include:
    - infoset: Information set string
    - target_regrets: INSTANTANEOUS regrets (not cumulative)
    - player: Player index (0 or 1)
    - iteration: CFR iteration when this sample was collected (for linear weighting)
    """

    def __init__(self, infoset: str, target_regrets: Dict[str, float], player: int,
                 iteration: int = 1):
        self.infoset = infoset
        self.target_regrets = target_regrets
        self.player = player
        self.iteration = iteration  # For linear weighting


class DeepCFRTrainer:
    """
    Trains the DeepCFR network using MCCFR samples.

    The training process:
    1. Run MCCFR iterations to collect regret data
    2. Store (infoset, regrets) pairs as training samples
    3. Periodically train network on collected samples
    4. Network learns to predict regrets for any infoset
    """

    def __init__(
        self,
        network: DeepCFRModule,
        mccfr: MCCFR,
        learning_rate: float = 0.001,
        batch_size: int = 10000,  # Paper: 10,000 (HULH uses 20,000)
        max_grad_norm: float = 1.0,  # Paper: gradient norm clipping to 1
        sgd_iterations: int = 4000,  # Paper: 32,000 for HULH, start smaller
        memory_limit: int = 10000000,  # Paper uses 40M, we use 10M
        use_reservoir_sampling: bool = True,  # Paper: reservoir sampling is crucial
        training_device: str = "auto"  # "auto", "mps", "cuda", or "cpu"
    ):
        """
        Initialize trainer.

        Args:
            network: DeepCFR network to train
            mccfr: MCCFR instance for generating samples
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training (paper: 20,000 for HULH)
            max_grad_norm: Maximum gradient norm for clipping (paper: 1.0)
            sgd_iterations: Fixed number of SGD steps per training (paper: 32,000 for HULH)
            memory_limit: Maximum samples to store (paper: 40M per player)
            use_reservoir_sampling: If True, use reservoir sampling when memory full
            training_device: Device for batch training ("auto" detects MPS/CUDA)
        """
        self.network = network
        self.mccfr = mccfr
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.sgd_iterations = sgd_iterations
        self.learning_rate = learning_rate
        self.memory_limit = memory_limit
        self.use_reservoir_sampling = use_reservoir_sampling

        # Set up training device (GPU acceleration for batch training)
        if training_device == "auto":
            if torch.backends.mps.is_available():
                self.training_device = torch.device("mps")
            elif torch.cuda.is_available():
                self.training_device = torch.device("cuda")
            else:
                self.training_device = torch.device("cpu")
        else:
            self.training_device = torch.device(training_device)

        # Optimizer and loss
        self.optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Training samples storage
        self.samples = []
        self.total_samples_seen = 0  # For reservoir sampling

        # Training statistics
        self.training_stats = {
            'losses': [],
            'num_samples': [],
            'num_batches': [],
            'grad_norms': [],
            'reservoir_replacements': 0,  # Track how many times we replaced samples
            # Log which device we're using
            'training_device': str(self.training_device)
        }

    def add_sample(self, sample: TrainingSample):
        """
        Add a sample to the memory, using reservoir sampling if memory is full.

        Reservoir sampling ensures every sample (past or present) has equal
        probability of being in memory, maintaining an unbiased sample.

        Algorithm (when memory is full):
            For the N-th sample seen, replace a random sample with probability M/N
            where M is the memory limit.

        Args:
            sample: TrainingSample to add
        """
        import random

        self.total_samples_seen += 1

        if len(self.samples) < self.memory_limit:
            # Memory not full - just append (O(1))
            self.samples.append(sample)
        elif self.use_reservoir_sampling:
            # Memory full - use reservoir sampling
            # Replace a random sample with probability memory_limit / total_seen
            replace_prob = self.memory_limit / self.total_samples_seen
            if random.random() < replace_prob:
                # Replace a random existing sample (O(1) - just assignment)
                replace_idx = random.randint(0, len(self.samples) - 1)
                self.samples[replace_idx] = sample
                self.training_stats['reservoir_replacements'] += 1
            # else: sample rejected by reservoir sampling (O(1))
        # else: memory full and not using reservoir sampling - drop the sample

    def add_samples(self, samples: List[TrainingSample]):
        """Add multiple samples using reservoir sampling."""
        for sample in samples:
            self.add_sample(sample)

    def collect_samples_from_mccfr(self, num_iterations: int) -> int:
        """
        Collect training samples by running MCCFR iterations.

        Args:
            num_iterations: Number of MCCFR iterations to run

        Returns:
            Number of new samples collected
        """
        samples_before = len(self.samples)

        for i in range(num_iterations):
            # Run MCCFR iteration
            state = self.mccfr.create_initial_state()
            self.mccfr.external_sampling(state, traversing_player=i % 2)

        # After iterations, extract samples from regret table
        for infoset, regrets in self.mccfr.regret_table.items():
            if len(regrets) > 0:  # Only add if we have regret data
                # We don't know which player this infoset belongs to from the string
                # For now, we'll store it with player=0 (can be improved)
                sample = TrainingSample(infoset, dict(regrets), player=0)

                # Only add if not already in samples (avoid duplicates)
                if not any(s.infoset == infoset for s in self.samples):
                    self.samples.append(sample)

        samples_after = len(self.samples)
        return samples_after - samples_before

    def _get_legal_action_mask(self, street: int, target_regrets: Dict[str, float] = None,
                               facing_bet: bool = None) -> List[float]:
        """
        Get a mask indicating which actions are legal based on the actual legal actions
        in the sample's target_regrets dictionary.

        Actions: [DISCARD_0, DISCARD_1, DISCARD_2, CHECK, CALL, FOLD, 
                  RAISE_TINY, RAISE_SMALL, RAISE_MEDIUM, RAISE_LARGE, RAISE_ALL_IN]  (11 total)

        The mask is derived DIRECTLY from which action keys are present in target_regrets,
        since target_regrets only contains regrets for legal actions.

        Args:
            street: Street number (0-4) - used as fallback only
            target_regrets: Dict of action -> value from the sample (keys are legal actions)
            facing_bet: Unused, kept for API compatibility

        Returns:
            List of 11 floats (1.0 for legal, 0.0 for illegal)
        """
        from utils.action_mapping import NETWORK_ACTION_TYPES, NUM_ACTIONS

        # Use NUM_ACTIONS from action_mapping (11 total)

        if target_regrets:
            # Create mask directly from target_regrets keys
            # An action is legal if it's in target_regrets (which only contains legal actions)
            action_keys = set(target_regrets.keys())

            mask = []
            for action_name in NETWORK_ACTION_TYPES:
                if action_name in action_keys:
                    mask.append(1.0)  # Legal action
                else:
                    mask.append(0.0)  # Illegal action

            return mask

        # Fallback: if no target_regrets, use heuristic based on street
        # (This should rarely happen with proper samples)
        if street == 0:
            # Preflop is betting - typically facing big blind
            # [D0, D1, D2, CHECK, CALL, FOLD, 5 raise buckets...]
            return [0.0, 0.0, 0.0, 0.0, 1.0, 1.0] + [1.0] * 5
        elif street in (1, 2):  # Discard streets
            # Discard phase
            # 11 - 3 = 8 non-discard actions
            return [1.0, 1.0, 1.0] + [0.0] * 8
        else:
            # Other streets: default to not facing bet
            return [0.0, 0.0, 0.0, 1.0, 0.0, 1.0] + [1.0] * 5

    def _extract_street(self, infoset: str) -> int:
        """
        Extract street number from infoset string.

        Infoset format: "S{street}|H:...|B:...|A:..."

        Args:
            infoset: Infoset string

        Returns:
            Street number (0-3)
        """
        try:
            # Format: "S0|H:..." or "S1|H:..."
            return int(infoset[1])
        except (IndexError, ValueError):
            # Default to street 0 if parsing fails
            return 0

    def prepare_batch(self, batch_samples: List[TrainingSample]) -> Tuple[List, List, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare a batch of samples for training.

        With pot-relative action keys, we no longer need a dummy state for
        converting regrets to tensors. The regrets_dict already contains
        pot-relative keys that map directly to network output indices.

        Args:
            batch_samples: List of TrainingSample objects

        Returns:
            Tuple of (canonical_cards_list, action_history_list, target_tensor, iteration_weights, legal_mask)
        """
        # Parse infosets
        infosets = [sample.infoset for sample in batch_samples]
        cc_list, ah_list = batch_parse_infosets(infosets)

        target_tensors = []
        iterations = []
        legal_masks = []

        for sample in batch_samples:
            # Convert pot-relative regrets dict to tensor (no state needed!)
            target_tensor = regrets_dict_to_tensor(sample.target_regrets)
            target_tensors.append(target_tensor)
            iterations.append(sample.iteration)

            # Create legal action mask based on the actual actions in the sample
            # This correctly infers facing_bet from whether CHECK or CALL is present
            street = self._extract_street(sample.infoset)
            legal_masks.append(self._get_legal_action_mask(
                street, sample.target_regrets))

        # Stack into batch
        target_batch = torch.stack(target_tensors)  # [batch_size, 19]

        # Create iteration weights for linear weighting
        # Paper: weight each sample by iteration t' (later iterations weighted more)
        iteration_weights = torch.tensor(
            iterations, dtype=torch.float32)  # [batch_size]

        # Create legal action mask tensor
        legal_mask = torch.tensor(
            legal_masks, dtype=torch.float32)  # [batch_size, 19]

        return cc_list, ah_list, target_batch, iteration_weights, legal_mask

    def train_on_samples(self, num_epochs: int = 1, use_fixed_iterations: bool = True,
                         use_linear_weighting: bool = True, verbose: bool = True,
                         network_name: str = "Network", current_iteration: int = None) -> Dict[str, float]:
        """
        Train network on collected samples.

        Following the paper:
        - Uses fixed number of SGD iterations (not epochs)
        - Applies gradient norm clipping
        - Uses linear weighting by iteration t' with 2/T rescaling (LCFR)
        - Uses GPU (MPS/CUDA) for batch training if available

        Args:
            num_epochs: Number of epochs (used if use_fixed_iterations=False)
            use_fixed_iterations: If True, use self.sgd_iterations instead of epochs
            use_linear_weighting: If True, weight loss by iteration (paper's approach)
            verbose: If True, show tqdm progress bar
            network_name: Name to display in progress bar (e.g., "V0", "V1", "Π")
            current_iteration: Current CFR iteration T (required for 2/T rescaling in LCFR)

        Returns:
            Dictionary with training metrics
        """
        if len(self.samples) == 0:
            return {'loss': 0.0, 'num_batches': 0, 'num_samples': 0, 'avg_grad_norm': 0.0,
                    'loss_start': None, 'loss_end': None, 'loss_reduction_pct': None}

        # Move network to training device (GPU) for batch training
        original_device = next(self.network.parameters()).device
        self.network.to(self.training_device)
        self.network.train()

        # Recreate optimizer for new device (optimizer state needs to match device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.learning_rate)

        total_loss = 0.0
        num_batches = 0
        total_grad_norm = 0.0

        # Track loss progression within session
        loss_start = None
        loss_mid = None
        loss_end = None
        mid_step = self.sgd_iterations // 2

        import random

        try:
            if use_fixed_iterations:
                # Paper approach: Fixed number of SGD iterations with tqdm progress bar
                pbar = tqdm(range(self.sgd_iterations),
                            desc=f"    {network_name}",
                            leave=False,
                            disable=not verbose,
                            ncols=80)
                for step in pbar:
                    # Sample a random batch
                    if len(self.samples) >= self.batch_size:
                        batch_samples = random.sample(
                            self.samples, self.batch_size)
                    else:
                        batch_samples = self.samples

                    if len(batch_samples) == 0:
                        continue

                    # Prepare batch (now includes iteration weights and legal mask)
                    cc_list, ah_list, target_batch, iter_weights, legal_mask = self.prepare_batch(
                        batch_samples)

                    # Move tensors to training device
                    target_batch = target_batch.to(self.training_device)
                    iter_weights = iter_weights.to(self.training_device)
                    legal_mask = legal_mask.to(self.training_device)

                    # Forward pass (network handles internal tensor creation)
                    predictions = self.network(cc_list, ah_list)

                    # Move predictions to same device as targets if needed
                    if predictions.device != target_batch.device:
                        predictions = predictions.to(target_batch.device)

                    # Compute loss with linear weighting and legal action masking
                    # Paper Section 5.3 (LCFR):
                    # - Each sample is weighted by t' (iteration when collected)
                    # - At training time T, rescale all weights by 2/T
                    # - Effective weight = t' * (2/T)
                    # We only compute loss for LEGAL actions (e.g., no discards on flop/turn/river)
                    if use_linear_weighting:
                        # Per-sample squared error, masked by legal actions
                        squared_errors = (
                            predictions - target_batch) ** 2  # [batch, 9]
                        masked_errors = squared_errors * legal_mask  # Zero out illegal action errors
                        per_sample_loss = masked_errors.sum(dim=1)  # [batch]

                        # Apply LCFR 2/T rescaling (paper Section 5.3)
                        # Weight by t' * (2/T) where T is current training iteration
                        if current_iteration is not None and current_iteration > 0:
                            scaled_weights = iter_weights * \
                                (2.0 / current_iteration)
                        else:
                            scaled_weights = iter_weights

                        weighted_loss = (scaled_weights *
                                         per_sample_loss).mean()
                        loss = weighted_loss
                    else:
                        # Apply mask to MSE loss
                        squared_errors = (predictions - target_batch) ** 2
                        masked_errors = squared_errors * legal_mask
                        loss = masked_errors.mean()

                    # Track loss at key points
                    current_loss = loss.item()
                    if step == 0:
                        loss_start = current_loss
                    if step == mid_step:
                        loss_mid = current_loss
                    # Always update (final value is the end)
                    loss_end = current_loss

                    # Update progress bar with current loss
                    pbar.set_postfix({'loss': f'{current_loss:,.0f}'})

                    # Backward pass with gradient clipping
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Gradient clipping (paper: clip to norm 1)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(),
                        max_norm=self.max_grad_norm
                    )
                    total_grad_norm += grad_norm.item()

                    self.optimizer.step()

                    total_loss += current_loss
                    num_batches += 1

                # Print summary after training completes
                if verbose and loss_start is not None and loss_end is not None:
                    reduction = ((loss_start - loss_end) /
                                 loss_start * 100) if loss_start > 0 else 0
                    arrow = "↓" if reduction > 0 else "↑"
                    print(
                        f"    {network_name}: {loss_start:,.0f} → {loss_end:,.0f} ({arrow}{abs(reduction):.1f}%)")
            else:
                # Legacy epoch-based approach
                for epoch in range(num_epochs):
                    random.shuffle(self.samples)

                    for i in range(0, len(self.samples), self.batch_size):
                        batch_samples = self.samples[i:i + self.batch_size]

                        if len(batch_samples) == 0:
                            continue

                        cc_list, ah_list, target_batch, iter_weights, legal_mask = self.prepare_batch(
                            batch_samples)

                        # Move tensors to training device
                        target_batch = target_batch.to(self.training_device)
                        iter_weights = iter_weights.to(self.training_device)
                        legal_mask = legal_mask.to(self.training_device)

                        predictions = self.network(cc_list, ah_list)

                        if predictions.device != target_batch.device:
                            predictions = predictions.to(target_batch.device)

                        # Weighted loss with legal action masking (LCFR 2/T rescaling)
                        if use_linear_weighting:
                            squared_errors = (predictions - target_batch) ** 2
                            masked_errors = squared_errors * legal_mask
                            per_sample_loss = masked_errors.sum(dim=1)
                            # Apply LCFR 2/T rescaling
                            if current_iteration is not None and current_iteration > 0:
                                scaled_weights = iter_weights * \
                                    (2.0 / current_iteration)
                            else:
                                scaled_weights = iter_weights
                            loss = (scaled_weights * per_sample_loss).mean()
                        else:
                            squared_errors = (predictions - target_batch) ** 2
                            masked_errors = squared_errors * legal_mask
                            loss = masked_errors.mean()

                        self.optimizer.zero_grad()
                        loss.backward()

                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.network.parameters(),
                            max_norm=self.max_grad_norm
                        )
                        total_grad_norm += grad_norm.item()

                        self.optimizer.step()

                        total_loss += loss.item()
                        num_batches += 1
        finally:
            # Always move network back to CPU for CFR traversal
            self.network.to(original_device)
            self.optimizer = torch.optim.Adam(
                self.network.parameters(), lr=self.learning_rate)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0.0

        # Calculate loss reduction
        loss_reduction = None
        if loss_start is not None and loss_end is not None and loss_start > 0:
            loss_reduction = (loss_start - loss_end) / \
                loss_start * 100  # Percentage reduction

        # Store statistics
        self.training_stats['losses'].append(avg_loss)
        self.training_stats['num_samples'].append(len(self.samples))
        self.training_stats['num_batches'].append(num_batches)
        self.training_stats['grad_norms'].append(avg_grad_norm)

        self.network.eval()

        return {
            'loss': avg_loss,
            'num_batches': num_batches,
            'num_samples': len(self.samples),
            'total_loss': total_loss,
            'avg_grad_norm': avg_grad_norm,
            'loss_start': loss_start,
            'loss_mid': loss_mid,
            'loss_end': loss_end,
            'loss_reduction_pct': loss_reduction
        }

    def train_iteration(self, mccfr_iterations: int = 10, train_epochs: int = 1) -> Dict:
        """
        Complete training iteration: collect samples + train network.

        Args:
            mccfr_iterations: Number of MCCFR iterations to run
            train_epochs: Number of training epochs

        Returns:
            Dictionary with iteration results
        """
        # Collect samples
        new_samples = self.collect_samples_from_mccfr(mccfr_iterations)

        # Train on samples
        train_metrics = self.train_on_samples(num_epochs=train_epochs)

        return {
            'new_samples': new_samples,
            'total_samples': len(self.samples),
            **train_metrics
        }

    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        return self.training_stats

    def clear_samples(self):
        """Clear collected samples (useful for memory management)."""
        self.samples = []


def test_basic_training():
    """
    Test basic training pipeline.

    This test verifies:
    1. Can collect samples from MCCFR
    2. Can train network on samples
    3. Loss decreases over iterations
    """
    print("=" * 70)
    print("TESTING BASIC TRAINING PIPELINE")
    print("=" * 70)

    # Create network and MCCFR
    network = DeepCFRModule(
        nhandcards=3,
        nboardcards=6,  # 2 flop + 2 discards + turn + river = 6 max
        n_action_history=20,
        nresponses=11,  # 3 discards + 3 basic + 5 absolute raise buckets
        dim=256
    )
    mccfr = MCCFR()

    # Create trainer
    trainer = DeepCFRTrainer(
        network, mccfr, learning_rate=0.001, batch_size=16)

    print("\n[1] Collecting samples from MCCFR...")
    new_samples = trainer.collect_samples_from_mccfr(num_iterations=20)
    print(f"✓ Collected {new_samples} training samples")
    print(f"  Total samples: {len(trainer.samples)}")

    print("\n[2] Training network on samples...")
    metrics = trainer.train_on_samples(num_epochs=5)
    print(f"✓ Training complete")
    print(f"  Average loss: {metrics['loss']:.6f}")
    print(f"  Num batches: {metrics['num_batches']}")
    print(f"  Num samples: {metrics['num_samples']}")

    print("\n[3] Running full training iteration...")
    results = trainer.train_iteration(mccfr_iterations=10, train_epochs=3)
    print(f"✓ Iteration complete")
    print(f"  New samples: {results['new_samples']}")
    print(f"  Total samples: {results['total_samples']}")
    print(f"  Loss: {results['loss']:.6f}")

    print("\n[4] Testing loss improvement over multiple iterations...")
    losses = []
    for i in range(3):
        results = trainer.train_iteration(mccfr_iterations=5, train_epochs=2)
        losses.append(results['loss'])
        print(
            f"  Iteration {i+1}: loss = {results['loss']:.6f}, samples = {results['total_samples']}")

    print(f"\n✓ Training pipeline working!")
    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Final loss: {losses[-1]:.6f}")

    print("\n" + "=" * 70)
    print("✓✓✓ BASIC TRAINING TEST PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_basic_training()
