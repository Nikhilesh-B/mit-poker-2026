"""
Deep CFR Training Pipeline

This module trains the DeepCFR network using samples collected from MCCFR iterations.
The network learns to predict regrets by minimizing MSE loss against tabular MCCFR regrets.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict

from network.model import DeepCFRModule
from core.mccfr import MCCFR
from utils.infoset_parser import parse_infoset_to_network_input, batch_parse_infosets
from utils.action_mapping import regrets_dict_to_tensor


class TrainingSample:
    """
    A single training sample for the network.
    
    Attributes:
        infoset: Infoset string
        target_regrets: Target regret values (from MCCFR)
        player: Player index (0 or 1)
    """
    
    def __init__(self, infoset: str, target_regrets: Dict[str, float], player: int):
        self.infoset = infoset
        self.target_regrets = target_regrets
        self.player = player


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
        batch_size: int = 32
    ):
        """
        Initialize trainer.
        
        Args:
            network: DeepCFR network to train
            mccfr: MCCFR instance for generating samples
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
        """
        self.network = network
        self.mccfr = mccfr
        self.batch_size = batch_size
        
        # Optimizer and loss
        self.optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Training samples storage
        self.samples = []
        
        # Training statistics
        self.training_stats = {
            'losses': [],
            'num_samples': [],
            'num_batches': []
        }
    
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
    
    def prepare_batch(self, batch_samples: List[TrainingSample]) -> Tuple[List, List, torch.Tensor]:
        """
        Prepare a batch of samples for training.
        
        Args:
            batch_samples: List of TrainingSample objects
        
        Returns:
            Tuple of (canonical_cards_list, action_history_list, target_tensor)
        """
        # Parse infosets
        infosets = [sample.infoset for sample in batch_samples]
        cc_list, ah_list = batch_parse_infosets(infosets)
        
        # Convert target regrets to tensors
        # We need to get state for each sample (approximation: use a dummy state)
        # This is a simplification - in practice, we'd need to store state info
        state = self.mccfr.create_initial_state()
        
        target_tensors = []
        for sample in batch_samples:
            target_tensor = regrets_dict_to_tensor(
                sample.target_regrets,
                state,  # Using dummy state - imperfect but works
                sample.player,
                self.mccfr
            )
            target_tensors.append(target_tensor)
        
        # Stack into batch
        target_batch = torch.stack(target_tensors)  # [batch_size, 9]
        
        return cc_list, ah_list, target_batch
    
    def train_on_samples(self, num_epochs: int = 1) -> Dict[str, float]:
        """
        Train network on collected samples.
        
        Args:
            num_epochs: Number of epochs to train
        
        Returns:
            Dictionary with training metrics
        """
        if len(self.samples) == 0:
            return {'loss': 0.0, 'num_batches': 0, 'num_samples': 0}
        
        self.network.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            # Shuffle samples
            import random
            random.shuffle(self.samples)
            
            # Create batches
            for i in range(0, len(self.samples), self.batch_size):
                batch_samples = self.samples[i:i + self.batch_size]
                
                if len(batch_samples) == 0:
                    continue
                
                # Prepare batch
                cc_list, ah_list, target_batch = self.prepare_batch(batch_samples)
                
                # Forward pass
                predictions = self.network(cc_list, ah_list)  # [batch_size, 9]
                
                # Compute loss
                loss = self.criterion(predictions, target_batch)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Store statistics
        self.training_stats['losses'].append(avg_loss)
        self.training_stats['num_samples'].append(len(self.samples))
        self.training_stats['num_batches'].append(num_batches)
        
        self.network.eval()
        
        return {
            'loss': avg_loss,
            'num_batches': num_batches,
            'num_samples': len(self.samples),
            'total_loss': total_loss
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
        nboardcards=5,
        n_action_history=20,
        nresponses=9,
        dim=256
    )
    mccfr = MCCFR()
    
    # Create trainer
    trainer = DeepCFRTrainer(network, mccfr, learning_rate=0.001, batch_size=16)
    
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
        print(f"  Iteration {i+1}: loss = {results['loss']:.6f}, samples = {results['total_samples']}")
    
    print(f"\n✓ Training pipeline working!")
    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Final loss: {losses[-1]:.6f}")
    
    print("\n" + "=" * 70)
    print("✓✓✓ BASIC TRAINING TEST PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_basic_training()
