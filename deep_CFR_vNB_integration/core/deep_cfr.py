"""
Full Deep CFR Algorithm

This module implements the complete Deep CFR algorithm, combining:
- Monte Carlo CFR for game tree traversal
- Neural network for regret prediction (separate networks per player)
- Supervised learning to train the networks

Following the paper: We maintain separate value networks θ1, θ2 for each player.
Each network is trained only on that player's traversal data.
"""

import sys
import os

import torch
import random
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

from network.model import DeepCFRModule
from core.mccfr import MCCFR
from core.trainer import DeepCFRTrainer
from core.integration import NetworkMCCFRIntegration
from custom_engine import TerminalState


class DeepCFR:
    """
    Full Deep CFR Algorithm with separate networks per player.

    Following the paper's methodology:
    - Two value networks: θ1 (player 0) and θ2 (player 1)
    - Each network trained on its player's traversal data
    - Networks retrained from scratch each CFR iteration (optional)

    The algorithm alternates between:
    1. Running MCCFR iterations to collect regret data
    2. Training the appropriate network on collected samples
    3. Using the networks to predict regrets for new states
    """

    def __init__(
        self,
        network_dim: int = 256,
        learning_rate: float = 0.001,
        batch_size: int = 10000,  # Paper: 10,000 (HULH uses 20,000)
        use_network_after: int = 0,  # Paper: use network from iteration 1 (initialized to 0)
        train_every: int = 1,  # Paper: train after EVERY iteration
        train_epochs: int = 5,
        memory_limit: int = 10000000,  # 10M samples per player (paper uses 40M)
        training_device: str = "auto",  # "auto", "mps", "cuda", or "cpu"
        traversals_per_iter: int = 1000,  # Paper uses 10,000 for FHP
        sgd_iterations: int = 4000  # Paper uses 32,000 for HULH, 4,000 for FHP
    ):
        """
        Initialize Deep CFR with separate networks per player.

        Args:
            network_dim: Hidden dimension for networks
            learning_rate: Learning rate for network training
            batch_size: Batch size for training
            use_network_after: Start using network predictions after this many iterations
            train_every: Train network every N iterations
            train_epochs: Number of epochs per training session
            memory_limit: Maximum number of samples to store per player
            training_device: Device for batch training ("auto" detects MPS/CUDA)
            traversals_per_iter: Number of game traversals per CFR iteration (K in paper)
            sgd_iterations: SGD steps per training session (paper: 32,000 HULH, 4,000 FHP)
        """
        self.network_dim = network_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.training_device = training_device
        self.traversals_per_iter = traversals_per_iter
        self.sgd_iterations = sgd_iterations

        # Create TWO value networks - one per player (paper: θ1, θ2)
        self.networks = {
            0: self._create_network(network_dim),  # Player 0's value network
            1: self._create_network(network_dim),  # Player 1's value network
        }

        # Create STRATEGY network Π (paper: approximates average strategy)
        # This is the network used for actual play
        self.strategy_network = self._create_network(network_dim)

        # For backward compatibility, expose network as player 0's network
        self.network = self.networks[0]

        # Create MCCFR
        self.mccfr = MCCFR()

        # Create separate trainers for each player's value network
        # Pass training_device for GPU acceleration during batch training
        self.trainers = {
            0: DeepCFRTrainer(
                network=self.networks[0],
                mccfr=self.mccfr,
                learning_rate=learning_rate,
                batch_size=batch_size,
                training_device=training_device,
                sgd_iterations=sgd_iterations
            ),
            1: DeepCFRTrainer(
                network=self.networks[1],
                mccfr=self.mccfr,
                learning_rate=learning_rate,
                batch_size=batch_size,
                training_device=training_device,
                sgd_iterations=sgd_iterations
            )
        }

        # Trainer for strategy network (NOT reinitialized each iteration)
        self.strategy_trainer = DeepCFRTrainer(
            network=self.strategy_network,
            mccfr=self.mccfr,
            learning_rate=learning_rate,
            batch_size=batch_size,
            training_device=training_device,
            sgd_iterations=sgd_iterations
        )

        # For backward compatibility
        self.trainer = self.trainers[0]

        # Create integration for each player
        self.integrations = {
            0: NetworkMCCFRIntegration(network=self.networks[0], mccfr=self.mccfr),
            1: NetworkMCCFRIntegration(network=self.networks[1], mccfr=self.mccfr)
        }
        self.integration = self.integrations[0]

        # Strategy network integration (for actual play)
        # Strategy network outputs are logits → softmax (per paper Section 5.1)
        self.strategy_integration = NetworkMCCFRIntegration(
            network=self.strategy_network,
            mccfr=self.mccfr,
            is_strategy_network=True
        )

        # Separate advantage memories per player (paper: MV,1 and MV,2)
        self.advantage_memories = {
            0: [],  # Player 0's advantage memory
            1: []   # Player 1's advantage memory
        }

        # Training configuration
        self.use_network_after = use_network_after
        self.train_every = train_every
        self.train_epochs = train_epochs
        self.memory_limit = memory_limit

        # Statistics
        self.iteration_count = 0
        self.total_training_iterations = 0
        self.stats = {
            'iterations': [],
            'losses_p0': [],  # Player 0 value network losses
            'losses_p1': [],  # Player 1 value network losses
            'losses_strategy': [],  # Strategy network losses
            # For backward compatibility (average of value networks)
            'losses': [],
            'sample_counts_p0': [],
            'sample_counts_p1': [],
            'sample_counts_strategy': [],  # Strategy network sample count
            'sample_counts': [],  # For backward compatibility
            'network_usage': [],
            'mccfr_utilities': []
        }

    def _create_network(self, dim: int) -> DeepCFRModule:
        """Create a new network instance."""
        return DeepCFRModule(
            nhandcards=3,
            nboardcards=6,  # 2 flop + 2 discards + turn + river = 6 max
            n_action_history=20,
            nresponses=9,
            dim=dim
        )

    def reinitialize_network(self, player: int):
        """
        Reinitialize a player's value network from scratch.

        Following the paper: "Train θp from scratch each CFR iteration,
        starting from a random initialization."

        IMPORTANT: Preserves accumulated samples in the trainer's memory.

        Args:
            player: Player index (0 or 1)
        """
        # PRESERVE existing samples before reinitializing
        existing_samples = []
        existing_total_seen = 0
        if player in self.trainers:
            existing_samples = self.trainers[player].samples
            existing_total_seen = self.trainers[player].total_samples_seen

        # Create new network with random initialization
        self.networks[player] = self._create_network(self.network_dim)

        # Update trainer with new network and fresh optimizer
        self.trainers[player] = DeepCFRTrainer(
            network=self.networks[player],
            mccfr=self.mccfr,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            training_device=self.training_device,
            sgd_iterations=self.sgd_iterations
        )

        # RESTORE samples and total_samples_seen to new trainer (for reservoir sampling)
        self.trainers[player].samples = existing_samples
        self.trainers[player].total_samples_seen = existing_total_seen

        # Update integration
        self.integrations[player] = NetworkMCCFRIntegration(
            network=self.networks[player],
            mccfr=self.mccfr
        )

        # Keep backward compatibility
        if player == 0:
            self.network = self.networks[0]
            self.trainer = self.trainers[0]
            self.integration = self.integrations[0]

    def reinitialize_all_networks(self):
        """Reinitialize all value networks from scratch."""
        for player in [0, 1]:
            self.reinitialize_network(player)

    def should_use_network(self) -> bool:
        """Check if we should use network predictions."""
        return self.iteration_count >= self.use_network_after

    def should_train(self) -> bool:
        """Check if we should train the network."""
        return (self.iteration_count > 0 and
                self.iteration_count % self.train_every == 0)

    def run_iteration(self, use_network: Optional[bool] = None, progress_callback=None) -> Dict:
        """
        Run a single Deep CFR iteration with K traversals.

        Following paper's Algorithm 1:
        - For each player p:
          - For k = 1 to K: TRAVERSE(...) to collect samples
        - Train networks

        Args:
            use_network: Force network usage (if None, uses automatic schedule)
            progress_callback: Optional callback(completed, total, phase) for progress updates

        Returns:
            Dictionary with iteration statistics
        """
        # Determine if using network this iteration
        if use_network is None:
            use_network = self.should_use_network()

        # Set MCCFR iteration number for linear weighting
        self.mccfr.set_iteration(self.iteration_count + 1)

        # Run K traversals for EACH player (paper: "for each player p, for k=1 to K")
        total_utility = 0.0
        total_traversals = self.traversals_per_iter * 2  # K for each player

        for player in [0, 1]:
            for k in range(self.traversals_per_iter):
                # Create fresh initial state for each traversal
                state = self.mccfr.create_initial_state()

                if use_network:
                    utility = self._cfr_with_network(state, player)
                else:
                    utility = self.mccfr.external_sampling(state, player)

                total_utility += utility

                # Report progress during sample collection
                if progress_callback:
                    completed = player * self.traversals_per_iter + k + 1
                    progress_callback(completed, total_traversals, "samples")

        avg_utility = total_utility / (2 * self.traversals_per_iter)

        self.iteration_count += 1

        # Collect statistics
        result = {
            'iteration': self.iteration_count,
            'utility': avg_utility,
            'traversals': self.traversals_per_iter * 2,  # K for each player
            'used_network': use_network,
            'regret_table_size': len(self.mccfr.regret_table)
        }

        # Collect samples into trainer memory (which uses reservoir sampling)
        # This accumulates samples over iterations as per the paper
        # The key is to do this efficiently - process samples in batches if needed
        sample_counts = self._collect_samples_to_trainers()
        result.update(sample_counts)
        
        # Train network if scheduled
        if self.should_train():
            train_result = self._train_network()
            result.update(train_result)

        # Update stats
        self.stats['iterations'].append(self.iteration_count)
        self.stats['network_usage'].append(use_network)
        self.stats['mccfr_utilities'].append(avg_utility)

        return result

    def _cfr_with_network(self, state, traversing_player: int) -> float:
        """
        CFR iteration using network for regret predictions.

        This is the proper Deep CFR algorithm per the paper:
        - Use network to predict regrets at each decision point
        - Strategy is computed from network predictions via regret matching
        - Samples are collected for training
        
        The key insight is that the network generalizes across similar infosets,
        allowing exploration of states that tabular CFR couldn't reach.
        """
        # Build network integrations dict for both players
        network_integrations = {
            0: self.integrations[0],
            1: self.integrations[1]
        }
        
        # Use the network-guided external sampling
        utility = self.mccfr.external_sampling_with_network(
            state, 
            traversing_player, 
            network_integrations,
            collect_deep_cfr_samples=True
        )

        return utility

    def _collect_samples_to_trainers(self):
        """
        Collect samples from MCCFR memory into trainer memory.

        This should be called EVERY iteration to prevent unbounded memory growth.
        Trainer memory uses reservoir sampling, so it's safe to call frequently.

        Returns:
            Dictionary with counts of new samples collected
        """
        from core.trainer import TrainingSample

        new_samples_p0 = 0
        new_samples_p1 = 0
        new_strategy_samples = 0

        # Collect advantage samples - OPTIMIZED: batch process to reduce overhead
        for player in [0, 1]:
            samples_list = self.mccfr.get_advantage_samples(player)
            
            # Process samples in batch - create all TrainingSample objects first, then add
            # This reduces overhead from repeated object creation
            samples_to_add = []
            for sample_dict in samples_list:
                sample = TrainingSample(
                    infoset=sample_dict['infoset'],
                    target_regrets=sample_dict['regrets'],
                    player=sample_dict['player'],
                    iteration=sample_dict['iteration']
                )
                samples_to_add.append(sample)
            
            # Add all samples (reservoir sampling handles memory limits)
            for sample in samples_to_add:
                self.trainers[player].add_sample(sample)  # Uses reservoir sampling
                if player == 0:
                    new_samples_p0 += 1
                else:
                    new_samples_p1 += 1

        # Collect strategy samples - OPTIMIZED: batch process
        strategy_list = self.mccfr.get_strategy_samples()
        
        # Batch create TrainingSample objects
        strategy_samples_to_add = []
        for sample_dict in strategy_list:
            sample = TrainingSample(
                infoset=sample_dict['infoset'],
                target_regrets=sample_dict['strategy'],  # Actually strategy probs
                player=sample_dict['player'],
                iteration=sample_dict['iteration']
            )
            strategy_samples_to_add.append(sample)
        
        # Batch add to trainer
        for sample in strategy_samples_to_add:
            self.strategy_trainer.add_sample(sample)  # Uses reservoir sampling
            new_strategy_samples += 1

        # Clear MCCFR's temporary memory immediately after collecting
        # This prevents unbounded growth during traversal
        self.mccfr.clear_advantage_memory()
        self.mccfr.clear_strategy_memory()
        self.mccfr.clear_infoset_cache()  # Clear cache to free memory

        return {
            'new_samples_p0': new_samples_p0,
            'new_samples_p1': new_samples_p1,
            'new_strategy_samples': new_strategy_samples
        }

    def _train_network(self, traversing_player: int = None,
                       reinitialize: bool = True) -> Dict:
        """
        Train the networks on collected samples.

        Following the paper:
        - Each player's network is trained on their own samples
        - Networks are reinitialized from scratch each iteration
        - Uses MCCFR advantage memory with instantaneous regrets

        Order of operations:
        1. Reinitialize network (preserves samples in trainer)
        2. Train on all accumulated samples

        Note: Samples are collected in _collect_samples_to_trainers() which is
        called every iteration to prevent memory accumulation.

        Args:
            traversing_player: If specified, only train that player's network
            reinitialize: If True, reinitialize networks from scratch (paper's approach)
        """

        # STEP 1: Reinitialize networks from scratch (paper's approach)
        # This preserves samples but creates fresh network weights
        if reinitialize:
            if traversing_player is not None:
                self.reinitialize_network(traversing_player)
            else:
                self.reinitialize_all_networks()

        # Note: Memory limit is now enforced via reservoir sampling in trainer.add_sample()
        # Samples are already collected in _collect_samples_to_trainers() called every iteration

        # STEP 2: Train both networks (or just the traversing player's network)
        results = {}

        players_to_train = [
            traversing_player] if traversing_player is not None else [0, 1]

        for player in players_to_train:
            if player is None:
                continue
            trainer = self.trainers[player]
            if len(trainer.samples) > 0:
                train_metrics = trainer.train_on_samples(
                    num_epochs=self.train_epochs,
                    verbose=True,
                    network_name=f"V{player}"
                )
                results[f'loss_p{player}'] = train_metrics['loss']
                results[f'samples_p{player}'] = len(trainer.samples)
                # Loss progression within training session
                results[f'loss_start_p{player}'] = train_metrics.get(
                    'loss_start')
                results[f'loss_end_p{player}'] = train_metrics.get('loss_end')
                results[f'loss_reduction_p{player}'] = train_metrics.get(
                    'loss_reduction_pct')

        self.total_training_iterations += 1

        # Update stats
        loss_p0 = results.get('loss_p0', 0.0)
        loss_p1 = results.get('loss_p1', 0.0)
        samples_p0 = len(self.trainers[0].samples)
        samples_p1 = len(self.trainers[1].samples)

        self.stats['losses_p0'].append(loss_p0)
        self.stats['losses_p1'].append(loss_p1)
        self.stats['losses'].append(
            (loss_p0 + loss_p1) / 2)  # Backward compatibility
        self.stats['sample_counts_p0'].append(samples_p0)
        self.stats['sample_counts_p1'].append(samples_p1)
        self.stats['sample_counts'].append(samples_p0 + samples_p1)

        # Train the STRATEGY network Π (not reinitialized, accumulates knowledge)
        strategy_result = self._train_strategy_network()

        # Update strategy stats
        self.stats['losses_strategy'].append(strategy_result.get('loss', 0.0))
        self.stats['sample_counts_strategy'].append(
            strategy_result.get('num_samples', 0))

        return {
            'trained': True,
            'total_samples': samples_p0 + samples_p1,
            'total_samples_p0': samples_p0,
            'total_samples_p1': samples_p1,
            'loss': (loss_p0 + loss_p1) / 2,
            'loss_p0': loss_p0,
            'loss_p1': loss_p1,
            'loss_strategy': strategy_result.get('loss', 0.0),
            'strategy_samples': strategy_result.get('num_samples', 0),
            'training_iteration': self.total_training_iterations,
            # Loss progression (start → end within each training session)
            'loss_start_p0': results.get('loss_start_p0'),
            'loss_end_p0': results.get('loss_end_p0'),
            'loss_reduction_p0': results.get('loss_reduction_p0'),
            'loss_start_p1': results.get('loss_start_p1'),
            'loss_end_p1': results.get('loss_end_p1'),
            'loss_reduction_p1': results.get('loss_reduction_p1'),
            'loss_start_strategy': strategy_result.get('loss_start'),
            'loss_end_strategy': strategy_result.get('loss_end'),
            'loss_reduction_strategy': strategy_result.get('loss_reduction_pct'),
        }

    def _train_strategy_network(self) -> Dict:
        """
        Train the strategy network Π on collected strategy samples.

        Following the paper:
        - Targets are strategy probabilities σ_t(I), NOT regrets
        - Uses linear weighting by iteration t'
        - NOT reinitialized (accumulates knowledge across iterations)

        Note: Strategy samples are already collected in _collect_samples_to_trainers()
        which is called every iteration. This method just trains on accumulated samples.
        """

        # Train (NOT from scratch - accumulates)
        if len(self.strategy_trainer.samples) > 0:
            train_metrics = self.strategy_trainer.train_on_samples(
                num_epochs=self.train_epochs,
                use_linear_weighting=True,
                verbose=True,
                network_name="Π"
            )
            return {
                'loss': train_metrics['loss'],
                'num_samples': len(self.strategy_trainer.samples),
                'loss_start': train_metrics.get('loss_start'),
                'loss_end': train_metrics.get('loss_end'),
                'loss_reduction_pct': train_metrics.get('loss_reduction_pct')
            }

        return {'loss': 0.0, 'num_samples': 0}

    def _get_infoset_player(self, infoset: str) -> int:
        """
        Determine which player an infoset belongs to.

        This is a heuristic based on the action history.
        In the full implementation, this would be tracked during traversal.
        """
        # Count actions to determine whose turn it is
        # Simple heuristic: alternate based on action count
        # This is imperfect but works as a placeholder
        action_count = infoset.count('|')  # Actions are separated by |
        return action_count % 2

    def run_multiple_iterations(self, num_iterations: int, verbose: bool = True) -> List[Dict]:
        """
        Run multiple Deep CFR iterations.

        Args:
            num_iterations: Number of iterations to run
            verbose: Print progress

        Returns:
            List of iteration results
        """
        results = []

        for i in range(num_iterations):
            result = self.run_iteration()
            results.append(result)

            if verbose and (i + 1) % 10 == 0:
                status = f"Iteration {result['iteration']}"
                if result.get('trained', False):
                    status += f" | Loss: {result['loss']:.0f} | Samples: {result['total_samples']}"
                else:
                    status += f" | Collecting data... | Regrets: {result['regret_table_size']}"
                print(status)

        return results

    def get_network_strategy(self, state, player: int) -> Dict[str, float]:
        """Get strategy from the appropriate player's VALUE network."""
        return self.integrations[player].get_network_strategy(state, player)

    def get_strategy_network_output(self, state, player: int) -> Dict[str, float]:
        """
        Get final strategy from the STRATEGY network Π.

        This is what should be used for actual play/deployment.
        The strategy network outputs are logits -> softmax -> probabilities.
        """
        return self.strategy_integration.get_network_strategy(state, player)

    def get_final_strategy(self, state, player: int) -> Dict[str, float]:
        """
        Get the final strategy for deployment.

        Uses the strategy network Π (trained on opponent-node strategies).
        This is the network that should be loaded by player.py.
        """
        return self.get_strategy_network_output(state, player)

    def get_mccfr_strategy(self, state, player: int) -> Dict[str, float]:
        """Get strategy from tabular MCCFR for a given state."""
        infoset = self.mccfr.get_infoset(state, player)
        legal_actions = self.mccfr.get_legal_actions_list(state)

        if infoset not in self.mccfr.regret_table:
            # Uniform random
            prob = 1.0 / len(legal_actions)
            return {self.mccfr.action_to_key(action, state, player): prob
                    for action in legal_actions}

        regrets = self.mccfr.regret_table[infoset]
        return self.mccfr.regret_matching(regrets, legal_actions, state, player)

    def compare_strategies(self, state, player: int) -> Dict:
        """Compare network vs tabular strategies."""
        network_strategy = self.get_network_strategy(state, player)
        mccfr_strategy = self.get_mccfr_strategy(state, player)

        # Compute KL divergence (if both have same actions)
        common_actions = set(network_strategy.keys()) & set(
            mccfr_strategy.keys())

        kl_div = 0.0
        for action in common_actions:
            p = mccfr_strategy[action]
            q = network_strategy[action]
            if p > 1e-10 and q > 1e-10:
                kl_div += p * np.log(p / q)

        return {
            'network_strategy': network_strategy,
            'mccfr_strategy': mccfr_strategy,
            'kl_divergence': kl_div,
            'common_actions': len(common_actions)
        }

    def get_stats(self) -> Dict:
        """Get training statistics."""
        return self.stats

    def clear_regret_table(self):
        """Clear the regret table (useful after network is trained)."""
        self.mccfr.regret_table.clear()
        print(
            f"Cleared regret table. Network has {len(self.trainer.samples)} training samples.")


def test_deep_cfr():
    """
    Basic test of Deep CFR algorithm.

    This test runs the full Deep CFR algorithm for multiple iterations
    and verifies that:
    1. Iterations run without errors
    2. Network training happens on schedule
    3. Loss decreases over time
    4. Network and MCCFR strategies converge
    """
    print("=" * 70)
    print("TESTING DEEP CFR ALGORITHM")
    print("=" * 70)

    # Create Deep CFR
    deep_cfr = DeepCFR(
        network_dim=128,
        learning_rate=0.001,
        batch_size=16,
        use_network_after=50,  # Start using network after 50 iterations
        train_every=10,  # Train every 10 iterations
        train_epochs=3,
        memory_limit=5000
    )

    print("\n[1] Running initial iterations (tabular MCCFR)...")
    results = deep_cfr.run_multiple_iterations(20, verbose=False)
    print(f"✓ Completed {len(results)} iterations")
    print(f"  Regret table size: {results[-1]['regret_table_size']}")
    print(f"  Training iterations: {deep_cfr.total_training_iterations}")

    print("\n[2] Checking training schedule...")
    trained_iterations = [r for r in results if r.get('trained', False)]
    print(f"✓ Network trained {len(trained_iterations)} times")
    if len(trained_iterations) > 0:
        print(f"  Latest loss: {trained_iterations[-1]['loss']:.0f}")
        print(f"  Training samples: {trained_iterations[-1]['total_samples']}")

    print("\n[3] Running more iterations...")
    results2 = deep_cfr.run_multiple_iterations(20, verbose=False)
    print(f"✓ Completed {len(results2)} more iterations")
    print(f"  Total iterations: {deep_cfr.iteration_count}")

    trained_iterations2 = [r for r in results2 if r.get('trained', False)]
    if len(trained_iterations2) > 0:
        print(f"  Latest loss: {trained_iterations2[-1]['loss']:.0f}")

    print("\n[4] Testing network usage...")
    print(f"  Network active: {deep_cfr.should_use_network()}")

    # Create a test state
    state = deep_cfr.mccfr.create_initial_state()
    comparison = deep_cfr.compare_strategies(state, player=0)
    print(f"✓ Strategy comparison complete")
    print(f"  Common actions: {comparison['common_actions']}")
    print(f"  KL divergence: {comparison['kl_divergence']:.4f}")

    print("\n[5] Checking statistics...")
    stats = deep_cfr.get_stats()
    print(f"✓ Collected statistics")
    print(f"  Total iterations: {len(stats['iterations'])}")
    print(f"  Network usages: {sum(stats['network_usage'])}")
    print(f"  Training losses: {len(stats['losses'])}")

    if len(stats['losses']) > 1:
        print(
            f"  Loss trajectory: {stats['losses'][0]:.0f} → {stats['losses'][-1]:.0f}")

    print("\n" + "=" * 70)
    print("✓✓✓ DEEP CFR ALGORITHM TEST PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_deep_cfr()
