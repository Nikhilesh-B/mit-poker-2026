"""
Training Monitor for Deep CFR

Provides real-time feedback on training progress:
1. Logs training metrics (loss, etc.) to CSV
2. Evaluates model against baseline bots every N epochs
3. Tracks action distributions (am I too tight/loose?)
4. Generates plots to visualize improvement

This gives you visibility into what's happening during training!
"""

import torch
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from regret_network import RegretNetwork, regret_matching, sample_action
from deep_cfr_encoding import encode_state
from action_mapping import get_legal_mask, action_index_to_engine_action, ActionIndex
from skeleton.states import RoundState, TerminalState, STARTING_STACK, SMALL_BLIND, BIG_BLIND
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction


class RandomBot:
    """Simple random baseline bot for evaluation"""

    def get_action(self, round_state: RoundState, active: int):
        """Pick a random legal action"""
        legal_actions = round_state.legal_actions()
        return random.choice(legal_actions)


class TrainingMonitor:
    """
    Monitors training progress and logs metrics.

    Tracks:
    - Training loss over epochs
    - Win rate vs baseline (every N epochs)
    - Action distribution (fold/call/raise/discard %)
    - Expected value trends
    """

    def __init__(self, log_dir: str = "Deep_CFR/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # CSV files for metrics
        self.metrics_file = self.log_dir / "training_metrics.csv"
        self.eval_file = self.log_dir / "evaluation_results.csv"

        # Initialize CSV files
        self._init_csv_files()

        # Action distribution tracking
        self.action_counts = {i: 0 for i in range(8)}
        self.action_history = []

    def _init_csv_files(self):
        """Create CSV files with headers"""
        # Training metrics
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'loss', 'avg_p0_value', 'avg_p1_value',
                'buffer_size', 'fold_pct', 'call_pct', 'raise_pct', 'discard_pct'
            ])

        # Evaluation results
        with open(self.eval_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'num_games', 'wins', 'losses', 'avg_chips_won',
                'win_rate', 'bb_per_hand'
            ])

    def log_epoch_metrics(
        self,
        epoch: int,
        loss: float,
        avg_p0_value: float,
        avg_p1_value: float,
        buffer_size: int,
        action_dist: Dict[str, float] = None
    ):
        """Log metrics for this epoch"""
        if action_dist is None:
            action_dist = {'fold': 0, 'call': 0, 'raise': 0, 'discard': 0}

        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, loss, avg_p0_value, avg_p1_value, buffer_size,
                action_dist['fold'], action_dist['call'],
                action_dist['raise'], action_dist['discard']
            ])

    def evaluate_vs_baseline(
        self,
        regret_net: RegretNetwork,
        num_games: int = 100,
        device: str = "cpu"
    ) -> Dict:
        """
        Play N games vs random baseline bot.

        Returns:
            dict with wins, losses, avg_chips_won, win_rate, bb_per_hand
        """
        print(f"\n  → Running evaluation ({num_games} hands vs Random Bot)...")

        baseline = RandomBot()
        total_chips = 0
        wins = 0
        losses = 0

        regret_net.eval()

        for game_idx in range(num_games):
            # Play one hand
            chips_won = self._play_one_hand(regret_net, baseline, device)
            total_chips += chips_won

            if chips_won > 0:
                wins += 1
            elif chips_won < 0:
                losses += 1

        avg_chips = total_chips / num_games
        win_rate = wins / num_games * 100
        bb_per_hand = avg_chips / BIG_BLIND

        regret_net.train()

        results = {
            'wins': wins,
            'losses': losses,
            'avg_chips_won': avg_chips,
            'win_rate': win_rate,
            'bb_per_hand': bb_per_hand
        }

        print(f"  → Results: {wins}W-{losses}L ({win_rate:.1f}% win rate)")
        print(
            f"  → Avg: {avg_chips:+.1f} chips/hand ({bb_per_hand:+.2f} BB/hand)")

        return results

    def _play_one_hand(
        self,
        regret_net: RegretNetwork,
        baseline: RandomBot,
        device: str
    ) -> float:
        """
        Play one hand: regret_net vs baseline.

        Returns chips won by regret_net (can be negative if lost)
        """
        # Create initial state
        round_state = self._create_initial_state()

        # Play the hand
        while not isinstance(round_state, TerminalState):
            active_player = round_state.button % 2

            # Get action from appropriate bot
            if active_player == 0:
                # Our trained bot (player 0)
                action = self._get_model_action(
                    regret_net, round_state, active_player, device
                )
            else:
                # Baseline bot (player 1)
                action = baseline.get_action(round_state, active_player)

            # Apply action
            round_state = round_state.proceed(action)

        # Get result (from player 0's perspective)
        return round_state.deltas[0]

    def _create_initial_state(self) -> RoundState:
        """Create a fresh initial game state"""
        from skeleton.states import deal_cards

        hands = deal_cards()
        button = 0
        street = 0
        pips = [SMALL_BLIND, BIG_BLIND]
        stacks = [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND]
        board = []

        return RoundState(button, street, pips, stacks, hands, board, None)

    def _get_model_action(
        self,
        regret_net: RegretNetwork,
        round_state: RoundState,
        active: int,
        device: str
    ):
        """Get action from trained model"""
        try:
            # Encode state
            # Need a dummy game_state - use a minimal dict
            game_state = {
                'bankroll': round_state.stacks[active],
                'game_clock': 0,
                'round_num': 1
            }

            state_encoding = encode_state(game_state, round_state, active)
            legal_mask = get_legal_mask(round_state, active)

            # Get action from model
            with torch.no_grad():
                state_encoding = state_encoding.to(device)
                legal_mask = legal_mask.to(device)

                predicted_regrets = regret_net(state_encoding)
                strategy = regret_matching(predicted_regrets, legal_mask)
                action_idx = sample_action(strategy)

            return action_index_to_engine_action(action_idx, round_state)

        except Exception as e:
            # Fallback to random if model fails
            legal_actions = round_state.legal_actions()
            return random.choice(legal_actions)

    def log_evaluation(self, epoch: int, num_games: int, results: Dict):
        """Log evaluation results to CSV"""
        with open(self.eval_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, num_games,
                results['wins'], results['losses'],
                results['avg_chips_won'],
                results['win_rate'],
                results['bb_per_hand']
            ])

    def track_action(self, action_idx: int):
        """Track an action taken during training (for distribution analysis)"""
        self.action_counts[action_idx] += 1

    def get_action_distribution(self) -> Dict[str, float]:
        """
        Get action distribution percentages.

        Returns dict with fold%, call%, raise%, discard%
        """
        total = sum(self.action_counts.values())
        if total == 0:
            return {'fold': 0, 'call': 0, 'raise': 0, 'discard': 0}

        fold_pct = self.action_counts[ActionIndex.FOLD.value] / total * 100
        call_pct = (
            self.action_counts[ActionIndex.CHECK_CALL.value]) / total * 100
        raise_pct = (
            self.action_counts[ActionIndex.RAISE_SMALL.value] +
            self.action_counts[ActionIndex.RAISE_MEDIUM.value] +
            self.action_counts[ActionIndex.RAISE_LARGE.value]
        ) / total * 100
        discard_pct = (
            self.action_counts[ActionIndex.DISCARD_0.value] +
            self.action_counts[ActionIndex.DISCARD_1.value] +
            self.action_counts[ActionIndex.DISCARD_2.value]
        ) / total * 100

        return {
            'fold': fold_pct,
            'call': call_pct,
            'raise': raise_pct,
            'discard': discard_pct
        }

    def reset_action_counts(self):
        """Reset action counts for next epoch"""
        self.action_counts = {i: 0 for i in range(8)}


def create_training_plots(log_dir: str = "Deep_CFR/logs"):
    """
    Generate plots from training logs.

    Creates:
    - training_loss.png: Loss over epochs
    - win_rate.png: Win rate vs baseline over epochs
    - action_distribution.png: Fold/call/raise/discard % over time
    """
    import matplotlib.pyplot as plt

    log_dir = Path(log_dir)
    metrics_file = log_dir / "training_metrics.csv"
    eval_file = log_dir / "evaluation_results.csv"

    # Check if files exist
    if not metrics_file.exists():
        print(f"No metrics file found at {metrics_file}")
        return

    # Read metrics
    epochs, losses, p0_vals, p1_vals = [], [], [], []
    fold_pcts, call_pcts, raise_pcts = [], [], []

    with open(metrics_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            losses.append(float(row['loss']))
            p0_vals.append(float(row['avg_p0_value']))
            p1_vals.append(float(row['avg_p1_value']))
            fold_pcts.append(float(row['fold_pct']))
            call_pcts.append(float(row['call_pct']))
            raise_pcts.append(float(row['raise_pct']))

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Deep CFR Training Progress', fontsize=16, fontweight='bold')

    # Plot 1: Training Loss
    axes[0, 0].plot(epochs, losses, linewidth=2, color='#2E86AB')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Regret Prediction Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Expected Values
    axes[0, 1].plot(epochs, p0_vals, label='P0', linewidth=2, color='#06A77D')
    axes[0, 1].plot(epochs, p1_vals, label='P1', linewidth=2, color='#D62246')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Avg Value (chips)')
    axes[0, 1].set_title('Expected Values During Training')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(0, color='black', linestyle='--', alpha=0.5)

    # Plot 3: Action Distribution
    axes[1, 0].plot(epochs, fold_pcts, label='Fold',
                    linewidth=2, color='#D62246')
    axes[1, 0].plot(epochs, call_pcts, label='Call',
                    linewidth=2, color='#F77F00')
    axes[1, 0].plot(epochs, raise_pcts, label='Raise',
                    linewidth=2, color='#06A77D')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Action %')
    axes[1, 0].set_title('Action Distribution Over Training')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 100)

    # Plot 4: Evaluation Results (if available)
    if eval_file.exists():
        eval_epochs, win_rates, bb_per_hand = [], [], []
        with open(eval_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                eval_epochs.append(int(row['epoch']))
                win_rates.append(float(row['win_rate']))
                bb_per_hand.append(float(row['bb_per_hand']))

        ax4_1 = axes[1, 1]
        ax4_2 = ax4_1.twinx()

        line1 = ax4_1.plot(eval_epochs, win_rates, 'o-', linewidth=2,
                           color='#2E86AB', label='Win Rate %')
        line2 = ax4_2.plot(eval_epochs, bb_per_hand, 's-', linewidth=2,
                           color='#06A77D', label='BB/hand')

        ax4_1.set_xlabel('Epoch')
        ax4_1.set_ylabel('Win Rate (%)', color='#2E86AB')
        ax4_2.set_ylabel('BB per Hand', color='#06A77D')
        ax4_1.set_title('Performance vs Random Baseline')
        ax4_1.grid(True, alpha=0.3)
        ax4_1.axhline(50, color='gray', linestyle='--', alpha=0.5)
        ax4_2.axhline(0, color='gray', linestyle='--', alpha=0.5)

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4_1.legend(lines, labels, loc='upper left')
    else:
        axes[1, 1].text(0.5, 0.5, 'No evaluation data yet\n(Run with --eval-interval)',
                        ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Evaluation vs Baseline')

    plt.tight_layout()

    # Save plot
    plot_path = log_dir / "training_progress.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Training plots saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    # Generate plots from existing logs
    import sys
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        log_dir = "Deep_CFR/logs"

    create_training_plots(log_dir)
