"""
Weights & Biases Integration for Deep CFR Training

This module provides optional W&B logging for:
- Training loss curves
- Network weight histograms
- Gradient norms
- Sample statistics
- Configuration tracking

Usage:
    from wandb_logger import WandbLogger
    
    logger = WandbLogger(project="deep-cfr", config={...})
    logger.log_iteration(iteration, metrics)
    logger.log_weights(network, name="strategy_network")
    logger.finish()
"""

import os
from typing import Dict, Optional, Any

# Try to import wandb, but don't fail if not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


class WandbLogger:
    """
    Weights & Biases logger for Deep CFR training.
    
    Tracks:
    - Loss per network (V0, V1, Strategy)
    - Loss progression within training sessions
    - Sample counts
    - Gradient norms
    - Network weight distributions
    - Training configuration
    """
    
    def __init__(
        self,
        project: str = "deep-cfr",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        log_weights_every: int = 10,
        log_gradients: bool = True
    ):
        """
        Initialize W&B logger.
        
        Args:
            project: W&B project name
            name: Run name (auto-generated if None)
            config: Training configuration to log
            enabled: Whether to actually log (set False to disable)
            log_weights_every: Log weight histograms every N iterations
            log_gradients: Whether to log gradient norms
        """
        self.enabled = enabled and WANDB_AVAILABLE
        self.log_weights_every = log_weights_every
        self.log_gradients = log_gradients
        self.iteration = 0
        
        if not self.enabled:
            if enabled and not WANDB_AVAILABLE:
                print("W&B logging disabled: wandb not installed")
            return
        
        # Initialize W&B run
        wandb.init(
            project=project,
            name=name,
            config=config or {},
            reinit=True
        )
        
        print(f"W&B logging initialized: {wandb.run.url}")
    
    def log_config(self, config: Dict[str, Any]):
        """Log training configuration."""
        if not self.enabled:
            return
        wandb.config.update(config)
    
    def log_iteration(self, iteration: int, metrics: Dict[str, Any]):
        """
        Log metrics for a training iteration.
        
        Expected metrics:
        - loss_p0, loss_p1: Value network losses
        - loss_strategy: Strategy network loss
        - loss_start_*, loss_end_*: Loss progression
        - samples_p0, samples_p1, samples_strategy: Sample counts
        - new_samples_*: New samples this iteration
        """
        if not self.enabled:
            return
        
        self.iteration = iteration
        
        # Prepare log dict with prefixes for organization
        log_dict = {"iteration": iteration}
        
        # Losses
        if "loss_p0" in metrics:
            log_dict["loss/value_p0"] = metrics["loss_p0"]
        if "loss_p1" in metrics:
            log_dict["loss/value_p1"] = metrics["loss_p1"]
        if "loss_strategy" in metrics:
            log_dict["loss/strategy"] = metrics["loss_strategy"]
        if "loss" in metrics:
            log_dict["loss/average"] = metrics["loss"]
        
        # Loss progression (within training session)
        if "loss_start_p0" in metrics and metrics["loss_start_p0"] is not None:
            log_dict["loss_progression/p0_start"] = metrics["loss_start_p0"]
            log_dict["loss_progression/p0_end"] = metrics.get("loss_end_p0", 0)
        if "loss_start_p1" in metrics and metrics["loss_start_p1"] is not None:
            log_dict["loss_progression/p1_start"] = metrics["loss_start_p1"]
            log_dict["loss_progression/p1_end"] = metrics.get("loss_end_p1", 0)
        if "loss_start_strategy" in metrics and metrics["loss_start_strategy"] is not None:
            log_dict["loss_progression/strategy_start"] = metrics["loss_start_strategy"]
            log_dict["loss_progression/strategy_end"] = metrics.get("loss_end_strategy", 0)
        
        # Loss reduction percentages
        if "loss_reduction_p0" in metrics and metrics["loss_reduction_p0"] is not None:
            log_dict["loss_reduction/p0_pct"] = metrics["loss_reduction_p0"]
        if "loss_reduction_p1" in metrics and metrics["loss_reduction_p1"] is not None:
            log_dict["loss_reduction/p1_pct"] = metrics["loss_reduction_p1"]
        if "loss_reduction_strategy" in metrics and metrics["loss_reduction_strategy"] is not None:
            log_dict["loss_reduction/strategy_pct"] = metrics["loss_reduction_strategy"]
        
        # Sample counts
        if "total_samples_p0" in metrics:
            log_dict["samples/total_p0"] = metrics["total_samples_p0"]
        if "total_samples_p1" in metrics:
            log_dict["samples/total_p1"] = metrics["total_samples_p1"]
        if "strategy_samples" in metrics:
            log_dict["samples/strategy"] = metrics["strategy_samples"]
        if "total_samples" in metrics:
            log_dict["samples/total"] = metrics["total_samples"]
        
        # New samples this iteration
        if "new_samples_p0" in metrics:
            log_dict["samples/new_p0"] = metrics["new_samples_p0"]
        if "new_samples_p1" in metrics:
            log_dict["samples/new_p1"] = metrics["new_samples_p1"]
        
        # Traversals
        if "traversals" in metrics:
            log_dict["traversals"] = metrics["traversals"]
        
        wandb.log(log_dict, step=iteration)
    
    def log_weights(self, network, name: str = "network", iteration: Optional[int] = None):
        """
        Log network weight histograms.
        
        Args:
            network: PyTorch network
            name: Name prefix for weights (e.g., "strategy", "value_p0")
            iteration: Step number (uses self.iteration if None)
        """
        if not self.enabled:
            return
        
        step = iteration if iteration is not None else self.iteration
        
        # Only log weights periodically to avoid overhead
        if step % self.log_weights_every != 0:
            return
        
        try:
            import torch
            log_dict = {}
            
            for param_name, param in network.named_parameters():
                if param.requires_grad:
                    # Log weight histogram
                    log_dict[f"weights/{name}/{param_name}"] = wandb.Histogram(
                        param.data.cpu().numpy().flatten()
                    )
                    
                    # Log weight stats
                    log_dict[f"weight_stats/{name}/{param_name}_mean"] = param.data.mean().item()
                    log_dict[f"weight_stats/{name}/{param_name}_std"] = param.data.std().item()
                    log_dict[f"weight_stats/{name}/{param_name}_max"] = param.data.abs().max().item()
            
            wandb.log(log_dict, step=step)
        except Exception as e:
            print(f"Warning: Failed to log weights: {e}")
    
    def log_gradients(self, network, name: str = "network", iteration: Optional[int] = None):
        """
        Log gradient statistics.
        
        Args:
            network: PyTorch network (after backward pass)
            name: Name prefix
            iteration: Step number
        """
        if not self.enabled or not self.log_gradients:
            return
        
        step = iteration if iteration is not None else self.iteration
        
        try:
            import torch
            log_dict = {}
            total_norm = 0.0
            
            for param_name, param in network.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    total_norm += grad_norm ** 2
                    
                    # Log per-layer gradient norm
                    log_dict[f"gradients/{name}/{param_name}_norm"] = grad_norm
            
            total_norm = total_norm ** 0.5
            log_dict[f"gradients/{name}/total_norm"] = total_norm
            
            wandb.log(log_dict, step=step)
        except Exception as e:
            print(f"Warning: Failed to log gradients: {e}")
    
    def log_action_distribution(self, action_probs: Dict[str, float], name: str = "strategy"):
        """
        Log action probability distribution.
        
        Args:
            action_probs: Dict mapping action name to probability
            name: Name for this distribution
        """
        if not self.enabled:
            return
        
        # Create a bar chart
        data = [[action, prob] for action, prob in action_probs.items()]
        table = wandb.Table(data=data, columns=["action", "probability"])
        wandb.log({
            f"action_dist/{name}": wandb.plot.bar(table, "action", "probability", title=f"{name} Action Distribution")
        }, step=self.iteration)
    
    def log_custom(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log custom metrics."""
        if not self.enabled:
            return
        wandb.log(metrics, step=step or self.iteration)
    
    def watch_network(self, network, name: str = "network", log_freq: int = 100):
        """
        Use W&B's automatic gradient/weight watching.
        
        Note: This adds overhead, use sparingly.
        """
        if not self.enabled:
            return
        wandb.watch(network, log=name, log_freq=log_freq)
    
    def finish(self):
        """Finish the W&B run."""
        if not self.enabled:
            return
        wandb.finish()
    
    def is_enabled(self) -> bool:
        """Check if logging is enabled."""
        return self.enabled


def create_wandb_config(args) -> Dict[str, Any]:
    """
    Create W&B config from argparse args.
    
    Args:
        args: argparse Namespace
    
    Returns:
        Config dict for W&B
    """
    return {
        "iterations": getattr(args, "iterations", None),
        "network_dim": getattr(args, "network_dim", None),
        "learning_rate": getattr(args, "learning_rate", None),
        "batch_size": getattr(args, "batch_size", None),
        "use_network_after": getattr(args, "use_network_after", None),
        "train_every": getattr(args, "train_every", None),
        "train_epochs": getattr(args, "train_epochs", None),
        "traversals_per_iter": getattr(args, "traversals", None),
        "sgd_iterations": getattr(args, "sgd_steps", None),
        "device": getattr(args, "device", None),
    }


# Quick test
if __name__ == "__main__":
    print("Testing WandbLogger...")
    
    if WANDB_AVAILABLE:
        # Test with a dummy run
        logger = WandbLogger(
            project="deep-cfr-test",
            name="test-run",
            config={"test": True},
            enabled=True
        )
        
        # Log some test metrics
        for i in range(5):
            logger.log_iteration(i, {
                "loss_p0": 10.0 - i,
                "loss_p1": 12.0 - i,
                "loss_strategy": 8.0 - i,
                "total_samples": i * 100
            })
        
        logger.finish()
        print("Test complete! Check W&B dashboard.")
    else:
        print("wandb not installed, skipping test")
