"""
Core Deep CFR Algorithm Components

This package contains the main algorithm logic:
- deep_cfr: Main Deep CFR orchestrator
- mccfr: Monte Carlo CFR implementation
- trainer: Network training pipeline
- integration: Network + MCCFR bridge
"""

from core.deep_cfr import DeepCFR
from core.mccfr import MCCFR
from core.trainer import DeepCFRTrainer, TrainingSample
from core.integration import NetworkMCCFRIntegration

__all__ = [
    'DeepCFR',
    'MCCFR', 
    'DeepCFRTrainer',
    'TrainingSample',
    'NetworkMCCFRIntegration'
]
