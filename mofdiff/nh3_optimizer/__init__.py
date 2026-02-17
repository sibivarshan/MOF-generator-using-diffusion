"""
NH3 Latent Vector Optimizer for MOFDiff

This module provides a gradient-based optimizer that generates optimized
latent vectors for MOFDiff, targeting specific NH3 adsorption capacities.

Key components:
- NH3PredictorEnsemble: Deep ensemble predictor mapping z -> NH3 uptake
- LatentOptimizer: Gradient-based optimizer with chemistry constraints
- train_predictor: Training pipeline for the NH3 predictor ensemble
- create_optimizer: Factory to load a trained optimizer from checkpoint

Usage:
    from mofdiff.nh3_optimizer import create_optimizer
    opt = create_optimizer("pretrained/nh3_optimizer.pt")
    result = opt.optimize(target_nh3=5.0, n_samples=10)
    z_optimized = result['z']  # Feed to model.sample(z=z_optimized)
"""

from mofdiff.nh3_optimizer.predictor import NH3PredictorEnsemble, NH3PredictorNet
from mofdiff.nh3_optimizer.optimizer import LatentOptimizer, create_optimizer
from mofdiff.nh3_optimizer.train import train_predictor, load_predictor

__all__ = [
    "NH3PredictorEnsemble",
    "NH3PredictorNet",
    "LatentOptimizer",
    "create_optimizer",
    "train_predictor",
    "load_predictor",
]
