"""
NH3 Uptake Predictor - Deep Ensemble MLP

Maps MOFDiff latent vectors (z ∈ R^256) to predicted NH3 uptake (mmol/g).

Architecture:
  - Uses ALL 256 latent dimensions (no feature selection bottleneck)
  - Deep ensemble of K models for uncertainty quantification
  - Spectral normalization for smooth gradient landscape (critical for optimization)
  - Residual connections for training stability
  - Separate mean and log-variance heads for heteroscedastic uncertainty

Why this design:
  1. The optimizer will backprop through this network - smooth gradients are essential
  2. Ensemble disagreement gives confidence bounds on predictions
  3. Full 256-dim input preserves all latent information (vs. 30-dim feature selection)
  4. Spectral norm prevents gradient explosion during latent optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple


class SpectralLinear(nn.Module):
    """Linear layer with spectral normalization for Lipschitz-bounded gradients."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.utils.spectral_norm(nn.Linear(in_features, out_features))

    def forward(self, x):
        return self.linear(x)


class ResidualBlock(nn.Module):
    """Residual block with spectral normalization."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            SpectralLinear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            SpectralLinear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class NH3PredictorNet(nn.Module):
    """
    Single predictor network: z (256-dim) -> NH3 uptake prediction.
    
    Architecture:
      Input(256) -> Linear(256, 512) -> GELU -> Dropout
                 -> ResBlock(512) -> ResBlock(512)
                 -> Linear(512, 256) -> GELU -> Dropout
                 -> ResBlock(256)
                 -> Linear(256, 128) -> GELU
                 -> Linear(128, 1)  [mean]
                 -> Linear(128, 1)  [log_var, optional]
    """
    def __init__(
        self,
        in_dim: int = 256,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.15,
        heteroscedastic: bool = True,
    ):
        super().__init__()
        self.heteroscedastic = heteroscedastic

        # Encoder pathway
        layers = []
        prev_dim = in_dim
        for i, hdim in enumerate(hidden_dims[:-1]):
            layers.append(SpectralLinear(prev_dim, hdim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            layers.append(ResidualBlock(hdim, dropout))
            prev_dim = hdim
        
        # Final hidden layer (no residual, just transform)
        layers.append(SpectralLinear(prev_dim, hidden_dims[-1]))
        layers.append(nn.GELU())

        self.encoder = nn.Sequential(*layers)
        
        # Mean head
        self.mean_head = nn.Linear(hidden_dims[-1], 1)
        
        # Log-variance head (for heteroscedastic uncertainty)
        if heteroscedastic:
            self.logvar_head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            z: Latent vectors [batch, 256]
        Returns:
            mean: Predicted NH3 uptake [batch, 1]
            log_var: Log variance of prediction [batch, 1] (if heteroscedastic)
        """
        h = self.encoder(z)
        mean = self.mean_head(h)
        
        if self.heteroscedastic:
            log_var = self.logvar_head(h)
            # Clamp log_var for stability
            log_var = torch.clamp(log_var, min=-10, max=5)
            return mean, log_var
        
        return mean, None


class NH3PredictorEnsemble(nn.Module):
    """
    Deep ensemble of NH3 predictors for robust prediction + uncertainty.
    
    Ensemble disagreement + heteroscedastic variance = total uncertainty.
    This is used by the optimizer to:
      1. Get gradient of predicted NH3 w.r.t. z (for optimization direction)
      2. Get uncertainty estimate (to penalize moving into uncertain regions)
    """
    def __init__(
        self,
        n_models: int = 10,
        in_dim: int = 256,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.15,
    ):
        super().__init__()
        self.n_models = n_models
        self.models = nn.ModuleList([
            NH3PredictorNet(in_dim, hidden_dims, dropout, heteroscedastic=True)
            for _ in range(n_models)
        ])
        
        # Input normalization parameters (set during training)
        self.register_buffer('z_mean', torch.zeros(in_dim))
        self.register_buffer('z_std', torch.ones(in_dim))
        self.register_buffer('y_mean', torch.tensor(0.0))
        self.register_buffer('y_std', torch.tensor(1.0))
        
        # Whether y was log-transformed during training
        self.log_transform_y = False
    
    def normalize_z(self, z: torch.Tensor) -> torch.Tensor:
        """Normalize latent vectors using training statistics."""
        return (z - self.z_mean) / (self.z_std + 1e-8)
    
    def denormalize_y(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized prediction back to original scale (mmol/g)."""
        y = y_norm * self.y_std + self.y_mean
        if self.log_transform_y:
            y = torch.expm1(y)  # Inverse of log1p
        return y
    
    def normalize_y(self, y: torch.Tensor) -> torch.Tensor:
        """Normalize target values."""
        if self.log_transform_y:
            y = torch.log1p(y)
        return (y - self.y_mean) / (self.y_std + 1e-8)

    def forward(
        self, z: torch.Tensor, return_individual: bool = False
    ) -> dict:
        """
        Predict NH3 uptake with uncertainty.
        
        Args:
            z: Raw latent vectors [batch, 256] (will be normalized internally)
            return_individual: Whether to return individual model predictions
        
        Returns:
            dict with:
                'mean': Ensemble mean prediction (original scale) [batch]
                'std': Total uncertainty [batch]
                'epistemic': Epistemic uncertainty (model disagreement) [batch]
                'aleatoric': Aleatoric uncertainty (data noise) [batch]
                'individual_means': Per-model predictions [n_models, batch] (optional)
        """
        z_norm = self.normalize_z(z)
        
        all_means = []
        all_vars = []
        
        for model in self.models:
            mean, log_var = model(z_norm)
            all_means.append(mean.squeeze(-1))
            all_vars.append(torch.exp(log_var).squeeze(-1))
        
        # Stack: [n_models, batch]
        all_means = torch.stack(all_means)
        all_vars = torch.stack(all_vars)
        
        # Ensemble statistics (in normalized space)
        ensemble_mean_norm = all_means.mean(dim=0)
        
        # Epistemic uncertainty: variance of means across ensemble
        epistemic_var = all_means.var(dim=0)
        
        # Aleatoric uncertainty: mean of individual variances
        aleatoric_var = all_vars.mean(dim=0)
        
        # Total uncertainty
        total_var = epistemic_var + aleatoric_var
        
        # Convert to original scale
        ensemble_mean = self.denormalize_y(ensemble_mean_norm)
        
        # For uncertainty: approximate conversion from normalized -> original scale
        # If log transform: delta method gives std_orig ≈ std_log * mean_orig
        # but for optimization we just need monotonic uncertainty measure
        scale = self.y_std
        if self.log_transform_y:
            # Approximate: multiply by the exp of the log-space mean
            log_mean = ensemble_mean_norm * self.y_std + self.y_mean
            scale = self.y_std * torch.exp(log_mean).clamp(min=0.01)
        
        total_std = torch.sqrt(total_var) * scale
        epistemic_std = torch.sqrt(epistemic_var) * scale
        aleatoric_std = torch.sqrt(aleatoric_var) * scale
        
        result = {
            'mean': ensemble_mean,
            'std': total_std,
            'epistemic': epistemic_std,
            'aleatoric': aleatoric_std,
        }
        
        if return_individual:
            result['individual_means'] = self.denormalize_y(all_means)
        
        return result

    def predict_with_grad(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict NH3 uptake (differentiable, for optimization).
        Returns only the mean prediction in original scale.
        """
        z_norm = self.normalize_z(z)
        means = []
        for model in self.models:
            mean, _ = model(z_norm)
            means.append(mean.squeeze(-1))
        
        ensemble_mean_norm = torch.stack(means).mean(dim=0)
        return self.denormalize_y(ensemble_mean_norm)
