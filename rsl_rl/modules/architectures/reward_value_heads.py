"""
Reward and Value heads for the latent-space world model (Phase 3).

Contains:
- RewardHead: Predicts scalar reward from (z, a) with optional frozen random prior.
- ValueHead: Predicts scalar state-value V(z) with optional frozen random prior.
- ValueHeadEnsemble: Wraps multiple ValueHeads; provides mean, per-head, and uncertainty.

All heads follow the prior mechanism from LatentDynamicsHead:
  output = main_mlp(x) + prior_scale * prior_mlp(x).detach()

Reference: TD-MPC2 (Hansen et al., 2024), Osband et al. (2018) randomized priors.
"""

import copy
import torch
import torch.nn as nn

from rsl_rl.modules.architectures.latent_layers import latent_mlp


class RewardHead(nn.Module):
    """Predicts scalar reward from concatenated (latent_state, action).

    Architecture: latent_mlp (NormedLinear hidden layers) -> scalar output.
    Optional frozen random prior for diversity.

    Args:
        latent_dim: Latent state dimension.
        action_dim: Action dimension.
        hidden_dims: Hidden layer widths (default [256, 256]).
        prior_scale: Scale for randomized prior output (0 = disabled).
        prior_hidden_div: Divisor for prior hidden dimensions (default 4).
        device: Device to place tensors on.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
        prior_scale: float = 0.1,
        prior_hidden_div: int = 4,
        device: str = "cpu",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.prior_scale = prior_scale
        in_dim = latent_dim + action_dim

        # Main trainable MLP: (z, a) -> scalar reward
        self.main_mlp = latent_mlp(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            out_dim=1,
            output_act=None,  # unconstrained scalar output
        ).to(device)

        # Frozen random prior
        if self.prior_scale > 0:
            prior_hidden_dims = [max(h // prior_hidden_div, 8) for h in hidden_dims]
            layers: list[nn.Module] = []
            curr_in = in_dim
            for h in prior_hidden_dims:
                layers.append(nn.Linear(curr_in, h))
                layers.append(nn.ReLU())
                curr_in = h
            layers.append(nn.Linear(prior_hidden_dims[-1], 1))
            self.prior_mlp = nn.Sequential(*layers).to(device)
            for m in self.prior_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
        else:
            self.prior_mlp = None

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Predict reward.

        Args:
            z: Latent state [B, latent_dim].
            a: Action [B, action_dim].

        Returns:
            Predicted reward [B, 1].
        """
        x = torch.cat([z, a], dim=-1)
        out = self.main_mlp(x)
        if self.prior_mlp is not None:
            with torch.no_grad():
                prior_out = self.prior_mlp(x)
            out = out + prior_out.detach() * self.prior_scale
        return out


class ValueHead(nn.Module):
    """Predicts scalar state-value V(z) from latent state.

    Architecture: latent_mlp (NormedLinear hidden layers) -> scalar output.
    Optional frozen random prior for diversity.

    Args:
        latent_dim: Latent state dimension.
        hidden_dims: Hidden layer widths (default [256, 256]).
        prior_scale: Scale for randomized prior output (0 = disabled).
        prior_hidden_div: Divisor for prior hidden dimensions (default 4).
        device: Device to place tensors on.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
        prior_scale: float = 0.1,
        prior_hidden_div: int = 4,
        device: str = "cpu",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.latent_dim = latent_dim
        self.prior_scale = prior_scale

        # Main trainable MLP: z -> scalar value
        self.main_mlp = latent_mlp(
            in_dim=latent_dim,
            hidden_dims=hidden_dims,
            out_dim=1,
            output_act=None,
        ).to(device)

        # Frozen random prior
        if self.prior_scale > 0:
            prior_hidden_dims = [max(h // prior_hidden_div, 8) for h in hidden_dims]
            layers: list[nn.Module] = []
            curr_in = latent_dim
            for h in prior_hidden_dims:
                layers.append(nn.Linear(curr_in, h))
                layers.append(nn.ReLU())
                curr_in = h
            layers.append(nn.Linear(prior_hidden_dims[-1], 1))
            self.prior_mlp = nn.Sequential(*layers).to(device)
            for m in self.prior_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
        else:
            self.prior_mlp = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict state value.

        Args:
            z: Latent state [B, latent_dim].

        Returns:
            Predicted value [B, 1].
        """
        out = self.main_mlp(z)
        if self.prior_mlp is not None:
            with torch.no_grad():
                prior_out = self.prior_mlp(z)
            out = out + prior_out.detach() * self.prior_scale
        return out


class ValueHeadEnsemble(nn.Module):
    """Ensemble of ValueHeads for uncertainty estimation.

    Provides:
    - Ensemble mean value prediction
    - Per-head predictions (for TD target computation per ensemble member)
    - Epistemic uncertainty (std across heads)

    Args:
        n_heads: Number of value heads in the ensemble.
        latent_dim: Latent state dimension.
        hidden_dims: Hidden layer widths per head.
        prior_scale: Scale for randomized prior per head.
        prior_hidden_div: Divisor for prior hidden dimensions.
        device: Device to place tensors on.
    """

    def __init__(
        self,
        n_heads: int = 5,
        latent_dim: int = 256,
        hidden_dims: list[int] | None = None,
        prior_scale: float = 0.1,
        prior_hidden_div: int = 4,
        device: str = "cpu",
    ):
        super().__init__()
        self.n_heads = n_heads
        self.latent_dim = latent_dim
        self.heads = nn.ModuleList([
            ValueHead(
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                prior_scale=prior_scale,
                prior_hidden_div=prior_hidden_div,
                device=device,
            )
            for _ in range(n_heads)
        ])

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through all value heads.

        Args:
            z: Latent state [B, latent_dim].

        Returns:
            Tuple of:
            - mean_value: Ensemble mean [B, 1]
            - all_values: Per-head predictions [n_heads, B, 1]
            - uncertainty: Std across heads [B, 1]
        """
        all_values = torch.stack([head(z) for head in self.heads], dim=0)  # [n_heads, B, 1]
        mean_value = all_values.mean(dim=0)   # [B, 1]
        uncertainty = all_values.std(dim=0)    # [B, 1]
        return mean_value, all_values, uncertainty

    def forward_single(self, z: torch.Tensor, head_idx: int) -> torch.Tensor:
        """Forward pass through a single value head.

        Args:
            z: Latent state [B, latent_dim].
            head_idx: Which head to use.

        Returns:
            Predicted value [B, 1].
        """
        return self.heads[head_idx](z)
