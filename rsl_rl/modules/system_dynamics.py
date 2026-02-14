import copy
import torch
import torch.nn as nn
from rsl_rl.modules.architectures import MLPBase, RNNBase, MLPStateHead, MLPStateHeadWithPrior, MLPAuxiliaryHead
from rsl_rl.modules.architectures.mlp import LatentDynamicsHead
from rsl_rl.modules.architectures.encoder_decoder import StateEncoder, StateDecoder
from rsl_rl.modules.architectures.xlstm_base import xLSTMBase, XLSTM_AVAILABLE

class SystemDynamicsEnsemble(nn.Module):
    """
    Ensemble of world models for model-based RL.
    
    Supports two operating modes:
    
    1. Raw state mode (latent_mode=False, default): 
       Predicts next raw state directly. Backbone processes (state, action) pairs.
       Heads predict (mean, std) in raw state space with residual connections.
    
    2. Latent state mode (latent_mode=True):
       Encoder maps raw state to latent representation (with SimNorm).
       Backbone processes (latent_state, action) pairs.
       Heads predict next latent state (with SimNorm, no residual).
       Decoder maps latent back to raw state for reconstruction loss and imagination.
       Uncertainty is purely epistemic (ensemble disagreement in latent space).
    
    Supports randomized priors for ensemble diversity (Osband et al., 2018).
    
    Args:
        state_dim: State dimension.
        action_dim: Action dimension.
        extension_dim: Extension output dimension (0 to disable).
        contact_dim: Contact prediction dimension (0 to disable).
        termination_dim: Termination prediction dimension (0 to disable).
        device: Device to place tensors on.
        ensemble_size: Number of ensemble members (default 1 = single model).
        history_horizon: Number of past timesteps to condition on.
        architecture_config: Dict specifying network architecture.
        freeze_auxiliary: Whether to freeze auxiliary heads.
        uncertainty_metric: "std" (original) or "variance" (theoretically sound).
        prior_scale: Scale for randomized priors (0 = disabled, 1.0 = standard).
        prior_hidden_div: Divisor for prior hidden dims (default 4).
        bootstrap: Whether to use bootstrap sampling per ensemble member.
        latent_mode: Whether to use latent-space dynamics (default False).
        latent_dim: Latent representation dimension (required if latent_mode=True).
        simnorm_dim: SimNorm group size (default 8).
        encoder_hidden_dims: Hidden layer widths for encoder (default [256]).
        decoder_hidden_dims: Hidden layer widths for decoder (default [256]).
        latent_head_hidden_dims: Hidden layer widths for latent dynamics head (default [256]).
        encoder_dropout: Dropout rate for encoder (default 0.0).
        consistency_coef: Weight for consistency loss (default 2.0).
        reconstruction_coef: Weight for reconstruction loss (default 1.0).
        target_encoder_momentum: EMA momentum for target encoder (default 0.99).
            Only used when latent_mode=True. The target encoder parameters are
            updated as: θ_target ← momentum * θ_target + (1 - momentum) * θ_online
        encoder_consistency_coef: Weight for bidirectional encoder consistency loss
            (default 0.0, disabled). When > 0, adds a loss term that trains the encoder
            to produce representations consistent with dynamics predictions (TD-MPC2 style).
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        extension_dim: int,
        contact_dim: int,
        termination_dim: int,
        device: str,
        ensemble_size: int = 1,
        history_horizon: int = 1,
        architecture_config: dict = None,
        freeze_auxiliary: bool = False,
        uncertainty_metric: str = "std",
        prior_scale: float = 0.0,
        prior_hidden_div: int = 4,
        bootstrap: bool = True,
        # Latent space parameters
        latent_mode: bool = False,
        latent_dim: int = 256,
        simnorm_dim: int = 8,
        encoder_hidden_dims: list = None,
        decoder_hidden_dims: list = None,
        latent_head_hidden_dims: list = None,
        encoder_dropout: float = 0.0,
        consistency_coef: float = 2.0,
        reconstruction_coef: float = 1.0,
        target_encoder_momentum: float = 0.99,
        encoder_consistency_coef: float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.extension_dim = extension_dim
        self.contact_dim = contact_dim
        self.termination_dim = termination_dim
        self.device = device
        self.ensemble_size = ensemble_size
        self.history_horizon = history_horizon
        self.architecture_config = architecture_config
        self.freeze_auxiliary = freeze_auxiliary
        self.uncertainty_metric = uncertainty_metric
        self.prior_scale = prior_scale
        self.prior_hidden_div = prior_hidden_div
        self.bootstrap = bootstrap
        
        # Latent space config
        self.latent_mode = latent_mode
        self.latent_dim = latent_dim if latent_mode else state_dim
        self.simnorm_dim = simnorm_dim
        self.encoder_hidden_dims = encoder_hidden_dims or [256]
        self.decoder_hidden_dims = decoder_hidden_dims or [256]
        self.latent_head_hidden_dims = latent_head_hidden_dims or [256]
        self.encoder_dropout = encoder_dropout
        self.consistency_coef = consistency_coef
        self.reconstruction_coef = reconstruction_coef
        self.target_encoder_momentum = target_encoder_momentum
        self.encoder_consistency_coef = encoder_consistency_coef
        
        self._init_networks()

    def _init_networks(self):
        # --- Encoder and Decoder (latent mode only) ---
        if self.latent_mode:
            self.encoder = StateEncoder(
                state_dim=self.state_dim,
                latent_dim=self.latent_dim,
                hidden_dims=self.encoder_hidden_dims,
                simnorm_dim=self.simnorm_dim,
                dropout=self.encoder_dropout,
            ).to(self.device)
            
            self.decoder = StateDecoder(
                latent_dim=self.latent_dim,
                state_dim=self.state_dim,
                hidden_dims=self.decoder_hidden_dims,
            ).to(self.device)
        else:
            self.encoder = None
            self.decoder = None
        
        # --- EMA target encoder (latent mode only, Phase 2) ---
        if self.latent_mode and self.encoder is not None:
            self.target_encoder = copy.deepcopy(self.encoder)
            for param in self.target_encoder.parameters():
                param.requires_grad = False
        else:
            self.target_encoder = None
        
        # --- Base network (backbone) ---
        self.state_base = self._create_base()
        
        # --- State/Latent prediction heads ---
        if self.latent_mode:
            # Inject latent_head_hidden_dims into architecture_config for heads
            latent_arch_config = dict(self.architecture_config)
            latent_arch_config["latent_head_hidden_dims"] = self.latent_head_hidden_dims
            
            self.state_heads = nn.ModuleList([
                LatentDynamicsHead(
                    self.base_output_dim,
                    self.latent_dim,
                    self.device,
                    latent_arch_config,
                    simnorm_dim=self.simnorm_dim,
                    prior_scale=self.prior_scale,
                    prior_hidden_div=self.prior_hidden_div,
                ).to(self.device) for _ in range(self.ensemble_size)
            ])
        elif self.prior_scale > 0:
            self.state_heads = nn.ModuleList([
                MLPStateHeadWithPrior(
                    self.base_output_dim,
                    self.state_dim,
                    self.device,
                    self.architecture_config,
                    prior_scale=self.prior_scale,
                    prior_hidden_div=self.prior_hidden_div,
                ).to(self.device) for _ in range(self.ensemble_size)
            ])
        else:
            self.state_heads = nn.ModuleList([
                MLPStateHead(
                    self.base_output_dim,
                    self.state_dim,
                    self.device,
                    self.architecture_config
                ).to(self.device) for _ in range(self.ensemble_size)
            ])

        self.auxiliary_base = self._create_base(use_latent_dim=False)
        self.auxiliary_heads = nn.ModuleList([
            MLPAuxiliaryHead(
                self.base_output_dim,
                self.extension_dim,
                self.contact_dim,
                self.termination_dim,
                self.device,
                self.architecture_config
            ).to(self.device) for _ in range(self.ensemble_size)
        ])

        if self.freeze_auxiliary:
            for param in self.auxiliary_base.parameters():
                param.requires_grad = False
            for head in self.auxiliary_heads:
                for param in head.parameters():
                    param.requires_grad = False

    def _create_base(self, use_latent_dim: bool = True):
        """Create backbone network.
        
        Args:
            use_latent_dim: If True and latent_mode is active, use latent_dim
                for input dimensions. If False, always use state_dim.
                State base should use True, auxiliary base should use False.
        """
        base_state_dim = self.latent_dim if (self.latent_mode and use_latent_dim) else self.state_dim
        
        if self.architecture_config["type"] == "mlp":
            input_dim = self.history_horizon * (base_state_dim + self.action_dim)
            self.base_output_dim = self.architecture_config["base_shape"][-1]
            self.prediction_type = "single"
            return MLPBase(
                input_dim=input_dim,
                device=self.device,
                architecture_config=self.architecture_config,
            )
        elif self.architecture_config["type"] == "rnn":
            input_dim = base_state_dim + self.action_dim
            self.base_output_dim = self.architecture_config["rnn_hidden_size"]
            self.prediction_type = "single"
            return RNNBase(
                input_dim=input_dim,
                device=self.device,
                architecture_config=self.architecture_config
            )
        elif self.architecture_config["type"] == "xlstm":
            if not XLSTM_AVAILABLE:
                raise ImportError(
                    "xlstm library not found. Install with: pip install xlstm"
                )
            input_dim = base_state_dim + self.action_dim
            self.base_output_dim = self.architecture_config.get("xlstm_embedding_dim", 256)
            self.prediction_type = "single"
            return xLSTMBase(
                input_dim=input_dim,
                device=self.device,
                architecture_config=self.architecture_config
            )
        else:
            raise ValueError("Invalid architecture type.")

    def encode(self, raw_states: torch.Tensor) -> torch.Tensor:
        """Encode raw states to latent representation.
        
        Args:
            raw_states: Tensor of shape [..., state_dim].
            
        Returns:
            Latent states of shape [..., latent_dim] if latent_mode,
            otherwise returns raw_states unchanged.
        """
        if self.latent_mode and self.encoder is not None:
            return self.encoder(raw_states)
        return raw_states
    
    def decode(self, latent_states: torch.Tensor) -> torch.Tensor:
        """Decode latent states to raw state space.
        
        Args:
            latent_states: Tensor of shape [..., latent_dim].
            
        Returns:
            Raw states of shape [..., state_dim] if latent_mode,
            otherwise returns latent_states unchanged.
        """
        if self.latent_mode and self.decoder is not None:
            return self.decoder(latent_states)
        return latent_states

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None):
        """Update target encoder parameters via exponential moving average.
        
        θ_target ← momentum * θ_target + (1 - momentum) * θ_online
        
        Only has an effect in latent mode. No-op if target_encoder is None.
        
        Args:
            momentum: EMA momentum. If None, uses self.target_encoder_momentum.
        """
        if self.target_encoder is None:
            return
        if momentum is None:
            momentum = self.target_encoder_momentum
        for param_online, param_target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_target.data.mul_(momentum).add_(param_online.data, alpha=1.0 - momentum)

    def forward(self, x_state_batch, x_action_batch, model_ids=None):
        """Forward pass through the ensemble.
        
        In latent mode:
        - Input states are encoded to latent space
        - Base processes (latent, action) pairs
        - Heads predict next latent state (with SimNorm)
        - Output is decoded back to raw state space
        
        In raw mode: unchanged from original behavior.
        
        Args:
            x_state_batch: Raw state history [batch, horizon, state_dim].
            x_action_batch: Action history [batch, horizon, action_dim].
            model_ids: Optional ensemble member selection [1, batch, 1].
            
        Returns:
            Tuple of (state_means, aleatoric_unc, epistemic_unc, extensions, contacts, terminations)
            where state_means are in RAW state space (decoded if latent mode).
        """
        state_means, state_stds, extensions, contacts, terminations = [], [], [], [], []
        
        # Encode states if latent mode
        if self.latent_mode:
            x_latent_batch = self.encode(x_state_batch)
            state_base_output = self.state_base(x_latent_batch, x_action_batch)
        else:
            state_base_output = self.state_base(x_state_batch, x_action_batch)
        
        for head in self.state_heads:
            state_mean, state_std = head(state_base_output, x_state_batch)
            if self.prediction_type == "sequence":
                state_mean = state_mean[:, -1]
                state_std = state_std[:, -1]
            state_means.append(state_mean.unsqueeze(0))
            state_stds.append(state_std.unsqueeze(0))

        # Auxiliary branch (always uses raw states for compatibility)
        auxiliary_base_output = self.auxiliary_base(x_state_batch, x_action_batch)
        for head in self.auxiliary_heads:
            extension, contact, termination = head(auxiliary_base_output, x_state_batch)
            if self.prediction_type == "sequence":
                extension = extension[:, -1] if extension is not None else None
                contact = contact[:, -1] if contact is not None else None
                termination = termination[:, -1] if termination is not None else None
            extensions.append(extension.unsqueeze(0) if extension is not None else None)
            contacts.append(contact.unsqueeze(0) if contact is not None else None)
            terminations.append(termination.unsqueeze(0) if termination is not None else None)

        state_means = torch.cat(state_means, dim=0)
        state_stds = torch.cat(state_stds, dim=0)
        extensions = torch.cat(extensions, dim=0) if self.extension_dim > 0 else None
        contacts = torch.cat(contacts, dim=0) if self.contact_dim > 0 else None
        terminations = torch.cat(terminations, dim=0) if self.termination_dim > 0 else None
        
        # In latent mode, state_means are in latent space at this point
        # Store for potential consistency loss computation
        if self.latent_mode:
            self._last_latent_predictions = state_means  # [E, B, latent_dim]
        
        if model_ids is None:
            if self.latent_mode:
                output_latent_means = state_means.mean(dim=0)
                output_state_means = self.decode(output_latent_means)
            else:
                output_state_means = state_means.mean(dim=0)
            output_extensions = extensions.mean(dim=0) if extensions is not None else None
            output_contacts = contacts.mean(dim=0) if contacts is not None else None
            output_terminations = terminations.mean(dim=0) if terminations is not None else None
        else:
            gather_dim = self.latent_dim if self.latent_mode else self.state_dim
            gathered = torch.gather(state_means, 0, model_ids.repeat(1, 1, gather_dim)).squeeze(0)
            if self.latent_mode:
                output_state_means = self.decode(gathered)
            else:
                output_state_means = gathered
            output_extensions = torch.gather(extensions, 0, model_ids.repeat(1, 1, self.extension_dim)).squeeze(0) if extensions is not None else None
            output_contacts = torch.gather(contacts, 0, model_ids.repeat(1, 1, self.contact_dim)).squeeze(0) if contacts is not None else None
            output_terminations = torch.gather(terminations, 0, model_ids.repeat(1, 1, self.termination_dim)).squeeze(0) if terminations is not None else None
        
        # Uncertainty computation
        if self.latent_mode:
            # Aleatoric: zero (latent heads don't predict std)
            aleatoric_uncertainty = torch.zeros(output_state_means.shape[0], device=self.device)
            # Epistemic: ensemble disagreement in latent space
            if self.ensemble_size > 1:
                if self.uncertainty_metric == "variance":
                    epistemic_uncertainty = state_means.var(dim=0).sum(dim=1)
                else:
                    epistemic_uncertainty = state_means.std(dim=0).sum(dim=1)
            else:
                epistemic_uncertainty = torch.zeros(output_state_means.shape[0], device=self.device)
        else:
            aleatoric_uncertainty = state_stds.mean(dim=0).sum(dim=1)
            if self.ensemble_size > 1:
                if self.uncertainty_metric == "variance":
                    epistemic_uncertainty = state_means.var(dim=0).sum(dim=1)
                else:
                    epistemic_uncertainty = state_means.std(dim=0).sum(dim=1)
            else:
                epistemic_uncertainty = torch.zeros(output_state_means.shape[0], device=self.device)
        
        return output_state_means, aleatoric_uncertainty, epistemic_uncertainty, output_extensions, output_contacts, output_terminations

    def forward_latent(self, x_latent_batch, x_action_batch, model_ids=None):
        """Forward pass that takes and returns latent states directly.
        
        Used during autoregressive rollouts in latent space to avoid
        repeated encode/decode cycles.
        
        Only available in latent mode. Raises error if called in raw mode.
        
        Args:
            x_latent_batch: Latent state history [batch, horizon, latent_dim].
            x_action_batch: Action history [batch, horizon, action_dim].
            model_ids: Optional ensemble member selection [1, batch, 1].
        
        Returns:
            Tuple of (latent_means, epistemic_uncertainty) where latent_means
            are in latent space (NOT decoded).
        """
        assert self.latent_mode, "forward_latent() can only be called in latent mode"
        
        latent_means = []
        state_base_output = self.state_base(x_latent_batch, x_action_batch)
        
        for head in self.state_heads:
            latent_mean, _ = head(state_base_output, x_latent_batch)
            if self.prediction_type == "sequence":
                latent_mean = latent_mean[:, -1]
            latent_means.append(latent_mean.unsqueeze(0))
        
        latent_means = torch.cat(latent_means, dim=0)
        
        if model_ids is None:
            output_latent_means = latent_means.mean(dim=0)
        else:
            output_latent_means = torch.gather(
                latent_means, 0, model_ids.repeat(1, 1, self.latent_dim)
            ).squeeze(0)
        
        if self.ensemble_size > 1:
            if self.uncertainty_metric == "variance":
                epistemic_uncertainty = latent_means.var(dim=0).sum(dim=1)
            else:
                epistemic_uncertainty = latent_means.std(dim=0).sum(dim=1)
        else:
            epistemic_uncertainty = torch.zeros(output_latent_means.shape[0], device=self.device)
        
        return output_latent_means, epistemic_uncertainty

    def compute_loss(self, state_batch, action_batch, extension_batch, contact_batch, termination_batch, bootstrap=False):
        state_losses = []
        sequence_losses = []
        bound_losses = []
        kl_losses = []
        consistency_losses = []
        reconstruction_losses = []
        encoder_consistency_losses = []
        extension_losses = []
        contact_losses = []
        termination_losses = []
        
        for i in range(self.ensemble_size):
            if bootstrap:
                ids = torch.randint(0, state_batch.shape[0], (state_batch.shape[0],), device=self.device)
            else:
                ids = torch.arange(0, state_batch.shape[0], device=self.device)
            
            state_loss, sequence_loss, bound_loss, kl_loss, consistency_loss, reconstruction_loss, encoder_consistency_loss = self.compute_state_loss(
                self.state_heads[i], state_batch[ids], action_batch[ids]
            )
            
            if self.auxiliary_heads is not None:
                extension_loss, contact_loss, termination_loss = self.compute_auxiliary_loss(
                    self.auxiliary_heads[i],
                    state_batch[ids],
                    action_batch[ids],
                    extension_batch[ids] if extension_batch is not None else None,
                    contact_batch[ids] if contact_batch is not None else None,
                    termination_batch[ids] if termination_batch is not None else None
                )
            else:
                extension_loss = torch.tensor(0.0, device=self.device)
                contact_loss = torch.tensor(0.0, device=self.device)
                termination_loss = torch.tensor(0.0, device=self.device)
            
            state_losses.append(state_loss.unsqueeze(0))
            sequence_losses.append(sequence_loss.unsqueeze(0))
            bound_losses.append(bound_loss.unsqueeze(0))
            kl_losses.append(kl_loss.unsqueeze(0))
            consistency_losses.append(consistency_loss.unsqueeze(0))
            reconstruction_losses.append(reconstruction_loss.unsqueeze(0))
            encoder_consistency_losses.append(encoder_consistency_loss.unsqueeze(0))
            extension_losses.append(extension_loss.unsqueeze(0))
            contact_losses.append(contact_loss.unsqueeze(0))
            termination_losses.append(termination_loss.unsqueeze(0))
        
        state_loss = torch.mean(torch.cat(state_losses, dim=0), dim=0)
        sequence_loss = torch.mean(torch.cat(sequence_losses, dim=0), dim=0)
        bound_loss = torch.mean(torch.cat(bound_losses, dim=0), dim=0)
        kl_loss = torch.mean(torch.cat(kl_losses, dim=0), dim=0)
        consistency_loss = torch.mean(torch.cat(consistency_losses, dim=0), dim=0)
        reconstruction_loss = torch.mean(torch.cat(reconstruction_losses, dim=0), dim=0)
        encoder_consistency_loss = torch.mean(torch.cat(encoder_consistency_losses, dim=0), dim=0)
        extension_loss = torch.mean(torch.cat(extension_losses, dim=0), dim=0)
        contact_loss = torch.mean(torch.cat(contact_losses, dim=0), dim=0)
        termination_loss = torch.mean(torch.cat(termination_losses, dim=0), dim=0)
        return state_loss, sequence_loss, bound_loss, kl_loss, consistency_loss, reconstruction_loss, encoder_consistency_loss, extension_loss, contact_loss, termination_loss

    def compute_state_loss(self, head, state_batch, action_batch):
        """Compute state prediction loss for a single ensemble head.
        
        In latent mode, this computes:
        - consistency_loss: MSE between predicted latent and EMA target encoder output
        - reconstruction_loss: MSE between decoded prediction and raw target
        - encoder_consistency_loss: (optional) MSE between online encoder output and 
            stop-gradient dynamics prediction (bidirectional, TD-MPC2 style)
        - state_loss: weighted sum of the above (for backward compatibility)
        
        In raw mode: unchanged from original behavior (consistency/reconstruction/encoder_consistency = 0).
        """
        forecast_horizon = state_batch.shape[1] - self.history_horizon
        state_losses = []
        sequence_losses = []
        bound_losses = []
        kl_losses = []
        consistency_losses = []
        reconstruction_losses = []
        encoder_consistency_losses = []
        
        if self.latent_mode:
            # Encode the full state batch with EMA target encoder for consistency targets
            with torch.no_grad():
                all_latent_targets = self.target_encoder(state_batch)
            
            # Encode initial history with online encoder (with gradients for encoder training)
            x_latent_batch = self.encoder(state_batch[:, :self.history_horizon])
        else:
            x_state_batch = state_batch[:, :self.history_horizon]
        
        for i in range(forecast_horizon):
            if self.latent_mode:
                # Target: EMA target encoder output (JEPA-style consistency)
                latent_target = all_latent_targets[:, self.history_horizon + i]
                raw_target = state_batch[:, self.history_horizon + i]
                
                if self.architecture_config["type"] in ["rnn", "rssm", "xlstm"] and i > 0:
                    x_action_batch = action_batch[:, self.history_horizon + i:self.history_horizon + i + 1]
                else:
                    x_action_batch = action_batch[:, i + 1:self.history_horizon + i + 1]
                
                # Forward through base + head in latent space
                base_output = self.state_base.forward(x_latent_batch, x_action_batch)
                latent_pred, _ = head.forward(base_output, x_latent_batch)
                
                # Consistency loss: MSE between dynamics prediction and EMA target
                consistency_loss = torch.sum(torch.square(latent_pred - latent_target), dim=1).mean(dim=0)
                
                # Reconstruction loss: decode predicted latent, compare to raw target
                raw_pred = self.decoder(latent_pred)
                reconstruction_loss = torch.sum(torch.square(raw_pred - raw_target), dim=1).mean(dim=0)
                
                # Bidirectional encoder consistency loss (TD-MPC2 style, optional)
                # Trains the online encoder to produce outputs consistent with dynamics predictions
                if self.encoder_consistency_coef > 0:
                    online_next_latent = self.encoder(state_batch[:, self.history_horizon + i])
                    encoder_consistency_loss = torch.sum(
                        torch.square(online_next_latent - latent_pred.detach()), dim=1
                    ).mean(dim=0)
                else:
                    encoder_consistency_loss = torch.tensor(0.0, device=self.device)
                
                # Combined state loss for backward compatibility
                state_loss = (
                    self.consistency_coef * consistency_loss 
                    + self.reconstruction_coef * reconstruction_loss
                    + self.encoder_consistency_coef * encoder_consistency_loss
                )
                sequence_loss = torch.tensor(0.0, device=self.device)
                bound_loss = torch.tensor(0.0, device=self.device)
                kl_loss = torch.tensor(0.0, device=self.device)
                
                consistency_losses.append(consistency_loss.unsqueeze(0))
                reconstruction_losses.append(reconstruction_loss.unsqueeze(0))
                encoder_consistency_losses.append(encoder_consistency_loss.unsqueeze(0))
                state_losses.append(state_loss.unsqueeze(0))
                sequence_losses.append(sequence_loss.unsqueeze(0))
                bound_losses.append(bound_loss.unsqueeze(0))
                kl_losses.append(kl_loss.unsqueeze(0))
                
                # Autoregressive: use predicted latent as next input
                if self.architecture_config["type"] in ["rnn", "rssm", "xlstm"]:
                    x_latent_batch = latent_pred.unsqueeze(1).detach()
                else:
                    x_latent_batch = torch.cat(
                        [x_latent_batch[:, 1:].clone(), latent_pred.unsqueeze(1).detach()],
                        dim=1
                    )
            else:
                # Original raw state mode (unchanged)
                if self.prediction_type == "single":
                    state_target = state_batch[:, self.history_horizon + i]
                elif self.prediction_type == "sequence":
                    state_target = state_batch[:, i + 1:self.history_horizon + i + 1]
                else:
                    raise ValueError("Invalid state prediction type.")
                
                if self.architecture_config["type"] in ["rnn", "rssm", "xlstm"] and i > 0:
                    x_action_batch = action_batch[:, self.history_horizon + i:self.history_horizon + i + 1]
                    if self.prediction_type == "sequence":
                        state_target = state_target[:, [-1]]
                else:
                    x_action_batch = action_batch[:, i + 1:self.history_horizon + i + 1]
                
                state_mean_pred, state_std_pred = head.forward(self.state_base.forward(x_state_batch, x_action_batch), x_state_batch)
                state_loss, sequence_loss = self.compute_regression_loss(state_mean_pred, state_std_pred, state_target)
                bound_loss = self.compute_bound_loss(head) if head.output_std else torch.tensor(0.0, device=self.device)
                kl_loss = self.state_base.kl_loss if self.architecture_config["type"] == "rssm" else torch.tensor(0.0, device=self.device)
                
                consistency_losses.append(torch.tensor(0.0, device=self.device).unsqueeze(0))
                reconstruction_losses.append(torch.tensor(0.0, device=self.device).unsqueeze(0))
                encoder_consistency_losses.append(torch.tensor(0.0, device=self.device).unsqueeze(0))
                state_losses.append(state_loss.unsqueeze(0))
                sequence_losses.append(sequence_loss.unsqueeze(0))
                bound_losses.append(bound_loss.unsqueeze(0))
                kl_losses.append(kl_loss.unsqueeze(0))
                
                if self.prediction_type == "sequence":
                    state_mean_pred = state_mean_pred[:, -1]
                    state_std_pred = state_std_pred[:, -1]
                
                if self.architecture_config["type"] in ["rnn", "rssm", "xlstm"]:
                    x_state_batch = (torch.randn_like(state_mean_pred, device=self.device) * state_std_pred + state_mean_pred).unsqueeze(1) if head.output_std else state_mean_pred.unsqueeze(1)
                else:
                    x_state_batch = torch.cat(
                        [
                            x_state_batch[:, 1:].clone(),
                            (torch.randn_like(state_mean_pred, device=self.device) * state_std_pred + state_mean_pred).unsqueeze(1) if head.output_std else state_mean_pred.unsqueeze(1),
                        ],
                        dim=1
                    )
        
        state_loss = torch.mean(torch.cat(state_losses, dim=0), dim=0)
        sequence_loss = torch.mean(torch.cat(sequence_losses, dim=0), dim=0)
        bound_loss = torch.mean(torch.cat(bound_losses, dim=0), dim=0)
        kl_loss = torch.mean(torch.cat(kl_losses, dim=0), dim=0)
        consistency_loss = torch.mean(torch.cat(consistency_losses, dim=0), dim=0)
        reconstruction_loss = torch.mean(torch.cat(reconstruction_losses, dim=0), dim=0)
        encoder_consistency_loss = torch.mean(torch.cat(encoder_consistency_losses, dim=0), dim=0)
        return state_loss, sequence_loss, bound_loss, kl_loss, consistency_loss, reconstruction_loss, encoder_consistency_loss

    def compute_auxiliary_loss(self, head, state_batch, action_batch, extension_batch, contact_batch, termination_batch):
        """Compute auxiliary prediction losses.
        
        Note: Auxiliary branch always operates in raw state space, even in latent mode.
        This is because auxiliary predictions (extension, contact, termination) are
        physical quantities that don't benefit from latent encoding.
        """
        forecast_horizon = state_batch.shape[1] - self.history_horizon
        x_state_batch = state_batch[:, :self.history_horizon]
        extension_losses = []
        contact_losses = []
        termination_losses = []
        
        for i in range(forecast_horizon):
            extension_target = extension_batch[:, self.history_horizon + i] if extension_batch is not None else None
            contact_target = contact_batch[:, self.history_horizon + i] if contact_batch is not None else None
            termination_target = termination_batch[:, self.history_horizon + i] if termination_batch is not None else None
            
            if self.architecture_config["type"] in ["rnn", "rssm", "xlstm"] and i > 0:
                x_action_batch = action_batch[:, self.history_horizon + i:self.history_horizon + i + 1]
            else:
                x_action_batch = action_batch[:, i + 1:self.history_horizon + i + 1]
            
            extension_pred, contact_pred, termination_pred = head.forward(self.auxiliary_base.forward(x_state_batch, x_action_batch), x_state_batch)
            
            extension_loss = self.compute_extension_loss(extension_pred, extension_target) if self.extension_dim > 0 else torch.tensor(0.0, device=self.device)
            contact_loss = self.compute_contact_loss(contact_pred, contact_target) if self.contact_dim > 0 else torch.tensor(0.0, device=self.device)
            termination_loss = self.compute_termination_loss(termination_pred, termination_target) if self.termination_dim > 0 else torch.tensor(0.0, device=self.device)
            
            extension_losses.append(extension_loss.unsqueeze(0))
            contact_losses.append(contact_loss.unsqueeze(0))
            termination_losses.append(termination_loss.unsqueeze(0))
            
            if self.architecture_config["type"] in ["rnn", "rssm", "xlstm"]:
                x_state_batch = state_batch[:, self.history_horizon + i:self.history_horizon + i + 1]
            else:
                x_state_batch = torch.cat([x_state_batch[:, 1:].clone(), state_batch[:, self.history_horizon + i:self.history_horizon + i + 1]], dim=1)
        
        extension_loss = torch.mean(torch.cat(extension_losses, dim=0), dim=0)
        contact_loss = torch.mean(torch.cat(contact_losses, dim=0), dim=0)
        termination_loss = torch.mean(torch.cat(termination_losses, dim=0), dim=0)
        return extension_loss, contact_loss, termination_loss

    def compute_regression_loss(self, state_mean_pred, state_std_pred, state_target, loss_type="mse"):
        if loss_type == "mse":
            if self.prediction_type == "sequence":
                state_mean_pred_seq, state_mean_pred = state_mean_pred[:, :-1], state_mean_pred[:, -1]
                state_std_pred_seq, state_std_pred = state_std_pred[:, :-1], state_std_pred[:, -1]
                state_target_seq, state_target = state_target[:, :-1], state_target[:, -1]
                state_pred_seq = torch.randn_like(state_mean_pred_seq, device=self.device) * state_std_pred_seq + state_mean_pred_seq
                state_pred_seq = state_pred_seq.flatten(0, 1)
                state_target_seq = state_target_seq.flatten(0, 1)
                sequence_loss = torch.sum(torch.square(state_pred_seq - state_target_seq), dim=1).mean(dim=0)
            else:
                sequence_loss = torch.tensor(0.0, device=self.device)
            state_pred = torch.randn_like(state_mean_pred, device=self.device) * state_std_pred + state_mean_pred
            state_loss = torch.sum(torch.square(state_pred - state_target), dim=1).mean(dim=0)
            return state_loss, sequence_loss
        elif loss_type == "gaussian_nll":
            if self.prediction_type == "sequence":
                state_mean_pred_seq, state_mean_pred = state_mean_pred[:, :-1], state_mean_pred[:, -1]
                state_std_pred_seq, state_std_pred = state_std_pred[:, :-1], state_std_pred[:, -1]
                state_target_seq, state_target = state_target[:, :-1], state_target[:, -1]
                state_mean_pred_seq = state_mean_pred_seq.flatten(0, 1)
                state_std_pred_seq = state_std_pred_seq.flatten(0, 1)
                state_target_seq = state_target_seq.flatten(0, 1)
                sequence_loss = nn.GaussianNLLLoss()(state_mean_pred_seq, state_target_seq, state_std_pred_seq ** 2)
            else:
                sequence_loss = torch.tensor(0.0, device=self.device)
            state_loss = nn.GaussianNLLLoss()(state_mean_pred, state_target, state_std_pred ** 2)
            return state_loss, sequence_loss
        else:
            raise ValueError("Invalid loss type.")
        
    def compute_bound_loss(self, head):
        return torch.mean(head.state_max_logstd) - torch.mean(head.state_min_logstd)

    def compute_extension_loss(self, extension_pred, extension_target):
        if extension_pred is None or extension_target is None:
            return torch.tensor(0.0, device=self.device)
        if self.prediction_type == "sequence":
            extension_pred = extension_pred[:, -1]
        return nn.MSELoss()(extension_pred, extension_target)
    
    def compute_contact_loss(self, contact_pred, contact_target):
        if contact_pred is None or contact_target is None:
            return torch.tensor(0.0, device=self.device)
        if self.prediction_type == "sequence":
            contact_pred = contact_pred[:, -1]
        return nn.BCEWithLogitsLoss()(contact_pred, contact_target)
    
    def compute_termination_loss(self, termination_pred, termination_target):
        if termination_pred is None or termination_target is None:
            return torch.tensor(0.0, device=self.device)
        if self.prediction_type == "sequence":
            termination_pred = termination_pred[:, -1]
        return nn.BCEWithLogitsLoss()(termination_pred, termination_target)

    def reset(self):
        self.state_base.reset()
        for head in self.state_heads:
            head.reset()
        if self.auxiliary_base is not None:
            self.auxiliary_base.reset()
            for head in self.auxiliary_heads:
                head.reset()
    
    def reset_partial(self, env_ids):
        """Reset hidden states for specific environment indices."""
        self.state_base.reset_partial(env_ids)
        for head in self.state_heads:
            if hasattr(head, 'reset_partial'):
                head.reset_partial(env_ids)
        if self.auxiliary_base is not None:
            self.auxiliary_base.reset_partial(env_ids)
            for head in self.auxiliary_heads:
                if hasattr(head, 'reset_partial'):
                    head.reset_partial(env_ids)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_auxiliary:
            self.auxiliary_base.eval()
            for head in self.auxiliary_heads:
                head.eval()
        # Target encoder should always be in eval mode (no dropout/batchnorm train behavior)
        if self.target_encoder is not None:
            self.target_encoder.eval()
        return self
