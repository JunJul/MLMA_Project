import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torchmetrics.image import StructuralSimilarityIndexMeasure
from .CVAE import CondVariationalAutoencoder   # ensure this returns (reconstruction, kl)
from .perceptual_loss import PerceptualLoss


class CVAEModel(pl.LightningModule):
    """
    Conditional Variational Autoencoder for chest X-ray reconstruction.
    Supports uncertain labels via configurable policies and Beta-VAE training.
    """
    def __init__(
        self,
        latent_dims: int,
        n_classes: int = 14,
        meta_dims: int = 3,
        embedding_dims: int = 32,
        learning_rate: float = 0.001,
        policy: str = "U-Ones",
        mse_weight: float = 1.0,
        ssim_weight: float = 0.3,
        beta_max: float = 1.0,
        beta_warmup_epochs: int = 10,
        ssim_ramp_epochs: int = 10,
        beta_start: float = 1e-6,
        perceptual_weight: float = 0.1,       # add this
        perceptual_warmup_epochs: int = 5
    ):
        super().__init__()
        self.save_hyperparameters()  # saves all arguments for checkpointing

        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.policy = policy
        self.meta_dims = meta_dims
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.beta_max = beta_max
        # Ensure numeric types (YAML may parse numeric values as strings)
        self.beta_warmup_epochs = float(beta_warmup_epochs)
        self.ssim_ramp_epochs = float(ssim_ramp_epochs)
        self.beta_start = beta_start

        self.perceptual_weight = perceptual_weight
        self.perceptual_warmup_epochs = float(perceptual_warmup_epochs)
        self.perceptual_loss = PerceptualLoss()

        # Learnable embeddings
        self.embed_disease = nn.Embedding(num_embeddings=n_classes, embedding_dim=embedding_dims)
        self.embed_meta = nn.Linear(meta_dims, embedding_dims)

        # Conditional VAE (must return (reconstruction, kl_divergence))
        self.cvae = CondVariationalAutoencoder(latent_dims, embedding_dims)

        # SSIM metric for image quality
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def apply_policy(self, y: torch.Tensor) -> torch.Tensor:
        """
        Transform uncertain labels (-1) based on the selected policy.
        Policies: 'U-Zeros', 'U-Ones', 'U-Smooth' (maps to 0.55).
        """
        y_processed = y.clone().float()
        if self.policy == "U-Zeros":
            y_processed[y_processed == -1.0] = 0.0
        elif self.policy == "U-Ones":
            y_processed[y_processed == -1.0] = 1.0
        elif self.policy == "U-Smooth":
            y_processed[y_processed == -1.0] = 0.55
        return y_processed

    def get_multi_label_embedding(self, y_processed):
        # Ensure embedding weight is on the same device as y_processed
        weight = self.embed_disease.weight.to(y_processed.device)
        y_embed = torch.matmul(y_processed, weight)
        label_sum = y_processed.sum(dim=1, keepdim=True)
        y_embed = torch.where(
            label_sum > 0,
            y_embed / (label_sum + 1e-6),
            torch.zeros_like(y_embed)
        )
        return y_embed

    def forward(self, x: torch.Tensor, meta: torch.Tensor, y: torch.Tensor):
        """
        Forward pass through the CVAE.
        Returns: (reconstructed_image, kl_divergence)
        """
        y_processed = self.apply_policy(y)
        y_embed = self.get_multi_label_embedding(y_processed)

        if meta is None:
            meta = torch.zeros(x.size(0), self.meta_dims, device=x.device)
        meta_embed = torch.relu(self.embed_meta(meta.float()))

        # cvae must return (recon_x, kl)
        recon_x, kl = self.cvae(x, y_embed, meta_embed, epoch=self.current_epoch)
        return recon_x, kl

    def training_step(self, batch, batch_idx):
        x, meta, y = batch
        x_hat, kl = self(x, meta, y)

        raw_mse = F.mse_loss(x_hat, x, reduction='mean')
        x_hat_clamped = torch.clamp(x_hat, min=1e-6, max=1.0 - 1e-6)
        raw_ssim_score = self.ssim(x_hat_clamped, x)
        raw_ssim_loss = 1.0 - raw_ssim_score

        ramp_multiplier = min(1.0, self.current_epoch / self.ssim_ramp_epochs)
        ssim_loss = raw_ssim_loss * ramp_multiplier

        beta = self.beta_start + (self.beta_max - self.beta_start) * min(
            1.0, self.current_epoch / self.beta_warmup_epochs
        )

        # Perceptual loss — ramped in after a few epochs so MSE stabilises first
        p_ramp = min(1.0, self.current_epoch / self.perceptual_warmup_epochs)
        p_loss = self.perceptual_loss(x_hat, x) * p_ramp

        loss = (
            self.mse_weight * raw_mse +
            self.ssim_weight * ssim_loss +
            self.perceptual_weight * p_loss +   # add this
            beta * kl
        )

        if torch.isnan(loss):
            self.log('warning_nan_loss', 1.0, prog_bar=True)

        self.log('epoch_num', float(self.current_epoch), prog_bar=False)
        self.log('beta', beta, prog_bar=False)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_ssim', raw_ssim_score, prog_bar=False)
        self.log('train_mse', raw_mse, prog_bar=False)
        self.log('train_ssim_loss', raw_ssim_loss, prog_bar=False)
        self.log('train_kl', kl, prog_bar=False)
        self.log('train_perceptual', p_loss, prog_bar=False)   # add this

        return loss

    def validation_step(self, batch, batch_idx):
        x, meta, y = batch
        x_hat, kl = self(x, meta, y)

        mse_recon_loss = F.mse_loss(x_hat, x, reduction='mean')
        x_hat_clamped = torch.clamp(x_hat, min=1e-6, max=1.0 - 1e-6)
        ssim_score = self.ssim(x_hat_clamped, x)

        self.log('val_mse', mse_recon_loss, prog_bar=True)
        self.log('val_kl', kl, prog_bar=True)
        self.log('val_ssim_score', ssim_score, prog_bar=True)

        return mse_recon_loss   # Lightning uses this for epoch‑level aggregation

    def test_step(self, batch, batch_idx):
        x, meta, y = batch
        x_hat, kl = self(x, meta, y)

        mse_recon_loss = F.mse_loss(x_hat, x, reduction='mean')
        x_hat_clamped = torch.clamp(x_hat, min=1e-6, max=1.0 - 1e-6)
        ssim_score = self.ssim(x_hat_clamped, x)

        self.log('test_mse', mse_recon_loss, prog_bar=True)
        self.log('test_kl', kl, prog_bar=False)
        self.log('test_ssim_score', ssim_score, prog_bar=True)

        return mse_recon_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_ssim_score",
                "interval": "epoch",
                "frequency": 1,
            }
        }

    # Optional: use Lightning’s built‑in gradient clipping instead of manual method
    # If you prefer manual clipping, keep the method below. Otherwise, set
    # Trainer(gradient_clip_val=1.0) in your main script.
    # def on_before_optimizer_step(self, optimizer, optimizer_idx=None):
    #     torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)