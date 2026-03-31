import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from .CVAE import CondVariationalAutoencoder


class CVAEModel(pl.LightningModule):
    def __init__(self, latent_dims, n_classes=14, meta_dims=3, embedding_dims=32, learning_rate=0.001, policy="U-Ones"):
        super().__init__()
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.policy = policy
        self.meta_dims = meta_dims
        
        # Learnable embeddings
        self.embed_disease = nn.Embedding(num_embeddings=n_classes, embedding_dim=embedding_dims)
        self.embed_meta = nn.Linear(meta_dims, embedding_dims)
        
        # CNN Autoencoder
        self.cvae = CondVariationalAutoencoder(latent_dims, embedding_dims)

    def apply_policy(self, y):
        """
        Transforms the -1 (uncertain) labels based on the chosen Stanford policy.
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
        """
        Uses matrix multiplication to instantly blend the disease embeddings 
        based on the processed label probabilities!
        """
        return torch.matmul(y_processed, self.embed_disease.weight)

    def forward(self, x, meta, y):
        # Apply policy to handle any -1s in the real data
        y_processed = self.apply_policy(y)

        # Generate embeddings for disease labels
        y_embed = self.get_multi_label_embedding(y_processed)

        if meta is None:
            meta = torch.zeros(x.size(0), self.meta_dims, device=x.device)
        meta_embed = torch.relu(self.embed_meta(meta.float()))

        # Pass into CVAE
        return self.cvae(x, y_embed, meta_embed)

    def training_step(self, batch, batch_idx):
        x, meta, y = batch 
        x_hat = self(x, meta, y)
        
        recon_loss = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)
        kl = self.cvae.encoder.kl / x.size(0)
        
        loss = recon_loss + kl
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_recon', recon_loss, prog_bar=False)
        self.log('train_kl', kl, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, meta, y = batch 
        x_hat = self(x, meta, y)
        recon_loss = F.mse_loss(x_hat, x, reduction='mean')
        kl = self.cvae.encoder.kl / x.size(0)
        val_loss = recon_loss + kl
        self.log('val_loss', val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, meta, y = batch 
        x_hat = self(x, meta, y)
        recon_loss = F.mse_loss(x_hat, x, reduction='mean')
        kl = self.cvae.encoder.kl / x.size(0)
        test_loss = recon_loss + kl
        self.log('test_loss', test_loss, prog_bar=True)
        return test_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)