import torch
import torch.nn as nn


class SimpleResBlock(nn.Module):
    """
    A lightweight block that helps the network 'remember' fine details
    (like ribs and edges) as it compresses and decompresses the image.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x
        out = self.leaky(self.conv1(x))
        out = self.conv2(out)
        return self.leaky(out + residual)


class CondVariationalEncoder(nn.Module):
    def __init__(self, latent_dims, condition_dims):
        super().__init__()

        self.cnn = nn.Sequential(
            # 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            SimpleResBlock(64),

            # 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            SimpleResBlock(128),

            # 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Flatten()
        )

        # 256 * 14 * 14 = 50176
        cnn_out_size = 50176
        combined_size = cnn_out_size + condition_dims + condition_dims

        self.fc_mu = nn.Linear(combined_size, latent_dims)
        self.fc_logvar = nn.Linear(combined_size, latent_dims)

        # Will hold the KL divergence for the last forward pass
        self.kl = torch.tensor(0.0)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-20, max=20)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y_embed, meta_embed, epoch=None):
        x_features = self.cnn(x)
        combined = torch.cat((x_features, y_embed, meta_embed), dim=1)

        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        z = self.reparameterize(mu, logvar)

        # Free bits KL (per dimension floor at 0.5 nats)
        logvar_clamped = torch.clamp(logvar, min=-20, max=20)
        kl_per_dim = -0.5 * (1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp())
        
        free_bits_lambda = 0.5
        
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits_lambda)
        kl = kl_per_dim.sum(dim=1).mean()   # scalar tensor

        self.kl = kl
        return z


class CondVariationalDecoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.fc = nn.Linear(latent_dims, 50176)   # maps z to 256*14*14

        self.deconv = nn.Sequential(
            SimpleResBlock(256),
            SimpleResBlock(256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            SimpleResBlock(128),
            SimpleResBlock(128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            SimpleResBlock(64),
            SimpleResBlock(64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = torch.relu(self.fc(z))
        x = x.view(-1, 256, 14, 14)
        return self.deconv(x)


class CondVariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, condition_dims):
        super().__init__()
        self.encoder = CondVariationalEncoder(latent_dims, condition_dims)
        self.decoder = CondVariationalDecoder(latent_dims)

    def forward(self, x, y_embed, meta_embed, epoch=None):
        z = self.encoder(x, y_embed, meta_embed, epoch=epoch)
        recon = self.decoder(z)
        kl = self.encoder.kl
        return recon, kl