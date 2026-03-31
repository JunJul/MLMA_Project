import torch
import torch.nn as nn

class CondVariationalEncoder(nn.Module):
    def __init__(self, latent_dims, condition_dims):
        super().__init__()
   
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        cnn_out_size = 200704 
        
        combined_size = cnn_out_size + condition_dims + condition_dims
        
        self.fc_mu = nn.Linear(combined_size, latent_dims)
        # predict log-variance for numerical stability
        self.fc_logvar = nn.Linear(combined_size, latent_dims)

        self.kl = 0.0

    def reparameterize(self, mu, logvar):
        # logvar is log(sigma^2); use 0.5*logvar to get log(sigma)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y_embed, meta_embed):
        x_features = self.cnn(x) 
        
        # Image, Disease, and Metadata together!
        combined = torch.cat((x_features, y_embed, meta_embed), dim=1)
        
        # Calculate Latent Space
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)

        z = self.reparameterize(mu, logvar)

        # KL divergence between N(mu, sigma^2) and N(0,1)
        # stable formula: 0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        self.kl = kl
        return z

class CondVariationalDecoder(nn.Module):
    def __init__(self, latent_dims, condition_dims):
        super().__init__()
        # Latent Vector (z) + Disease Embed + Meta Embed
        combined_size = latent_dims + condition_dims + condition_dims
        
        # Project it back up to a 2D shape (e.g., 64 channels of 56x56)
        self.fc = nn.Linear(combined_size, 200704) 
        
        # Upscale back to 224x224
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Sigmoid keeps pixel values between 0 and 1
        )

        pass

    def forward(self, z, y_embed, meta_embed):
        # Glue the Latent Space, Disease, and Metadata together!
        combined = torch.cat((z, y_embed, meta_embed), dim=1)
        
        # Expand into a flat vector
        x = torch.relu(self.fc(combined))
        
        # Reshape into an image map (Batch Size, Channels, Height, Width)
        x = x.view(-1, 64, 56, 56)
        
        # Generate the final X-ray
        return self.deconv(x)

class CondVariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, condition_dims):
        super().__init__()
        self.encoder = CondVariationalEncoder(latent_dims, condition_dims)
        self.decoder = CondVariationalDecoder(latent_dims, condition_dims)

        pass

    def forward(self, x, y_embed, meta_embed):
        z = self.encoder(x, y_embed, meta_embed)
        return self.decoder(z, y_embed, meta_embed)