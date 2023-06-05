import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class image_encoder(nn.Module):
    def __init__(self, in_features = 784, z_dim = 64, dim = 128, n_layer = 3):
        super().__init__()

        block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),            
            nn.GELU()
        )

        self.stem = nn.Linear(in_features, dim)
        self.blocks = nn.ModuleList([block for _ in range(n_layer)])
        self.to_mu = nn.Linear(dim, z_dim)
        self.to_log_var = nn.Linear(dim, z_dim)

    def encode(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = x + block(x)

        return self.to_mu(x), self.to_log_var(x)
    
    def sampling(self, mu, log_var):        
        std = torch.exp(0.5 * log_var)
        eps = torch.rand_like(std)
        return eps * std + mu
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.sampling(mu, log_var)
        return z, mu, log_var
    
class label_encoder(nn.Module):
    def __init__(self, num_label = 10, z_dim = 64, dim = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Embedding(num_label, dim),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU()
        )

        self.to_mu = nn.Linear(dim, z_dim)
        self.to_log_var = nn.Linear(dim, z_dim)

    def encode(self, x):
        x = self.net(x)
        return self.to_mu(x), self.to_log_var(x)
    
    def sampling(self, mu, log_var):        
        std = torch.exp(0.5 * log_var)
        eps = torch.rand_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.sampling(mu, log_var)
        return z, mu, log_var
    
class image_decoder(nn.Module):
    def __init__(self, out_features = 784, z_dim = 64, dim = 128, n_layer = 3):
        super().__init__()

        block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU()
        )

        self.stem = nn.Linear(z_dim, dim)
        self.blocks = nn.ModuleList([block for _ in range(n_layer)])
        self.out = nn.Linear(dim, out_features)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = x + block(x)
        x = self.out(x)
        x = F.sigmoid(x)
        return x
    
class label_decoder(nn.Module):
    def __init__(self, num_label = 10, z_dim = 64, dim = 128):
        super().__init__()

        self.stem = nn.Linear(z_dim, dim)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU()
        )
        self.out = nn.Linear(dim, num_label)

    def forward(self, x):
        x = self.stem(x)
        x = self.net(x)
        x = self.out(x)
        return x
    
class clip_vae(nn.Module):
    def __init__(self, in_features = 784, num_label = 10, z_dim = 64, dim = 128, n_layer = 3):
        super().__init__()
        self.image_encoder = image_encoder(in_features, z_dim, dim, n_layer)
        self.label_encoder = label_encoder(num_label, z_dim, dim)
        self.image_decoder = image_decoder(in_features, z_dim, dim, n_layer)
        self.label_decoder = label_decoder(num_label, z_dim, dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.ce_clip_image = nn.CrossEntropyLoss()
        self.ce_clip_label = nn.CrossEntropyLoss()
        self.ce_vae_label = nn.CrossEntropyLoss()

    def encode_image(self, x):
        z, mu, log_var = self.image_encoder(x)
        return z
    
    def encode_label(self, x):
        z, mu, log_var = self.label_encoder(x)
        return z
    
    def decode_image(self, x):
        return self.image_decoder(x)
    
    def decode_label(self, x):
        return self.label_decoder(x)
    
    def clip_loss(self, image, label):
        image_features = self.encode_image(image)
        label_features = self.encode_label(label)

        # normalized features
        image_features = image_features / image_features.norm(dim = 1, keepdim = True)
        label_features = label_features / label_features.norm(dim = 1, keepdim = True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ label_features.t()
        logits_per_label = logits_per_image.t()

        # loss
        ground_truth = torch.arange(len(image), dtype = torch.long, device = image.device)
        loss_image = self.ce_clip_image(logits_per_image, ground_truth)
        logg_label = self.ce_clip_label(logits_per_label, ground_truth)
        
        return (loss_image + logg_label) / 2
    
    def vae_loss(self, image, label):
        image_z, image_mu, image_log_var = self.image_encoder(image)
        image_d = self.decode_image(image_z)
        image_kld = -0.5 * torch.mean(1 + image_log_var - image_mu.pow(2) - image_log_var.exp())
        image_rec = F.binary_cross_entropy(image_d, image, reduction = "mean")

        label_z, label_mu, label_log_var = self.label_encoder(label)
        label_d = self.decode_label(label_z)
        label_kld = -0.5 * torch.mean(1 + label_log_var - label_mu.pow(2) - label_log_var.exp())
        label_ce = self.ce_vae_label(label_d, label)

        return image_rec + image_kld + label_ce + label_kld
    
    def loss(self, image, label):
        image = image.flatten(1).float()
        label = label.long()
        return self.clip_loss(image, label) + self.vae_loss(image, label)
    
def build_model(in_features = 784, num_label = 10, z_dim = 16, dim = 64, n_layer = 6):
    return clip_vae(
        in_features = in_features,
        num_label = num_label,
        z_dim = z_dim,
        dim = dim,
        n_layer = n_layer
    )