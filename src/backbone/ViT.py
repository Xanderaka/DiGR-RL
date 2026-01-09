import torch
import torch.nn as nn

# Patch Embedding Module: Converts image into a sequence of flattened patches
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=1, embed_dim=128, dropout=0.3):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Linear projection of patches using Conv2d
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B = x.shape[0]  # Batch size
        x = self.proj(x)                         # [B, E, H/P, W/P]
        x = self.dropout(x)                      # Apply dropout after projection
        x = x.flatten(2).transpose(1, 2)         # Flatten patches: [B, N, E]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # Repeat class token for batch
        x = torch.cat((cls_tokens, x), dim=1)    # Concatenate class token: [B, N+1, E]
        x = x + self.pos_embed                   # Add positional embedding
        return x

# Transformer Encoder Module
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, hidden_dim=256, depth=6):
        super().__init__()
        # Create a standard Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True
        )
        # Stack multiple encoder layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
    def forward(self, x):
        return self.transformer(x)  # Input and output: [B, N+1, E]

# Vision Transformer Backbone
class ViTBackbone(nn.Module):
    def __init__(self,
                 input_dim=1,
                 latent_dim=120,
                 base_channels=128,
                 dropout=0.3,
                 patch_size=8,
                 img_size=64,
                 vit_depth=6,
                 vit_heads=4,
                 vit_hidden_dim=256,
                 ViT_ac_func=nn.ReLU,       
                 latent_ac_func=nn.Identity): 
        """Vision Transformer backbone with a classification-friendly API."""
        super().__init__()
        self.latent_dim = latent_dim
        
        # Patch embedding layer
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=input_dim,
            embed_dim=base_channels,
            dropout=dropout
        )
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            embed_dim=base_channels,
            num_heads=vit_heads,
            hidden_dim=vit_hidden_dim,
            depth=vit_depth
        )
        
        # Latent projection from transformer output to desired latent dimension
        self.latent_proj = nn.Sequential(
            ViT_ac_func(),            
            nn.Linear(base_channels, latent_dim),
            latent_ac_func()         
        )

    def forward(self, x):
        x = self.patch_embed(x)       # Convert image to patch embeddings [B, N+1, base_channels]
        x = self.transformer(x)       # Apply transformer layers
        cls_token = x[:, 0]           # Extract class token [B, base_channels]
        z = self.latent_proj(cls_token)  # Project class token to latent dimension [B, latent_dim]
        return z

if __name__ == "__main__":
    x = torch.randn(1, 3, 64, 64) 
    vit = ViTBackbone(
        input_dim=3,      
        latent_dim=120     
    )
    logits = vit(x)
    print(logits.shape)   
