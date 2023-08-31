import torch
import torch.nn as nn

class pyt_transf(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_patches=64, embed_dim=64, num_heads=2, num_classes=10):
        super(pyt_transf, self).__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        
        self.patch_embeddings = nn.Linear(patch_size * patch_size * 3, embed_dim)
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        self.transformer = nn.MultiheadAttention(embed_dim, num_heads)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Create patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(-1, self.num_patches, self.patch_size * self.patch_size * 3)
        x = self.patch_embeddings(x)

        # Add positional embeddings
        x = x + self.position_embeddings
        x = x.permute(1, 0, 2)

        # Transformer block
        x, _ = self.transformer(x, x, x)

        # Classification head
        x = x.mean(dim=0)
        x = self.classifier(x)
        
        return x