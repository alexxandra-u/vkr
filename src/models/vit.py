import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig

class VIT(nn.Module):

    def __init__(self, d_input, d_output, d_hidden, d_ff=None, n_channels=1, patch_size=16, n_heads=4, n_blocks=4):
        super().__init__()

        d_ff = d_hidden*2 if d_ff is None else d_ff
        self.model = ViTForImageClassification(ViTConfig(
            image_size=d_input,
            patch_size=patch_size,
            num_channels=n_channels,
            num_labels=d_output,
            hidden_size=d_hidden,
            intermediate_size=d_ff,
            num_hidden_layers=n_blocks,
            num_attention_heads=n_heads,
            return_dict=False
        ))

    def forward(self, X):
        return self.model(X)