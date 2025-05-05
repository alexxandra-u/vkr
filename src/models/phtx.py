import torch
import torch.nn as nn
import torch.nn.functional as F


class PersistentHomologyTransformer(nn.Module):
    
    def __init__(self, transform=None, d_in=4, d_out=2, d_model=16, d_hidden=32, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.linear_in = nn.Linear(d_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_hidden, dropout, batch_first=True, activation=F.gelu)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.linear_out = nn.Linear(d_model, d_out)
        
    def _masked_mean(self, X, mask):
        X_masked = X * ~mask.unsqueeze(-1)
        n_masks = torch.sum(~mask, axis=1)
        X_masked_mean = torch.sum(X_masked, axis=1) / n_masks.unsqueeze(-1) # torch.mean(X, axis=1)
        return X_masked_mean
    
    def _masked_max(self, X, mask):
        X_masked_max, _ = torch.max(X.masked_fill(mask.unsqueeze(-1), -torch.inf), axis=1)
        return X_masked_max
        
    def forward(self, X, mask):
        X = self.linear_in(X)
        X = self.encoder(X, src_key_padding_mask=mask)
        X = self._masked_mean(X, mask)
        # X_max = self._masked_max(X, mask)
        # X = torch.cat([X_mean, X_max], axis=1)
        X = self.linear_out(X)
        return X