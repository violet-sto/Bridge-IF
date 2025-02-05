from copy import deepcopy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from esm.modules import ESM1bLayerNorm, gelu


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

    
class ConditionalLayerNorm(ESM1bLayerNorm):
    def __init__(self, normalized_shape, condition_size, eps=1e-5):
        super().__init__(normalized_shape, eps=eps)

        self.condition_size = condition_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_size, 2 * condition_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, condition):
        output = torch.zeros_like(x)
        for idx, c in enumerate(condition):
            gamma, beta = self.adaLN_modulation(c).chunk(2)
            # weight = (1 + gamma) * self.weight + beta
            # bias = self.bias
            
            # adaLN-Bias
            weight = self.weight + gamma
            bias = self.bias + beta
            
            # adaLN
            # weight = gamma
            # bias = beta
            
            output[:,idx] = F.layer_norm(x[:,idx], self.normalized_shape, weight, bias, self.eps)
        return output
    
    
class MyRobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, weight):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = ConditionalLayerNorm(embed_dim, embed_dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, c):
        x = self.dense(features)
        x = gelu(x)
        x = x.transpose(0, 1)
        x = self.layer_norm(x, c)
        x = x.transpose(0, 1)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x