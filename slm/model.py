"""GPT-style Transformer: explicit causal attention, pre-norm residual blocks."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 256  # UTF-8 bytes, so any text can be encoded
    context: int = 64
    width: int = 64
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1

    def __post_init__(self):
        if min(self.vocab_size, self.context, self.width, self.heads, self.layers) <= 0:
            raise ValueError('model dimensions must be positive')
        if self.width % self.heads:
            raise ValueError('width must be divisible by heads')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0, 1)')


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.heads
        self.qkv = nn.Linear(c.width, 3 * c.width)
        self.projection = nn.Linear(c.width, c.width)
        self.dropout = nn.Dropout(c.dropout)
        self.register_buffer('mask', torch.tril(torch.ones(c.context, c.context, dtype=torch.bool)))

    def forward(self, x):
        b, t, w = x.shape
        # Each head gets a separate query, key and value vector at each position.
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [a.view(b, t, self.heads, w // self.heads).transpose(1, 2) for a in (q, k, v)]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(w // self.heads)
        scores = scores.masked_fill(~self.mask[:t, :t], float('-inf'))
        weights = self.dropout(F.softmax(scores, dim=-1))
        out = (weights @ v).transpose(1, 2).contiguous().view(b, t, w)
        return self.dropout(self.projection(out))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(c.width), nn.LayerNorm(c.width)
        self.attention = Attention(c)
        self.mlp = nn.Sequential(nn.Linear(c.width, 4*c.width), nn.GELU(),
                                 nn.Linear(4*c.width, c.width), nn.Dropout(c.dropout))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.width)
        self.positions = nn.Embedding(config.context, config.width)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.layers)))
        self.norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, config.vocab_size, bias=False)

    def forward(self, ids, targets=None):
        if ids.ndim != 2 or not 0 < ids.shape[1] <= self.config.context:
            raise ValueError('expected [batch, time] with 1 <= time <= context')
        x = self.tokens(ids) + self.positions(torch.arange(ids.shape[1], device=ids.device))
        logits = self.output(self.norm(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=100, temperature=0.8, top_k=40):
        if temperature <= 0 or max_new_tokens < 0 or (top_k is not None and top_k <= 0):
            raise ValueError('temperature/top_k must be positive; token count nonnegative')
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                logits, _ = self(ids[:, -self.config.context:])
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    cutoff = logits.topk(min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < cutoff, float('-inf'))
                next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
                ids = torch.cat((ids, next_id), dim=1)
            return ids
        finally:
            self.train(was_training)
