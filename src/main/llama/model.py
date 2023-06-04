"""
Reduced scale single GPU LLaMa model with XFormers efficient attention and rotary embedding
Based off of original LLaMa model
"""

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

import xformers.ops as xops
from xformers.components.positional_embedding import RotaryEmbedding


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def xformers_attn(xq, xk, xv, is_causal):
    mask = xops.LowerTriangularMask() if is_causal else None
    return xops.memory_efficient_attention(
        xq, xk, xv, attn_bias=mask
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super(Attention, self).__init__()

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.in_proj = nn.Linear(
            args.dim,
            3 * args.n_heads * self.head_dim,
            bias=False
        )
        self.out_proj = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False
        )

        self.pos_embed = RotaryEmbedding(self.head_dim)
        self.attn_fn = xformers_attn

    def forward(self, x: torch.Tensor, is_causal: bool = False):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.in_proj(x).chunk(3, dim=-1)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = self.pos_embed(xq, xk)
        output = self.attn_fn(
            xq.to(xv.dtype),
            xk.to(xv.dtype),
            xv,
            is_causal=is_causal
        )

        output = output.view(bsz, seqlen, -1)

        return self.out_proj(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super(FeedForward, self).__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            in_features=dim,
            out_features=hidden_dim,
            bias=False
        )
        self.w2 = nn.Linear(
            in_features=hidden_dim,
            out_features=dim,
            bias=False
        )
        self.w3 = nn.Linear(
            in_features=dim,
            out_features=hidden_dim,
            bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super(TransformerBlock, self).__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, is_causal: bool = True):
        x_res = x + self.attention(self.attention_norm(x), is_causal)
        out = x_res + self.feed_forward(self.ffn_norm(x_res))
        return out


class XFormersTransformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.token_embeddings = nn.Embedding(
            num_embeddings=params.vocab_size,
            embedding_dim=params.dim
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(
            in_features=params.dim,
            out_features=params.vocab_size,
            bias=False
        )

    def forward(self, tokens: torch.Tensor, is_causal: bool = True):
        x = self.token_embeddings(tokens)
        for layer in self.layers:
            x = layer(x, is_causal)
        x = self.norm(x)
        output = self.output(x)  # compute logits for all instead of just last
        return output.float()
