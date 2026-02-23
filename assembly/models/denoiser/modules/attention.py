"""
Adapted from puzzlefusion++
https://github.com/eric-zqwang/puzzlefusion-plusplus
"""

from typing import Optional

import torch
import torch.nn as nn
import flash_attn
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import Timesteps, TimestepEmbedding


class MyAdaLayerNorm(nn.Module):
    """
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.timestep_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embbedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )
        # self.emb = nn.Linear(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.linear(
            self.silu(self.timestep_embbedder(self.timestep_proj(timestep)))
        )  # (n_points, embedding_dim * 2)
        # (valid_P, embedding_dim), (valid_P, embedding_dim)
        scale, shift = emb.chunk(2, dim=1)
        # broadcast to (n_points, embedding_dim)
        scale = scale[batch]
        shift = shift[batch]

        return self.norm(x) * (1 + scale) + shift


class EncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: int = None,
        attention_bias: bool = False,
        norm_elementwise_affine: bool = True,
        final_dropout: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        #  1. self attention
        self.norm1 = MyAdaLayerNorm(dim, num_embeds_ada_norm)

        self.self_attn_to_qkv = nn.Linear(dim, dim * 3, bias=attention_bias)
        self.self_attn_to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

        # 2. global attention
        self.norm2 = MyAdaLayerNorm(dim, num_embeds_ada_norm)

        self.global_attn_to_qkv = nn.Linear(dim, dim * 3, bias=attention_bias)
        self.global_attn_to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

        # 3. feed forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
        )

    def pad_sequence(
        self,
        x: torch.Tensor,
        seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
    ):
        seq_ranges = (
            torch.arange(max_seqlen, device=x.device)
            .unsqueeze(0)
            .expand(len(seqlens), -1)
        )
        valid_mask = seq_ranges < seqlens.unsqueeze(1)
        padded_x = torch.zeros(
            (seqlens.shape[0], max_seqlen, x.shape[-1]), device=x.device, dtype=x.dtype
        )
        padded_x[valid_mask] = x

        return padded_x, valid_mask

    def forward(
        self,
        hidden_states: torch.Tensor,  # (n_points, embed_dim)
        timestep: torch.Tensor,  # (valid_P,)
        batch: torch.Tensor,  # (valid_P,)
        self_attn_seqlens: torch.Tensor,
        self_attn_cu_seqlens: torch.Tensor,
        self_attn_max_seqlen: torch.Tensor,
        global_attn_seqlens: torch.Tensor,
        global_attn_cu_seqlens: torch.Tensor,
        global_attn_max_seqlen: torch.Tensor,
        # graph_mask: torch.Tensor,  # (B, global_attn_max_seqlen, global_attn_max_seqlen)
        coarse_seg_pred: Optional[torch.Tensor] = None,  # (n_points,)
    ):
        n_points, embed_dim = hidden_states.shape
        # we use ada_layer_norm
        # 1. self attention
        norm_hidden_states = self.norm1(hidden_states, timestep, batch)

        attn_output = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv=self.self_attn_to_qkv(norm_hidden_states).reshape(
                n_points, 3, self.num_attention_heads, self.attention_head_dim
            ),
            cu_seqlens=self_attn_cu_seqlens,
            max_seqlen=self_attn_max_seqlen,
            dropout_p=0.0,
        ).view(n_points, embed_dim)

        attn_output = self.self_attn_to_out(attn_output)
        hidden_states = hidden_states + attn_output

        if coarse_seg_pred is not None:
            hidden_states = hidden_states * (1 + coarse_seg_pred.unsqueeze(1))

        # 2. global attention
        norm_hidden_states = self.norm2(hidden_states, timestep, batch)

        global_out_flash = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv=self.global_attn_to_qkv(norm_hidden_states).reshape(
                n_points, 3, self.num_attention_heads, self.attention_head_dim
            ),
            cu_seqlens=global_attn_cu_seqlens,
            max_seqlen=global_attn_max_seqlen,
            dropout_p=0.0,
        ).view(n_points, embed_dim)
        global_out_flash = self.global_attn_to_out(global_out_flash)
        hidden_states = hidden_states + global_out_flash

        # 3. feed forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states

        return hidden_states  # (n_points, embed_dim)

    def forward_sdpa(
        self,
        hidden_states: torch.Tensor,    # (B, S, D) pre-padded
        timestep: torch.Tensor,         # (valid_P,)
        batch: torch.Tensor,            # (B, S) padded part indices
        self_attn_mask: torch.Tensor,   # (B, 1, S, S) bool mask
        global_attn_mask: torch.Tensor, # (B, 1, S, S) bool mask
        coarse_seg_pred: Optional[torch.Tensor] = None,  # (B, S) padded, or None
    ):
        B, S, D = hidden_states.shape

        # 1. self attention
        norm_hidden_states = self.norm1(hidden_states, timestep, batch)

        qkv = self.self_attn_to_qkv(norm_hidden_states).reshape(
            B, S, 3, self.num_attention_heads, self.attention_head_dim
        )
        q, k, v = qkv.unbind(dim=2)  # each (B, S, H, Dh)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=self_attn_mask
        )  # (B, H, S, Dh)
        attn_output = attn_output.transpose(1, 2).reshape(B, S, D)

        attn_output = self.self_attn_to_out(attn_output)
        hidden_states = hidden_states + attn_output

        if coarse_seg_pred is not None:
            hidden_states = hidden_states * (1 + coarse_seg_pred.unsqueeze(-1))

        # 2. global attention
        norm_hidden_states = self.norm2(hidden_states, timestep, batch)

        qkv = self.global_attn_to_qkv(norm_hidden_states).reshape(
            B, S, 3, self.num_attention_heads, self.attention_head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        global_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=global_attn_mask
        )
        global_out = global_out.transpose(1, 2).reshape(B, S, D)

        global_out = self.global_attn_to_out(global_out)
        hidden_states = hidden_states + global_out

        # 3. feed forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states

        return hidden_states  # (B, S, D)