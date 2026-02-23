"""
Adapted from puzzlefusion++
https://github.com/eric-zqwang/puzzlefusion-plusplus
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_scatter
import pytorch3d.transforms as p3dt

from assembly.models.utils import PositionalEncoding, EmbedderNerf
from .attention import EncoderLayer


class DenoiserTransformer(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        dropout_rate: float,
        trans_out_dim: int,
        rot_out_dim: int,
        use_flash_attn: bool = True,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.trans_out_dim = trans_out_dim
        self.rot_out_dim = rot_out_dim
        self.use_flash_attn = use_flash_attn
        self.ref_part_emb = nn.Embedding(2, self.embed_dim)
        self.activation = nn.SiLU()

        num_embeds_ada_norm = 6 * self.embed_dim

        self.transformer_layers = nn.ModuleList(
            [
                EncoderLayer(
                    dim=self.embed_dim,
                    num_attention_heads=self.num_heads,
                    attention_head_dim=self.embed_dim // self.num_heads,
                    dropout=self.dropout_rate,
                    activation_fn="geglu",
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=False,
                    norm_elementwise_affine=True,
                    final_dropout=False,
                )
                for _ in range(self.num_layers)
            ]
        )

        multires = 10
        embed_kwargs = {
            "include_input": True,
            "input_dims": 7,
            "max_freq_log2": multires - 1,
            "num_freqs": multires,
            "log_sampling": True,
            "periodic_fns": [torch.sin, torch.cos],
        }

        embedder_obj = EmbedderNerf(**embed_kwargs)
        self.param_embedding = lambda x, eo=embedder_obj: eo.embed(x)

        embed_pos_kwargs = {
            "include_input": True,
            "input_dims": 3,
            "max_freq_log2": multires - 1,
            "num_freqs": multires,
            "log_sampling": True,
            "periodic_fns": [torch.sin, torch.cos],
        }
        embedder_pos = EmbedderNerf(**embed_pos_kwargs)
        # Pos embedding for positions of points xyz
        self.pos_embedding = lambda x, eo=embedder_pos: eo.embed(x)

        embed_normal_kwargs = {
            "include_input": True,
            "input_dims": 3,
            "max_freq_log2": multires - 1,
            "num_freqs": multires,
            "log_sampling": True,
            "periodic_fns": [torch.sin, torch.cos],
        }
        embedder_normal = EmbedderNerf(**embed_normal_kwargs)
        # Normal embedding for points
        self.normal_embedding = lambda x, eo=embedder_normal: eo.embed(x)

        embed_scale_kwargs = {
            "include_input": True,
            "input_dims": 1,
            "max_freq_log2": multires - 1,
            "num_freqs": multires,
            "log_sampling": True,
            "periodic_fns": [torch.sin, torch.cos],
        }
        embedder_scale = EmbedderNerf(**embed_scale_kwargs)
        self.scale_embedding = lambda x, eo=embedder_scale: eo.embed(x)

        self.shape_embedding = nn.Linear(
            self.in_dim
            + embedder_scale.out_dim
            + embedder_pos.out_dim
            + embedder_normal.out_dim,
            self.embed_dim,
        )

        self.param_fc = nn.Linear(embedder_obj.out_dim, self.embed_dim)

        # Pos encoding for indicating the sequence.
        # self.pos_encoding = PositionalEncoding(self.embed_dim)

        # mlp out for translation N, 256 -> N, trans_out_dim
        self.mlp_out_trans = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.SiLU(),
            nn.Linear(self.embed_dim // 2, self.trans_out_dim),
        )

        # mlp out for rotation N, 256 -> N, rot_out_dim
        self.mlp_out_rot = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.SiLU(),
            nn.Linear(self.embed_dim // 2, self.rot_out_dim),
        )

        # # mlp out for graph
        # self.mlp_out_graph = nn.Sequential(
        #     nn.Linear(self.embed_dim, self.embed_dim // 4),
        #     nn.SiLU(),
        #     nn.Linear(self.embed_dim // 4, self.embed_dim // 16),
        # )
        # self.graph_param = nn.Parameter(
        #     torch.randn(self.embed_dim // 16, self.embed_dim // 16)
        # )

    def _gen_cond(
        self,
        x,  # (valid_P, 7)
        latent,  # PointTransformer Point instance
        scale,  # (valid, 1)
    ):
        trans = x[..., :3]  # (valid_P, 3)
        rot = x[..., 3:]  # (valid_P, 4)
        trans_broadcasted = trans[latent["batch"]]  # (n_points, 3)
        rot_broadcasted = rot[latent["batch"]]  # (n_points, 4)
        rot_6d = p3dt.matrix_to_rotation_6d(p3dt.quaternion_to_matrix(rot))

        # pos encoding for super points' coordinates
        xyz = latent["coord"]  # (n_points, 3)
        xyz = p3dt.quaternion_apply(rot_broadcasted, xyz)  # (n_points, 3)
        xyz_pos_emb = self.pos_embedding(xyz)  # (n_points, pos_emb_dim=63)

        normal = latent["normal"]
        normal = p3dt.quaternion_apply(rot_broadcasted, normal)
        normal_emb = self.normal_embedding(normal)  # (n_points, normal_emb_dim=63)

        scale_emb = self.scale_embedding(scale)  # (valid_P, scale_emb_dim=21)
        scale_emb = scale_emb[latent["batch"]]  # (n_points, scale_emb_dim=21)

        concat_emb = torch.cat(
            (latent["feat"], xyz_pos_emb, normal_emb, scale_emb), dim=-1
        )  # (n_points, in_dim + scale_emb_dim + pos_emb_dim)
        shape_emb = self.shape_embedding(concat_emb)  # (n_points, embed_dim)

        x_emb = self.param_fc(self.param_embedding(x))  # (valid_P, embed_dim)
        return x_emb, shape_emb

    def _out(self, data_emb):
        trans = self.mlp_out_trans(data_emb)
        rots = self.mlp_out_rot(data_emb)

        return torch.cat([trans, rots], dim=-1)

    # def _out_graph(
    #     self,
    #     data_emb,
    #     part_valids,
    # ):
    #     # data_emb (valid_P, embed_dim)
    #     graph_emb = self.mlp_out_graph(data_emb)  # (valid_P, embed_dim//4)
    #     valid_P, E = graph_emb.shape
    #     B, P = part_valids.shape

    #     # recover to padded parts
    #     padded_graph_emb = torch.zeros(
    #         (B, P, E),
    #         device=graph_emb.device,
    #         dtype=graph_emb.dtype,
    #     )
    #     padded_graph_emb[part_valids] = graph_emb

    #     graph_pred = torch.matmul(
    #         padded_graph_emb,
    #         self.graph_param,
    #     )
    #     graph_pred = torch.matmul(
    #         graph_pred,
    #         padded_graph_emb.transpose(1, 2),
    #     )
    #     graph_pred /= E**0.5

    #     return graph_pred

    def _add_ref_part_emb(
        self,
        x_emb,  # (valid_P, embed_dim)
        ref_part,  # (valid_P,)
    ):
        # ref_part_emb.weight[0] for non-ref part
        # ref_part_emb.weight[1] for ref part

        valid_P = x_emb.shape[0]
        ref_part_emb = self.ref_part_emb.weight[0].repeat(valid_P, 1)
        ref_part_emb[ref_part.to(torch.bool)] = self.ref_part_emb.weight[1]

        x_emb = x_emb + ref_part_emb
        return x_emb

    def _gen_mask(self, B, N, L, part_valids):
        self_block = torch.ones(
            L, L, device=part_valids.device
        )  # Each L points should talk to each other
        self_mask = torch.block_diag(
            *([self_block] * N)
        )  # Create block diagonal tensor
        self_mask = self_mask.unsqueeze(0).repeat(
            B, 1, 1
        )  # Expand dimensions to [B, N*L, N*L]
        self_mask = self_mask.to(torch.bool)
        gen_mask = part_valids.unsqueeze(-1).repeat(1, 1, L).flatten(1, 2)
        gen_mask = gen_mask.to(torch.bool)

        return self_mask, gen_mask

    def calc_graph_mask(
        self,
        graph: torch.Tensor,  # (B, P, P)
        points_per_part: torch.Tensor,  # (B, P)
        max_seq_len: torch.Tensor,
    ):
        B, P = points_per_part.shape

        # 计算每个 part 的点的起始和结束索引
        cum_points = torch.cumsum(points_per_part, dim=1)  # (B, P)
        start_indices = cum_points - points_per_part  # (B, P)
        end_indices = cum_points  # (B, P)

        # 生成 part_to_points 的掩码
        point_indices = torch.arange(
            max_seq_len, device=points_per_part.device
        )  # (max_seq_len,)

        part_to_points = (
            point_indices.unsqueeze(0).unsqueeze(0) >= start_indices.unsqueeze(2)
        ) & (
            point_indices.unsqueeze(0).unsqueeze(0) < end_indices.unsqueeze(2)
        )  # (B, P, max_seq_len)

        # 找到每个点所属的 part
        part_for_points = torch.argmax(
            part_to_points.float(), dim=1
        )  # (B, max_seq_len)

        connected_parts = graph.gather(
            1, part_for_points.unsqueeze(-1).expand(B, max_seq_len, P)
        )  # (B, max_seq_len, P)

        # 通过 part_to_points 将连接的 parts 映射为具体点
        connected_points = torch.bmm(connected_parts.float(), part_to_points.float())
        valid_points_mask = part_to_points.any(dim=1).float()
        connected_points = (
            connected_points
            * valid_points_mask.unsqueeze(1)
            * valid_points_mask.unsqueeze(2)
        )

        graph_mask = connected_points - 1
        graph_mask[graph_mask < 0] = -torch.inf

        valid_mask = valid_points_mask.unsqueeze(1) * valid_points_mask.unsqueeze(
            2
        )  # (B, max_seq_len, max_seq_len)
        valid_mask = valid_mask - 1
        valid_mask[valid_mask < 0] = -torch.inf

        # return graph_mask
        return graph_mask, valid_mask

    def forward(
        self,
        x,  # (valid_P, 7)
        timesteps,  # (valid_P,)
        latent,  # PointTransformer Point instance
        part_valids,  # (B, P)
        scale,  # (valid_P, 1)
        ref_part,  # (valid_P,)
    ):
        if not self.use_flash_attn:
            return self.forward_sdpa(x, timesteps, latent, part_valids, scale, ref_part)

        # (valid_P, embed_dim), (n_points, embed_dim)
        x_emb, shape_emb = self._gen_cond(x, latent, scale)
        # (valid_P, embed_dim)
        x_emb = self._add_ref_part_emb(x_emb, ref_part)
        # broadcast x_emb to all points (n_points, embed_dim)
        x_emb = x_emb[latent["batch"]]

        # (n_points, embed_dim)
        data_emb = x_emb + shape_emb

        # self_mask, gen_mask = self._gen_mask(B, N, L, part_valids)
        self_attn_seqlen = torch.bincount(latent["batch"])  # (valid_P,)
        self_attn_max_seqlen = self_attn_seqlen.max()
        self_attn_cu_seqlens = nn.functional.pad(
            torch.cumsum(self_attn_seqlen, 0), (1, 0)
        ).to(torch.int32)

        points_per_part = torch.zeros_like(part_valids, dtype=self_attn_seqlen.dtype)
        points_per_part[part_valids] = self_attn_seqlen
        global_attn_seqlen = points_per_part.sum(1)
        global_attn_max_seqlen = global_attn_seqlen.max()
        global_attn_cu_seqlens = nn.functional.pad(
            torch.cumsum(global_attn_seqlen, 0), (1, 0)
        ).to(torch.int32)

        for i, layer in enumerate(self.transformer_layers):
            data_emb = layer(
                hidden_states=data_emb,
                timestep=timesteps,
                batch=latent["batch"],
                self_attn_seqlens=self_attn_seqlen,
                self_attn_cu_seqlens=self_attn_cu_seqlens,
                self_attn_max_seqlen=self_attn_max_seqlen,
                global_attn_seqlens=global_attn_seqlen,
                global_attn_cu_seqlens=global_attn_cu_seqlens,
                global_attn_max_seqlen=global_attn_max_seqlen,
            )

        # scatter to each part
        data_emb = torch_scatter.segment_csr(
            data_emb,
            self_attn_cu_seqlens.long(),
            reduce="mean",
        )  # (valid_P, embed_dim)

        # data_emb (B, N*L, C)
        out_trans_rots = self._out(data_emb)

        return {
            "pred": out_trans_rots,  # (valid_P, 7)
            "graph_pred": None,
        }

    def forward_sdpa(
        self,
        x,  # (valid_P, 7)
        timesteps,  # (valid_P,)
        latent,  # PointTransformer Point instance
        part_valids,  # (B, P)
        scale,  # (valid_P, 1)
        ref_part,  # (valid_P,)
    ):
        # (valid_P, embed_dim), (n_points, embed_dim)
        x_emb, shape_emb = self._gen_cond(x, latent, scale)
        x_emb = self._add_ref_part_emb(x_emb, ref_part)
        x_emb = x_emb[latent["batch"]]
        data_emb = x_emb + shape_emb  # (n_points, embed_dim)

        batch_idx = latent["batch"]  # (n_points,) flat part index
        device = data_emb.device

        # --- sequence lengths ---
        self_attn_seqlen = torch.bincount(batch_idx)  # (valid_P,)
        self_attn_cu_seqlens = nn.functional.pad(
            torch.cumsum(self_attn_seqlen, 0), (1, 0)
        ).to(torch.int32)

        points_per_part = torch.zeros_like(part_valids, dtype=self_attn_seqlen.dtype)
        points_per_part[part_valids] = self_attn_seqlen
        global_attn_seqlen = points_per_part.sum(1)  # (B,)
        global_max_seqlen = global_attn_seqlen.max().item()
        B = part_valids.shape[0]

        # --- pad once: (n_points, D) -> (B, S, D) ---
        seq_ranges = torch.arange(global_max_seqlen, device=device).unsqueeze(0).expand(B, -1)
        valid_mask = seq_ranges < global_attn_seqlen.unsqueeze(1)  # (B, S)

        padded_data = torch.zeros(
            (B, global_max_seqlen, self.embed_dim), device=device, dtype=data_emb.dtype
        )
        padded_data[valid_mask] = data_emb

        # pad batch (part indices) — fill 0 for padding (safe: masked out in attention)
        padded_batch = torch.zeros(
            (B, global_max_seqlen), device=device, dtype=torch.long
        )
        padded_batch[valid_mask] = batch_idx

        # --- build masks (once, reused by all layers) ---
        # self-attention: block-diagonal per part (same part index -> attend)
        # part indices are globally unique, so equality correctly groups same-part points
        self_attn_mask = (padded_batch.unsqueeze(2) == padded_batch.unsqueeze(1))  # (B, S, S)
        self_attn_mask = self_attn_mask & valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
        self_attn_mask = self_attn_mask.unsqueeze(1)  # (B, 1, S, S) broadcast over heads

        # global attention: all valid positions in same batch item attend to each other
        global_attn_mask = (valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)).unsqueeze(1)

        # --- transformer loop (all in padded format) ---
        for layer in self.transformer_layers:
            padded_data = layer.forward_sdpa(
                hidden_states=padded_data,
                timestep=timesteps,
                batch=padded_batch,
                self_attn_mask=self_attn_mask,
                global_attn_mask=global_attn_mask,
            )

        # --- unpad once: (B, S, D) -> (n_points, D) ---
        data_emb = padded_data[valid_mask]

        # scatter to each part (mean pool)
        data_emb = torch_scatter.segment_csr(
            data_emb,
            self_attn_cu_seqlens.long(),
            reduce="mean",
        )  # (valid_P, embed_dim)

        out_trans_rots = self._out(data_emb)

        return {
            "pred": out_trans_rots,  # (valid_P, 7)
            "graph_pred": None,
        }
