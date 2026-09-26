"""
modules.py
=====================================================================
RA-LWLM 的基础模块 / Building blocks of RA-LWLM.

PosEncoder            参考点坐标 → token / reference position → token
CfgEncoder            场景配置 → token / scene config → token
ReasoningTransformer  在 [ref_1..ref_K; query] 上做上下文推理 / in-context reasoning
"""

import torch
import torch.nn as nn


class PosEncoder(nn.Module):
    """二维坐标 → embed_dim 的 MLP。输入已按 pos_scale 归一化（通常是相对质心的偏移）。
    MLP mapping a 2-D position to embed_dim. The input is already normalised by
    pos_scale (typically the offset relative to the retrieval centroid)."""
    def __init__(self, pos_dim=2, hidden_dim=128, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, pos):
        return self.net(pos)


class CfgEncoder(nn.Module):
    """场景配置向量 → embed_dim。cfg = [方位角, 带宽, 天线数, BS 高度]（已归一化）。
    Scene config vector → embed_dim, with cfg = [azimuth, bandwidth, n_ant,
    bs_height] already normalised by the caller."""
    def __init__(self, cfg_dim=3, embed_dim=256):
        super().__init__()
        self.proj = nn.Linear(cfg_dim, embed_dim)

    def forward(self, cfg):
        return self.proj(cfg)


class ReasoningTransformer(nn.Module):
    """上下文推理：把 K 个参考 token 和 1 个 query token 一起送进 Transformer，
    取 query 位置的输出。训练时随机丢弃 20% 参考 token，迫使模型不依赖固定的 k。
    In-context reasoning: the K reference tokens and the query token go through a
    Transformer encoder and the query position is read out. During training 20% of
    the reference tokens are dropped at random so the model cannot rely on a fixed k."""
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2,
                 dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.type_ref_embed   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.type_query_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.query_pos_embed  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.type_ref_embed,   std=0.02)
        nn.init.normal_(self.type_query_embed, std=0.02)
        nn.init.normal_(self.query_pos_embed,  std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation="relu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, ref_tokens, query_token):
        """ref_tokens: [B, K, D], query_token: [B, D] -> [B, D]."""
        ref_tokens  = ref_tokens + self.type_ref_embed
        query_token = query_token + self.type_query_embed.squeeze(1)

        if self.training:
            B, K, _ = ref_tokens.shape
            keep = (torch.rand(B, K, 1, device=ref_tokens.device) > 0.2).float()
            ref_tokens = ref_tokens * keep

        tokens = torch.cat([ref_tokens, query_token.unsqueeze(1)], dim=1)
        out    = self.norm(self.transformer(tokens))
        return out[:, -1, :]
