"""
ra_lwlm_icl.py
=====================================================================
RA-LWLM 的上下文学习模块（单个专家）/ The in-context learning module of RA-LWLM
(one expert).

中文
----
输入：query 的 CSI 特征 + 场景配置，以及检索回来的 k 个 (CSI 特征, 坐标, 配置)
      和它们的检索权重 w = softmax(-距离)。
输出：query 的二维坐标。

两个关键设计：
  1. 加权质心 —— 质心 = Σ_i w_i · pos_i，越近的参考点权重越大；
  2. 残差读出 —— 网络只回归相对质心的偏移（乘 pos_scale 还原成米）。
     这让模块与场景的绝对坐标系解耦，换到没见过的场景仍然可用；
     直接回归绝对坐标在跨场景时会失效。

同一套参数可以吃任意 k ∈ {1, ..., K_max}：训练时随机丢弃 20% 参考 token
（见 modules.ReasoningTransformer），迫使模型不依赖固定的上下文长度。

EN
--
Inputs: the query CSI feature and scene config, plus the k retrieved
(CSI feature, position, config) triplets and their retrieval weights
w = softmax(-distance). Output: the 2-D position of the query.

Two design choices carry the method:
  1. weighted centroid — centroid = Σ_i w_i · pos_i, so closer references count more;
  2. residual read-out — the network only regresses the offset from that centroid
     (scaled back to metres by pos_scale). This decouples the module from the
     absolute coordinate frame of any scene, which is what lets it transfer to
     unseen scenes; regressing absolute positions does not transfer.

One set of weights serves any k in {1, ..., K_max}: 20% of the reference tokens are
dropped at random during training (see modules.ReasoningTransformer), so the model
cannot rely on a fixed context length.
"""
import torch
import torch.nn as nn

from modules import PosEncoder, CfgEncoder, ReasoningTransformer


class RA_LWLM_ICL(nn.Module):
    """检索增强的上下文定位模块 / retrieval-augmented in-context localisation module."""

    def __init__(self, embed_dim=256, K_max=20,
                 num_heads=4, num_layers=2, dropout=0.1,
                 pos_scale=32.0, token_dim=None, cfg_dim=4):
        super().__init__()

        self.embed_dim  = embed_dim
        self.token_dim  = token_dim if token_dim is not None else embed_dim
        self.K_max      = K_max
        self._pos_scale = pos_scale

        # query/参考的 CSI 特征 [LST; patch 均值] (2*embed) → token_dim
        # Query/reference CSI feature [LST; patch mean] (2*embed) -> token_dim
        self.feature_proj = nn.Linear(embed_dim * 2, self.token_dim)

        # 坐标（相对质心、已归一化）→ token
        # Position (relative to the centroid, normalised) -> token
        self.pos_encoder = PosEncoder(pos_dim=2, hidden_dim=128, embed_dim=self.token_dim)

        # 场景配置 → token。cfg = [方位角, 带宽, 天线数, BS 高度]，由调用方归一化好
        # Scene config -> token. cfg = [azimuth, bandwidth, n_ant, bs_height],
        # already normalised by the caller; no further normalisation happens here.
        self.cfg_encoder = CfgEncoder(cfg_dim=cfg_dim, embed_dim=self.token_dim)

        # 上下文推理 Transformer / the in-context reasoning Transformer
        self.icl_tf = ReasoningTransformer(
            embed_dim=self.token_dim, num_heads=num_heads,
            num_layers=num_layers, dropout=dropout)

        # 定位头：输出相对质心的残差（乘 pos_scale 还原成米）
        # Localisation head: the residual w.r.t. the centroid (times pos_scale -> metres)
        self.loc_head = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.ReLU(),
            nn.Linear(self.token_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    @property
    def POS_SCALE(self):
        return self._pos_scale

    def _build_ref_tokens(self, ref_features, ref_positions, ref_configs, centroid):
        """参考 token = CSI 特征 + 配置编码 + 位置编码（坐标取相对质心的偏移）。
        Reference token = CSI feature + config embedding + position embedding, where
        the position is the offset relative to the centroid."""
        tokens = self.feature_proj(ref_features)
        tokens = tokens + self.cfg_encoder(ref_configs)
        pos_input = (ref_positions - centroid.unsqueeze(1)) / self._pos_scale
        if self.training:
            # 轻微位置抖动，防止模型死记参考点坐标 / small jitter so the model cannot memorise positions
            pos_input = pos_input + torch.randn_like(pos_input) * 0.02
        tokens = tokens + self.pos_encoder(pos_input)
        return tokens

    def _build_query_token(self, query_features, query_config):
        """query token = CSI 特征 + query 标记 + 配置编码（不含任何真实坐标）。
        Query token = CSI feature + query marker + config embedding; no ground-truth
        position ever enters here."""
        token = self.feature_proj(query_features)
        token = token + self.icl_tf.query_pos_embed.squeeze(0).squeeze(0)
        token = token + self.cfg_encoder(query_config)
        return token

    def forward_loc(self, query_features, query_config,
                    ref_features, ref_positions, ref_configs, weights, k=None):
        """用前 k 个检索结果定位 / localise from the first k retrieved references.

        query_features [B, 2*embed]       query 的 [LST; patch 均值]
        query_config   [B, cfg_dim]       场景配置（已归一化）
        ref_features   [B, K_max, 2*embed]
        ref_positions  [B, K_max, 2]      单位：米 / in metres
        ref_configs    [B, K_max, cfg_dim]
        weights        [B, K_max]         检索权重 softmax(-距离) / retrieval weights
        k              使用的上下文长度，默认 K_max / context length, defaults to K_max
        返回 / returns [B, 2] 坐标（米）/ positions in metres
        """
        if k is None:
            k = self.K_max

        ref_feat_k = ref_features[:, :k, :]
        ref_pos_k  = ref_positions[:, :k, :]
        ref_cfg_k  = ref_configs[:, :k, :]
        w_k        = weights[:, :k]
        w_k        = w_k / (w_k.sum(dim=-1, keepdim=True) + 1e-8)

        # 加权质心 / weighted centroid: Σ_i w_i · pos_i
        centroid = torch.bmm(w_k.unsqueeze(1), ref_pos_k).squeeze(1)          # [B, 2] metres

        ref_tokens = self._build_ref_tokens(ref_feat_k, ref_pos_k, ref_cfg_k, centroid)
        query_tok  = self._build_query_token(query_features, query_config)

        query_out = self.icl_tf(ref_tokens, query_tok)                        # [B, D]
        delta     = self.loc_head(query_out)                                  # [B, 2]
        # 质心 + 残差 / centroid + residual
        return centroid + delta * self.POS_SCALE
