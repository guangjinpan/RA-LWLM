"""
ra_lwlm_kmoe.py
=====================================================================
RA-LWLM 的最终方案：上下文长度专家混合（K-MoE）。
The final RA-LWLM scheme: a mixture of context-length experts (K-MoE).

中文
----
"该看几个参考点"本身是依赖 query 的：近邻密集且可靠时小 k 更准，NLOS 或参考稀疏
时需要更大的 k 来平均掉噪声。因此不固定 k，而是：
  * 对 k ∈ {3, 6, 9, 12, 15} 各训练一个独立的 ICL 专家；
  * selector 读入 query 特征、场景配置、检索权重与参考点的几何分布，输出 5 个 logits；
  * 最终坐标 = Σ_i softmax(logits)_i · pos_i，即按路由权重对 5 个专家的预测做混合。
可选的负载均衡辅助损失（Switch-Transformer 风格）防止 selector 塌缩到单个专家。

EN
--
How many references to attend to is itself query-dependent: a small k wins when the
neighbours are dense and reliable, a larger k averages away noise in NLOS or sparse
regions. So k is not fixed:
  * one independent ICL expert is trained per k in {3, 6, 9, 12, 15};
  * the selector reads the query feature, the scene config, the retrieval weights and
    the geometry of the retrieved references, and emits 5 logits;
  * the output is Σ_i softmax(logits)_i · pos_i, i.e. the experts' predictions mixed
    by the routing weights.
An optional Switch-Transformer-style load-balancing loss keeps the router from
collapsing onto a single expert.
"""
import torch
import torch.nn as nn

from ra_lwlm_icl import RA_LWLM_ICL
from selector import KSelector


# 5 个专家对应的上下文长度 / the context length of each of the five experts
K_ACTIONS = [3, 6, 9, 12, 15]


def load_balance_loss(gate: torch.Tensor) -> torch.Tensor:
    """负载均衡辅助损失（Switch-Transformer）：让 query 均匀分散到各专家。
    最小值 1.0（完全均匀），最大值 N（全部塌缩到一个专家）。

    Switch-Transformer load-balance auxiliary loss, encouraging queries to spread
    across experts. It equals 1.0 for a perfectly uniform router and N when every
    query collapses onto a single expert.

    L_lb = N · Σ_i (f_i · P_i)
        f_i = 路由到专家 i 的 query 比例（按 argmax 计数）/ fraction routed to expert i
        P_i = 专家 i 的平均软权重 / mean soft gate weight of expert i
    """
    if gate is None or gate.dim() != 2:
        return torch.tensor(0.0, device=gate.device if gate is not None else 'cpu')
    B, N = gate.shape
    # 离散计数，不回传梯度 / discrete counts, no gradient
    with torch.no_grad():
        idx = gate.argmax(dim=-1)
        f = torch.zeros(N, device=gate.device, dtype=gate.dtype)
        f.scatter_add_(0, idx, torch.ones(B, device=gate.device, dtype=gate.dtype))
        f = f / B
    # 平均软权重，带梯度 / mean soft weight, with gradient
    P = gate.mean(dim=0)
    return N * (f * P).sum()


class RA_LWLM_KMoE(nn.Module):
    """5 个 ICL 专家 + selector 软混合 / five ICL experts mixed by the selector."""

    def __init__(self,
                 embed_dim: int = 256, K_max: int = 20,
                 num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.2,
                 pos_scale: float = 32.0,
                 token_dim: int | None = None, cfg_dim: int = 4,
                 selector_hidden: int = 256, selector_dropout: float = 0.2,
                 gate_temperature: float = 1.0,
                 freeze_icl: bool = False,
                 icl_models: list | None = None,
                 k_actions: list | None = None):
        super().__init__()
        self.K_max = K_max
        self.gate_temperature = gate_temperature
        self.k_actions = list(k_actions) if k_actions is not None else list(K_ACTIONS)
        self.n_actions = len(self.k_actions)
        assert all(1 <= k <= K_max for k in self.k_actions), \
            f"k_actions {self.k_actions} must be in [1, K_max={K_max}]"

        # 5 个独立的 ICL 专家（可由训练脚本传入阶段 1 预训练好的模型）
        # The five independent ICL experts (the trainer passes in the stage-1 models)
        if icl_models is not None:
            assert len(icl_models) == self.n_actions, \
                f"need {self.n_actions} icls (matches k_actions), got {len(icl_models)}"
            self.icls = nn.ModuleList(icl_models)
        else:
            self.icls = nn.ModuleList([
                RA_LWLM_ICL(embed_dim=embed_dim, K_max=K_max,
                            num_heads=num_heads, num_layers=num_layers, dropout=dropout,
                            pos_scale=pos_scale, token_dim=token_dim, cfg_dim=cfg_dim)
                for _ in range(self.n_actions)])

        # 阶段 2 默认冻结专家，只训 selector / stage 2 freezes the experts by default
        if freeze_icl:
            for icl in self.icls:
                for p in icl.parameters():
                    p.requires_grad = False
        self.freeze_icl = freeze_icl

        # 路由器。注意 q_feat / ref_feat 是编码器原始输出 [LST; patch 均值]，
        # 所以每个向量的维度是 2*embed_dim 而不是 2*token_dim。
        # The selector's q_feat / ref_feat are raw encoder outputs [LST; patch mean],
        # so the per-vector dim is 2*embed_dim, not 2*token_dim.
        self.selector = KSelector(
            q_feat_dim=embed_dim * 2, k_max=K_max, n_actions=self.n_actions,
            cfg_dim=cfg_dim, ref_proj_dim=16,
            hidden_dim=selector_hidden, dropout=selector_dropout)

        self.last_gate  = None      # 最近一次的路由权重（日志与辅助损失用）/ last routing weights
        self.last_eff_k = None      # 有效上下文长度 Σ π_i k_i / effective context length

    @property
    def POS_SCALE(self):
        return self.icls[0].POS_SCALE

    @property
    def token_dim(self):
        return self.icls[0].token_dim

    def set_freeze_icl(self, freeze: bool):
        for icl in self.icls:
            for p in icl.parameters():
                p.requires_grad = not freeze
        self.freeze_icl = freeze

    def _compute_logits(self, query_features, query_config, ref_features,
                        ref_positions, weights):
        """selector 前向；质心只用于给出参考点的几何摘要，不回传梯度。
        Selector forward; the centroid only feeds the geometric summary of the
        references and carries no gradient."""
        with torch.no_grad():
            wn = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
            centroid = torch.bmm(wn.unsqueeze(1), ref_positions).squeeze(1)
        return self.selector(
            q_feat=query_features, q_cfg=query_config, weights=weights,
            ref_features=ref_features, ref_positions=ref_positions,
            centroid=centroid, pos_scale=float(self.POS_SCALE))            # [B, N]

    def _all_ICL_forwards(self, *fwd_args):
        """跑完 N 个专家，返回 [B, N, 2] / run all N experts, returning [B, N, 2]."""
        preds = []
        for k_val, icl in zip(self.k_actions, self.icls):
            ctx = torch.no_grad() if self.freeze_icl else torch.enable_grad()
            with ctx:
                p = icl.forward_loc(*fwd_args[:5], fwd_args[5], k=k_val)
            preds.append(p)
        return torch.stack(preds, dim=1)                                   # [B, N, 2]

    def forward_loc(self, query_features, query_config,
                    ref_features, ref_positions, ref_configs, weights, k=None):
        """坐标 = Σ_i π_i · pos_i，π = softmax(selector logits / 温度)。
        Position = Σ_i π_i · pos_i with π = softmax(selector logits / temperature)."""
        logits = self._compute_logits(query_features, query_config,
                                      ref_features, ref_positions, weights)
        preds = self._all_ICL_forwards(query_features, query_config, ref_features,
                                       ref_positions, ref_configs, weights)
        gate = torch.softmax(logits / self.gate_temperature, dim=-1)        # [B, N]
        pos = (gate.unsqueeze(-1) * preds).sum(dim=1)                       # [B, 2]
        self._update_logging(gate)
        return pos

    def _update_logging(self, gate):
        self.last_gate = gate
        with torch.no_grad():
            k_t = torch.tensor(self.k_actions, dtype=gate.dtype, device=gate.device)
            self.last_eff_k = (gate.detach() * k_t.unsqueeze(0)).sum(-1).mean().item()

    def get_load_balance_loss(self) -> torch.Tensor:
        """供训练脚本调用 / called by the trainer."""
        return load_balance_loss(self.last_gate)
