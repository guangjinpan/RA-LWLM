"""
selector.py
=====================================================================
K-MoE 路由器（selector）/ K-MoE router (selector).

中文：给定一个 query 及其 top-K 检索结果，selector 输出 N 个专家（即 N 个不同
      的上下文长度 k）的 logits，K-MoE 用它的 softmax 作为专家的混合权重。
EN:   Given a query and its top-K retrieved references, the selector produces
      logits over the N experts (N different in-context lengths k); the K-MoE uses
      their softmax as the mixture weights over the experts.

被 ra_lwlm_kmoe.py 使用 / used by ra_lwlm_kmoe.py.
"""
import torch
import torch.nn as nn



class KSelector(nn.Module):
    """路由 MLP：根据 query 与检索结果的丰富上下文，输出 N 个上下文长度的 logits。
    它看到的不只是 query 特征，还包括检索权重、参考点的几何分布，以及 query 与
    参考点之间的特征差 —— 这些正是"该看几个参考点"的判据。

    Router MLP producing logits over the N context lengths. It sees more than the
    query feature: the retrieval weights, the geometry of the retrieved references
    and the query-to-reference feature gaps, which are exactly the cues that tell
    how many references should be attended to.

    Input (concatenated) / 输入（拼接）：
      • q_feat              : query CSI embedding [LST; mean_patch]   (q_feat_dim)
      • q_cfg               : scene/BS config                         (cfg_dim)
      • log_w               : log retrieval weights for K_max refs    (k_max)
      • ref_pos_summary     : geometric summary of refs vs centroid   (6)
                                (mean_xy, std_xy, max_dist, mean_dist normalized by pos_scale)
      • first_ref_diff      : q_feat - z_{r1}  (KNN-1 feature gap)    (q_feat_dim)
      • mean_ref_diff       : q_feat - mean(z_{r_i})  (avg gap)        (q_feat_dim)
      • ref_z_proj          : compressed per-ref feature summary       (k_max * proj_dim)
                                (each z_i projected to proj_dim, then flattened)
    """
    POS_SUMMARY_DIM = 6   # mean_x, mean_y, std_x, std_y, max_dist, mean_dist

    def __init__(self, q_feat_dim: int, k_max: int, n_actions: int,
                 cfg_dim: int = 4, ref_proj_dim: int = 16,
                 hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        # Per-ref feature compressor (z_i ∈ R^{q_feat_dim} → R^{ref_proj_dim})
        self.ref_proj = nn.Linear(q_feat_dim, ref_proj_dim)

        in_dim = (
            q_feat_dim                      # q_feat
            + cfg_dim                       # q_cfg
            + k_max                         # log_w
            + self.POS_SUMMARY_DIM          # ref_pos_summary
            + q_feat_dim                    # first_ref_diff
            + q_feat_dim                    # mean_ref_diff
            + k_max * ref_proj_dim          # ref_z_proj (flattened)
        )
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, q_feat, q_cfg, weights,
                ref_features, ref_positions, centroid, pos_scale):
        """All shapes:
           q_feat   [B, q_feat_dim]
           q_cfg    [B, cfg_dim]
           weights  [B, K_max]
           ref_features  [B, K_max, q_feat_dim]
           ref_positions [B, K_max, 2]
           centroid [B, 2]
           pos_scale: float
        """
        log_w = torch.log(weights + 1e-8)                                # [B, K_max]

        # Geometric summary of ref positions (relative to centroid, normalized)
        rel = (ref_positions - centroid.unsqueeze(1)) / pos_scale         # [B, K_max, 2]
        pos_mean = rel.mean(dim=1)                                        # [B, 2]
        pos_std  = rel.std(dim=1, unbiased=False)                         # [B, 2]
        pos_dist = rel.norm(dim=-1)                                       # [B, K_max]
        pos_max  = pos_dist.max(dim=-1, keepdim=True).values              # [B, 1]
        pos_avg  = pos_dist.mean(dim=-1, keepdim=True)                    # [B, 1]
        pos_summary = torch.cat([pos_mean, pos_std, pos_max, pos_avg], dim=-1)  # [B, 6]

        # Query-vs-ref feature differences
        first_ref_diff = q_feat - ref_features[:, 0, :]                   # [B, q_feat_dim]
        mean_ref       = ref_features.mean(dim=1)                         # [B, q_feat_dim]
        mean_ref_diff  = q_feat - mean_ref                                # [B, q_feat_dim]

        # Per-ref compressed features (preserves "what each ref looks like")
        ref_z_proj = self.ref_proj(ref_features)                          # [B, K_max, ref_proj_dim]
        ref_z_flat = ref_z_proj.flatten(start_dim=1)                      # [B, K_max * ref_proj_dim]

        x = torch.cat([
            q_feat, q_cfg, log_w, pos_summary,
            first_ref_diff, mean_ref_diff, ref_z_flat
        ], dim=-1)
        return self.net(x)                                                # [B, N] logits
