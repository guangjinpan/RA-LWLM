"""
model.py
=====================================================================
无线基础模型（WFM）的网络定义 / Network definition of the wireless foundation model.

中文
----
wireless_loc_fm 是一个 Transformer 编码器：输入 (角度, 时延) 域的 CSI 幅度图
(B, 2, T, F)，切成 patch 后加上正弦位置编码，前面拼一个可学习的 LST token。
RA-LWLM 只用它的 fm_encoder()：
    enc[:, 0, :]     → LST token，作为检索用的特征（L2 距离）
    enc[:, 1:, :].mean(1) → patch 均值，和 LST 一起作为 ICL 的输入特征
DTI 预训练（train_model.py 的 task="pretrain_dti"）用 PatchMLPDecoder 重建
时延-角度域，不需要任何位置标签。

EN
--
wireless_loc_fm is a Transformer encoder: it takes the (angle, delay)-domain CSI
magnitude map (B, 2, T, F), splits it into patches, adds a sinusoidal positional
encoding and prepends a learnable LST token. RA-LWLM only calls fm_encoder():
    enc[:, 0, :]          → the LST token, used as the retrieval feature (L2)
    enc[:, 1:, :].mean(1) → the patch mean; together with the LST token it forms
                            the CSI feature fed to the ICL
DTI pretraining (task="pretrain_dti" in train_model.py) reconstructs the
delay-angle domain through PatchMLPDecoder and needs no position labels.

编码器是冻结的：RA-LWLM 训练时不更新它的任何参数。
The encoder is frozen: RA-LWLM never updates a single weight in it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from random import randrange
from torchvision.models import resnet34

def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)

def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def cosine_similarity_loss(H1, H2, eps = 1e-1 ):
    """DTI 重建损失中的余弦相似度项 / cosine-similarity term of the DTI reconstruction loss."""


    H1r = H1[:,0,:,:]
    H1i = H1[:,1,:,:]
    H2r = H2[:,0,:,:]
    H2i = H2[:,1,:,:]
    B, M, N = H1r.shape
    # 1) reshape to [B, M*N]
    xi = H1r.reshape(B,-1)  # x real
    xj = H1i.reshape(B,-1)  # x imag
    yi = H2r.reshape(B,-1)  # y real
    yj = H2i.reshape(B,-1)  # y imag

    # 2) calculate dot product
    #    xy_dot_r = Σ(xr * yr + xi * yi)
    #    xy_dot_i = Σ(xr * yi - xi * yr)
    xy_dot_r = torch.sum(xi * yi + xj * yj, dim=1)
    xy_dot_i = torch.sum(xi * yj - xj * yi, dim=1)

    # 3) calculate ||x|| and ||y||
    norm_x = torch.sqrt(torch.sum(xi**2, dim=1) + torch.sum(xj**2, dim=1))
    norm_y = torch.sqrt(torch.sum(yi**2, dim=1) + torch.sum(yj**2, dim=1))

    # 5) calculate sqrt( real^2 + imag^2 ) / (||x|| * ||y||)
    dot_mod = torch.sqrt(xy_dot_r**2 + xy_dot_i**2)
    cos_loss = 1- dot_mod / (norm_x * norm_y + eps)

    return torch.mean(cos_loss)



class CNNnet(nn.Module):
    """ResNet-34 主干的直接回归分支（RA-LWLM 不使用，保留是因为 Wrapper 里按 task 引用）。
    A plain ResNet-34 regression branch, referenced by Wrapper for other tasks and
    unused by RA-LWLM.
    input : [B, M, 128, 32]   output: [B, 2]
    """
    def __init__(self, input_feature_dim):
        super().__init__()

        self.net  = resnet34()    # torchvision>=0.13

        self.net.conv1 = nn.Conv2d(input_feature_dim, 64,
                                   kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(self.net.conv1.weight, mode="fan_out", nonlinearity="relu")


        # MLP layer
        self.net.fc = nn.Linear(self.net.fc.in_features, 2)

    def forward(self, x):
        return self.net(x)





class AttentionPooling(nn.Module):
    """注意力池化：把 N 个 token 加权成一个向量 / attention pooling of N tokens into one vector."""
    def __init__(self, input_dim, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):  # x: [B, N, D]
        attn_scores = self.attn(x).squeeze(-1)  # [B, N]
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, N]
        fused = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # [B, D]
        return fused, attn_weights

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
         (B, in_chans, H, W) -> (B, num_patches, embed_dim)

    img_size is used only to pre-compute num_patches for logging; the actual
    Conv2d projection works for any (H, W) that is divisible by patch_size.
    """
    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=512):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """把 CSI 图切成 patch 并线性投影成 token；H/W 只要能被 patch 整除即可，
        因此同一套权重可以吃 8/16/32 天线的输入。
        Split the CSI map into patches and project them to tokens. H and W only need
        to be divisible by the patch size, which is how one set of weights serves
        8/16/32-antenna inputs.

        x: (B, in_chans, H, W)  — H and W can differ from img_size at runtime,
            provided H % patch_size[0] == 0 and W % patch_size[1] == 0.
        output: (B, num_patches_actual, embed_dim)
        """
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def get_sinusoid_encoding(n_position, d_hid, device=None):
    """正弦位置编码 (1, n_position, d_hid)，直接在目标设备上用 torch 生成。
    token 数随天线数变化，所以按需计算并在 fm_encoder 里按 (n, device) 缓存。
    （早先的实现用 Python 循环在 CPU 上构表、每次 fm_encoder 都拷到 GPU，
    那是推理延迟的主要来源 —— 无论 batch 多大都要 ~65 ms。）

    Sinusoidal positional encoding (1, n_position, d_hid), built with torch directly
    on `device`. The token count depends on the antenna count, so it is computed on
    demand and cached per (n, device) inside fm_encoder. (The earlier version built
    the table with Python loops on the CPU and copied it to the GPU on every
    fm_encoder() call, which dominated inference latency at ~65 ms per forward.)"""
    pos = torch.arange(n_position, dtype=torch.float64, device=device).unsqueeze(1)       # (N,1)
    hid = torch.arange(d_hid, dtype=torch.float64, device=device)                         # (D,)
    angle = pos / torch.pow(torch.tensor(10000.0, dtype=torch.float64, device=device), 2 * (hid // 2) / d_hid)
    table = torch.empty_like(angle)
    table[:, 0::2] = torch.sin(angle[:, 0::2])
    table[:, 1::2] = torch.cos(angle[:, 1::2])
    return table.to(torch.float32).unsqueeze(0)


# ====================== DTI Decoder: Patch-wise MLP (方案B) ====================== #
class PatchMLPDecoder(nn.Module):
    """Patch-wise MLP decoder for DTI pretraining.
    Each patch token is independently mapped to the reconstruction target,
    with no cross-patch attention — forces the encoder to encode
    delay-angle structure into each individual patch token.
    """

    def __init__(self, latent_dim, hidden_dim, final_out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, final_out_dim),
        )

    def forward(self, x):
        """
        x: (B, N+1, latent_dim)  — encoder output (including cls token)
        return: (B, N+1, final_out_dim)
        """
        return self.mlp(x)


class wireless_loc_fm(nn.Module):

    def __init__(
        self,
        label_dim=567,
        # patch
        fshape=4, tshape=4, fstride=4, tstride=4,
        input_fdim=64, input_tdim=32, input_fmap=2,

        # Transformer(Encoder)
        embed_dim=512,
        depth=6,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        contrast_out_dim = 32,
        BSconf_dim = 3,

        device="cpu",

        EnvPara = None,
    ):
        super().__init__()
        self.device = device
        self.EnvPara = EnvPara

        # ----------------------------------------------------
        # 1) Patch Embedding (Encoder input)
        # ----------------------------------------------------
        self.fshape, self.tshape = fshape, tshape
        self.fstride, self.tstride = fstride, tstride
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        self.input_fmap = input_fmap

        patch_size = (fshape, tshape)
        in_chans = input_fmap

        # ----------------------------------------------------
        # Encoder(1): CNN layer for patch embedding
        # ----------------------------------------------------
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=(input_fdim, input_tdim),
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        self.num_patches = self.patch_embed.num_patches  # default-config value for logging

        # cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_num = 1

        # Positional embedding is computed on-the-fly in fm_encoder so the
        # model handles any (H, W) that is divisible by (fshape, tshape).
        # We keep a small cache to avoid re-generating for the same length.

        # ----------------------------------------------------
        # Encoder(2): TransformerEncoder + LayerNorm
        # ----------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        print(f"Encoder parameters: {(count_parameters(self.patch_embed)+count_parameters(self.transformer_encoder)+count_parameters(self.norm)) / 1e6:.2f}M")
        print(self.EnvPara["task"])

        if (self.EnvPara["task"] == "SingleBSLoc") | (self.EnvPara["task"] == "inference_SingleBSLoc"):
            # ----------------------------------------------------
            # Decoder(0): Single-BS-Loc Decoder
            # ----------------------------------------------------
            self.proj_BSconfig2 = nn.Sequential(
                nn.Linear(BSconf_dim, embed_dim),
                nn.ReLU(),
            )
            self.SB_positioninglayer = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
            )
            print(f"Single-Loc parameters: {(count_parameters(self.proj_BSconfig2)+count_parameters(self.SB_positioninglayer)) / 1e6:.2f}M")
        elif (self.EnvPara["task"] == "aoa") | (self.EnvPara["task"] == "inference_aoa"):
            # ----------------------------------------------------
            # Decoder: AOA Decoder
            # ----------------------------------------------------
            self.proj_BSconfig2 = nn.Sequential(
                nn.Linear(BSconf_dim, embed_dim),
                nn.ReLU(),
            )
            self.aoa_layer = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            )
            print(f"aoa parameters: {(count_parameters(self.proj_BSconfig2)+count_parameters(self.aoa_layer)) / 1e6:.2f}M")

        elif (self.EnvPara["task"] == "toa")| (self.EnvPara["task"] == "inference_toa"):
            # ----------------------------------------------------
            # Decoder: TOA Decoder
            # ----------------------------------------------------
            self.proj_BSconfig2 = nn.Sequential(
                nn.Linear(BSconf_dim, embed_dim),
                nn.ReLU(),
            )
            self.toa_layer = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            )
            print(f"toa parameters: {(count_parameters(self.proj_BSconfig2)+count_parameters(self.toa_layer)) / 1e6:.2f}M")

        elif (self.EnvPara["task"] == "MultiBSLoc") | (self.EnvPara["task"] == "inference_multiBSLoc"):
            # ----------------------------------------------------
            # Decoder: Multi-BS-Loc Decoder
            # ----------------------------------------------------
            print()
            self.proj_BSconfig2 = nn.Sequential(
                nn.Linear(BSconf_dim, embed_dim),
                nn.ReLU(),
            )
            self.MB_positioninglayer0 = nn.ModuleList()
            for i in range(5):
                SB_layer = nn.Sequential(
                    nn.Linear(512, 128),
                    nn.ReLU(),
                )
                self.MB_positioninglayer0.append(SB_layer)
            self.MB_positioninglayer1 = nn.ModuleList()
            for i in range(5):
                SB_layer = nn.Sequential(
                    nn.Linear(128, 2),
                )
                self.MB_positioninglayer1.append(SB_layer)
            self.attn_pool2 = AttentionPooling(input_dim=128, hidden_dim=32)
            print(f"Multi-Loc parameters: {(count_parameters(self.proj_BSconfig2)+count_parameters(self.MB_positioninglayer0)) / 1e6:.2f}M")


        # ----------------------------------------------------
        # DTI Decoder: Patch-wise MLP (方案B)
        # Replace TransformerDecoder with per-patch MLP to force
        # the encoder to learn robust per-token delay-angle features.
        # ----------------------------------------------------
        # patch_dim is fixed by patch size (not by input size),
        # so the decoder MLP weight shape is the same for all input configs.
        patch_dim = fshape * tshape * input_fmap
        self.decoder_DTI = PatchMLPDecoder(
            latent_dim=embed_dim,
            hidden_dim=embed_dim * 2,
            final_out_dim=patch_dim,
        )
        # nn.Fold is created on-the-fly in dti_pretraining so the model handles
        # any (H, W) input — no fixed self.fold stored here.
        print(f"DTI Decoder (PatchMLP) parameters: {count_parameters(self.decoder_DTI) / 1e6:.2f}M")

        self.L1loss = nn.L1Loss()
        if ((self.EnvPara["task"] == "CNN") | (self.EnvPara["task"] == "inference_cnn")):
            self.cnnnet1=CNNnet(EnvPara["input_feature_dim"])
            print(f"CNN parameters: {(count_parameters(self.cnnnet1)) / 1e6:.2f}M")


    def fm_encoder(self, input):
        """RA-LWLM 使用的唯一入口：CSI 图 → [LST token; patch tokens]。
        The only entry point RA-LWLM uses: a CSI map → [LST token; patch tokens].

        input: (B, in_chans, H, W)
        H and W can be any multiple of fshape / tshape respectively.
        Positional encoding is computed on-the-fly from the actual token count.
        """
        B = input.shape[0]

        x = self.patch_embed(input)          # (B, N_patches, embed_dim) — N varies
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N_patches+1, embed_dim)

        # Dynamic sinusoidal positional encoding — no parameters; built on-device and cached
        # per (token count, device) so repeated forwards pay nothing.
        cache = getattr(self, "_pe_cache", None)
        if cache is None:
            cache = self._pe_cache = {}
        key = (x.shape[1], x.device)
        pos_embed = cache.get(key)
        if pos_embed is None:
            pos_embed = cache[key] = get_sinusoid_encoding(x.shape[1], self.embed_dim, device=x.device)
        x = x + pos_embed
        x = self.transformer_encoder(x)
        x = self.norm(x)
        return x


    # ================== Pretraining: DTI only ================== #
    def dti_pretraining(self, channel_Antenna_subcarrier_aa, channel_delay_angle_ri):
        """DTI pretraining: encode spatial-frequency domain, decode to delay-angle domain.
        Uses Patch-wise MLP decoder (no cross-patch attention).
        Works for any (H, W) divisible by (fshape, tshape).

        Input tensors arrive as (B, 1, 2, F, T) — BS-dim=1, 2 channels (real/imag).
        """
        # Drop the BS dimension first → (B, 2, F, T)
        x_input    = channel_Antenna_subcarrier_aa[:, 0, :]  # (B, 2, F, T)
        x_input_ri = channel_delay_angle_ri[:, 0, :]         # (B, 2, F, T)
        B, _, H, W = x_input.shape                           # H=F (subcarriers), W=T (antennas)

        assert H % self.fshape == 0, \
            f"input H={H} must be divisible by fshape={self.fshape}"
        assert W % self.tshape == 0, \
            f"input W={W} must be divisible by tshape={self.tshape}"

        # Encoder — handles variable (H, W) via dynamic pos encoding
        enc_out = self.fm_encoder(x_input)  # (B, N_patches+1, embed_dim)

        # Patch-wise MLP decode
        rec = self.decoder_DTI(enc_out)           # (B, N_patches+1, patch_dim)
        rec_wo_cls = rec[:, self.cls_token_num:, :]  # (B, N_patches, patch_dim)

        # Fold back to (B, in_chans, H, W) — use actual dims, not fixed defaults
        rec_wo_cls = rec_wo_cls.transpose(1, 2)   # (B, patch_dim, N_patches)
        fold = nn.Fold(
            output_size=(H, W),
            kernel_size=(self.fshape, self.tshape),
            stride=(self.fstride, self.tstride),
        )
        x_recon = fold(rec_wo_cls)                # (B, in_chans, H, W)

        # Cosine similarity loss
        DTI_loss = cosine_similarity_loss(x_input_ri, x_recon)
        return DTI_loss


    def _freeze_encoder(self):
        modules_to_freeze = [
            self.patch_embed,
            self.transformer_encoder,
            self.norm
        ]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
        # pos_embed is now computed on-the-fly (no parameter), so no freeze needed
        self.cls_token.requires_grad = True


    def toa_est(self, channel_data, distance, BSconf_all):
        input = channel_data[:,0,:]
        BSconf = BSconf_all[:,0,:]
        B = input.shape[0]
        distance = distance[:,0].reshape((B,1))
        N_a = input.shape[3]
        N_s = input.shape[2]

        # pilot mask: 1 on every (pilot_subcarrier_interval, pilot_antenna_interval)-th (subcarrier, antenna).
        # Vectorised (identical result to the former per-element Python loop, which issued
        # N_a*N_s tiny GPU kernels per forward and dominated inference latency).
        mask_dense_ant_subc = torch.zeros_like(input)
        mask_dense_ant_subc[:, :, ::self.EnvPara["pilot_subcarrier_interval"], ::self.EnvPara["pilot_antenna_interval"]] = 1
        input = input * mask_dense_ant_subc

        if self.EnvPara["is_frozen"]==1:
            self._freeze_encoder()
            x = self.fm_encoder(input)
        else:
            x = self.fm_encoder(input)

        x_mean = x[:,0,:].view((B,-1))
        z_x_c = self.proj_BSconfig2(BSconf)
        z_x_list=[]
        z_x_list.append(x_mean)
        z_x_list.append(z_x_c)
        z_x_concat = torch.cat(z_x_list, dim=1)
        pred = self.toa_layer(z_x_concat)
        loss = self.L1loss(pred, distance)
        return pred, loss

    def aoa_est(self, channel_data, angle, BSconf_all):
        input = channel_data[:,0,:]

        BSconf = BSconf_all[:,0,:]
        B = input.shape[0]
        angle = angle[:,0].reshape((B,1))
        N_a = input.shape[3]
        N_s = input.shape[2]

        # pilot mask: 1 on every (pilot_subcarrier_interval, pilot_antenna_interval)-th (subcarrier, antenna).
        # Vectorised (identical result to the former per-element Python loop, which issued
        # N_a*N_s tiny GPU kernels per forward and dominated inference latency).
        mask_dense_ant_subc = torch.zeros_like(input)
        mask_dense_ant_subc[:, :, ::self.EnvPara["pilot_subcarrier_interval"], ::self.EnvPara["pilot_antenna_interval"]] = 1
        input = input * mask_dense_ant_subc

        # --- A. Encoder for x ---
        if self.EnvPara["is_frozen"]==1:
            self._freeze_encoder()
            x = self.fm_encoder(input)
        else:
            x = self.fm_encoder(input)
        x_mean = x[:,0,:].view((B,-1))
        z_x_c = self.proj_BSconfig2(BSconf)
        z_x_list=[]
        z_x_list.append(x_mean)
        z_x_list.append(z_x_c)
        z_x_concat = torch.cat(z_x_list, dim=1)
        pred = self.aoa_layer(z_x_concat)
        loss = self.L1loss(pred, angle)
        return pred, loss

    def cnn_est(self, channel_data, UElocation_all, BSconf_all):
        input = channel_data[:,0,:]
        y_position = UElocation_all[:,0,:]
        B = input.shape[0]
        N_a = input.shape[3]
        N_s = input.shape[2]

        # pilot mask: 1 on every (pilot_subcarrier_interval, pilot_antenna_interval)-th (subcarrier, antenna).
        # Vectorised (identical result to the former per-element Python loop, which issued
        # N_a*N_s tiny GPU kernels per forward and dominated inference latency).
        mask_dense_ant_subc = torch.zeros_like(input)
        mask_dense_ant_subc[:, :, ::self.EnvPara["pilot_subcarrier_interval"], ::self.EnvPara["pilot_antenna_interval"]] = 1
        input = input * mask_dense_ant_subc

        pred = self.cnnnet1(input)
        dis = torch.sum((pred - y_position) ** 2, 1)
        mse = torch.mean(torch.sqrt(dis))

        return pred, mse


    def singleBSLoc_with_pilot(self, channel_data, UElocation_all, BSconf_all):
        input = channel_data[:,0,:]
        y_position = UElocation_all[:,0,:]
        BSconf = BSconf_all[:,0,:]
        B = input.shape[0]
        N_a = input.shape[3]
        N_s = input.shape[2]

        # pilot mask: 1 on every (pilot_subcarrier_interval, pilot_antenna_interval)-th (subcarrier, antenna).
        # Vectorised (identical result to the former per-element Python loop, which issued
        # N_a*N_s tiny GPU kernels per forward and dominated inference latency).
        mask_dense_ant_subc = torch.zeros_like(input)
        mask_dense_ant_subc[:, :, ::self.EnvPara["pilot_subcarrier_interval"], ::self.EnvPara["pilot_antenna_interval"]] = 1
        input = input * mask_dense_ant_subc

        # --- A. Encoder for x ---
        if self.EnvPara["is_frozen"] == 1:
            if self.EnvPara.get("current_epoch", 0) <= 500:
                modules_to_freeze = [
                    self.patch_embed,
                    self.transformer_encoder,
                    self.norm
                ]
                for module in modules_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False
                # pos_embed is computed on-the-fly; nothing to freeze here
                self.cls_token.requires_grad = True
            else:
                # Ensure they are trainable if epoch < 20
                modules_to_unfreeze = [
                    self.patch_embed,
                    self.transformer_encoder,
                    self.norm
                ]
                for module in modules_to_unfreeze:
                    for param in module.parameters():
                        param.requires_grad = True
                # pos_embed is computed on-the-fly; nothing to freeze here
                self.cls_token.requires_grad = True

        x = self.fm_encoder(input)
        x_mean = x[:,0,:].view((B,-1))
        z_x_c = self.proj_BSconfig2(BSconf)
        z_x_list=[]
        z_x_list.append(x_mean)
        z_x_list.append(z_x_c)
        z_x_concat = torch.cat(z_x_list, dim=1)
        pred = self.SB_positioninglayer(z_x_concat)
        dis = torch.sum((pred - y_position) ** 2, 1)
        mse = torch.mean(torch.sqrt(dis))

        return pred, mse




    def multiBSLoc_att(self, channel_data, UElocation_all, BSconf_all, mode = "train"):
        y_position = UElocation_all[:, 0, :]
        B = channel_data.shape[0]
        if mode == "train":
            T = random.randint(2, 5)
        else:
            T = self.EnvPara["BS_Num"]

        emb_list1 = []

        for i in range(T):
            input = channel_data[:, i, :]
            BSconf = BSconf_all[:, i, :]

            if self.EnvPara["is_frozen"]==1:
                self._freeze_encoder()
                x = self.fm_encoder(input)
            else:
                x = self.fm_encoder(input)
            x_mean = x[:,0,:].view((B,-1))
            z_x_c = self.proj_BSconfig2(BSconf)
            z_x_list = []
            z_x_list.append(x_mean)
            z_x_list.append(z_x_c)
            z_x_concat = torch.cat(z_x_list, dim=1)
            pred = self.MB_positioninglayer0[i](z_x_concat)
            emb_list1.append(pred)

        bs_embeddings0 = torch.stack(emb_list1, dim=1)
        att_embed, attn_weights = self.attn_pool2(bs_embeddings0)
        emb_list2  = []
        for i in range(T):
            bs_embeddings_i = bs_embeddings0[:, i, :]
            pred = self.MB_positioninglayer1[i](bs_embeddings_i)
            emb_list2.append(pred)

        bs_embeddings1 = torch.stack(emb_list2, dim=1)
        pred_res = torch.bmm(attn_weights.unsqueeze(1), bs_embeddings1).squeeze(1)  # [B, D]
        dis = torch.sum((pred_res - y_position) ** 2, 1)
        mse = torch.mean(torch.sqrt(dis))*1

        if mode == "train":
            for i in range(T):
                dis1 = torch.sum((bs_embeddings1[:,i,:] - y_position) ** 2, 1)
                mse = mse + torch.mean(torch.sqrt(dis1))*0.1
        return pred_res, mse





    def forward(self, channel_Antenna_subcarrier_aa, channel_Antenna_subcarrier_ri, channel_delay_angle_ri, UElocation_all, BSconf_all, task = "pretrain_dti",
                cluster=True, mask_antenna_number=8, mask_subcarrier_number = 32):
        """
        x: (B, input_fmap, input_fdim, input_tdim)
        task:
            - "SingleBSLoc", "inference_SingleBSLoc"
            - "MultiBSLoc", "inference_multiBSLoc"
            - "toa", "inference_toa"
            - "aoa", "inference_aoa"
            - "pretrain_dti"   (DTI-only pretraining with Patch-wise MLP decoder)
        """

        channel_Antenna_subcarrier_aa = channel_Antenna_subcarrier_aa.transpose(3, 4)
        channel_Antenna_subcarrier_ri = channel_Antenna_subcarrier_ri.transpose(3, 4)
        channel_delay_angle_ri = channel_delay_angle_ri.transpose(3, 4)
        distance = UElocation_all[:,:,2]
        angle = UElocation_all[:,:,3]
        UElocation_all = UElocation_all[:,:,:2]


        if task == "SingleBSLoc":
            pred, mse = self.singleBSLoc_with_pilot(channel_Antenna_subcarrier_aa, UElocation_all, BSconf_all)
            return mse
        elif task == "inference_SingleBSLoc":
            return self.singleBSLoc_with_pilot(channel_Antenna_subcarrier_aa, UElocation_all, BSconf_all)
        elif task == "MultiBSLoc":
            pred, loss =  self.multiBSLoc_att(channel_Antenna_subcarrier_aa, UElocation_all, BSconf_all, mode ="train")
            return loss
        elif task == "inference_multiBSLoc":
            return self.multiBSLoc_att(channel_Antenna_subcarrier_aa, UElocation_all, BSconf_all, mode = "test")
        elif task == "toa":
            pred, loss = self.toa_est(channel_Antenna_subcarrier_aa, distance, BSconf_all)
            return loss
        elif task == "inference_toa":
            return self.toa_est(channel_Antenna_subcarrier_aa, distance, BSconf_all)
        elif task == "aoa":
            pred, loss = self.aoa_est(channel_Antenna_subcarrier_aa, angle, BSconf_all)
            return loss
        elif task == "inference_aoa":
            return self.aoa_est(channel_Antenna_subcarrier_aa, angle, BSconf_all)
        elif task == "CNN":
            pred, loss = self.cnn_est(channel_Antenna_subcarrier_aa, UElocation_all, BSconf_all)
            return loss
        elif task == "inference_cnn":
            return self.cnn_est(channel_Antenna_subcarrier_aa, UElocation_all, BSconf_all)


        # ============== pretrain task: DTI only ==============
        elif task == "pretrain_dti":
            return self.dti_pretraining(channel_Antenna_subcarrier_aa, channel_delay_angle_ri)
        else:
            raise ValueError(f"Unsupported task: {task}")
