"""
train_model.py
=====================================================================
Lightning 包装器 / Lightning wrapper around wireless_loc_fm.

中文：Wrapper 负责按 EnvPara["task"] 切换任务（DTI 自监督预训练 / 单基站定位等），
      并给出训练、验证步骤。RA-LWLM 只用它来加载预训练权重然后取 fm_encoder：
          Wrapper.load_from_checkpoint(ckpt, EnvPara=..., strict=False)
          wrapper.channel_fdmdl.fm_encoder(x)
EN:   Wrapper switches task by EnvPara["task"] (DTI self-supervised pretraining,
      single-BS localisation, ...) and defines the train/val steps. RA-LWLM only
      uses it to load the pretrained weights and reach fm_encoder:
          Wrapper.load_from_checkpoint(ckpt, EnvPara=..., strict=False)
          wrapper.channel_fdmdl.fm_encoder(x)

注意 / note: EnvPara 的字段必须与预训练时完全一致（patch 大小、embed_dim、depth、
num_heads、input_tdim=天线数），否则权重对不上。
The EnvPara fields must match pretraining exactly (patch size, embed_dim, depth,
num_heads, input_tdim = antenna count), otherwise the weights will not line up.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

import argparse
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, ConcatDataset, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
import math
import torch
import math
import os
import numpy as np
from scipy.spatial.distance import cdist
import random
from model import *
from torch.backends.cuda import sdp_kernel, SDPBackend


class Wrapper(pl.LightningModule):
    """把 wireless_loc_fm 包成 LightningModule / wraps wireless_loc_fm as a LightningModule."""
    def __init__(self, EnvPara):
        super().__init__()

        self.channel_fdmdl =  wireless_loc_fm(
                fshape = EnvPara["fshape"], tshape = EnvPara["tshape"], fstride = EnvPara["fstride"], tstride = EnvPara["tstride"],
                input_fdim = EnvPara["input_fdim"], input_tdim = EnvPara["input_tdim"], input_fmap = EnvPara["input_fmap"], embed_dim=EnvPara["embed_dim"], depth = EnvPara["depth"],
                num_heads = EnvPara["num_heads"], device = EnvPara["device"],
                BSconf_dim = EnvPara.get("BSconf_dim", 3),
                EnvPara = EnvPara)
        self.task = EnvPara["task"]
        self.train_epoch_loss = [] 
        self.valepoch_loss = []
        self.EnvPara = EnvPara
        torch.autograd.set_detect_anomaly(True)
        sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False)


    def forward(self, channel_Antenna_subcarrier_aa, channel_Antenna_subcarrier_ri, channel_delay_angle_ri, UElocation_all, BSconf_all):
             
        if (self.task == "pretrain_dti"):
            loss = self.channel_fdmdl(channel_Antenna_subcarrier_aa = channel_Antenna_subcarrier_aa, channel_Antenna_subcarrier_ri = channel_Antenna_subcarrier_ri, channel_delay_angle_ri = channel_delay_angle_ri, UElocation_all = UElocation_all, BSconf_all = BSconf_all, task = self.task)
        elif self.task == "SingleBSLoc":
            loss = self.channel_fdmdl(channel_Antenna_subcarrier_aa = channel_Antenna_subcarrier_aa, channel_Antenna_subcarrier_ri = channel_Antenna_subcarrier_ri, channel_delay_angle_ri = channel_delay_angle_ri, UElocation_all = UElocation_all, BSconf_all = BSconf_all, task = self.task)
        elif self.task == "MultiBSLoc":
            loss = self.channel_fdmdl(channel_Antenna_subcarrier_aa = channel_Antenna_subcarrier_aa, channel_Antenna_subcarrier_ri = channel_Antenna_subcarrier_ri, channel_delay_angle_ri = channel_delay_angle_ri, UElocation_all = UElocation_all, BSconf_all = BSconf_all, task = self.task)
        elif self.task == "toa":
            loss = self.channel_fdmdl(channel_Antenna_subcarrier_aa = channel_Antenna_subcarrier_aa, channel_Antenna_subcarrier_ri = channel_Antenna_subcarrier_ri, channel_delay_angle_ri = channel_delay_angle_ri, UElocation_all = UElocation_all, BSconf_all = BSconf_all, task = self.task)
        elif self.task == "aoa":
            loss = self.channel_fdmdl(channel_Antenna_subcarrier_aa = channel_Antenna_subcarrier_aa, channel_Antenna_subcarrier_ri = channel_Antenna_subcarrier_ri, channel_delay_angle_ri = channel_delay_angle_ri, UElocation_all = UElocation_all, BSconf_all = BSconf_all, task = self.task)

    
        return loss
    

    def training_step(self, batch, batch_idx):
        channel_Antenna_subcarrier_aa, channel_Antenna_subcarrier_ri, channel_delay_angle_ri, UElocation_all, BSconf_all = batch
        channel_Antenna_subcarrier_aa = channel_Antenna_subcarrier_aa.float()
        channel_Antenna_subcarrier_ri = channel_Antenna_subcarrier_ri.float()
        channel_delay_angle_ri = channel_delay_angle_ri.float()
        UElocation_all = UElocation_all.float()
        BSconf_all = BSconf_all.float()

        loss = self.forward(channel_Antenna_subcarrier_aa = channel_Antenna_subcarrier_aa, channel_Antenna_subcarrier_ri = channel_Antenna_subcarrier_ri, channel_delay_angle_ri = channel_delay_angle_ri, UElocation_all = UElocation_all, BSconf_all = BSconf_all)

                    
        self.train_epoch_loss.append(loss.detach())
        self.log('train/loss', loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        return {'loss': loss}

    def on_train_epoch_end(self):
        # calculate average loss
        avg_loss = torch.stack(self.train_epoch_loss).mean()
        print(f"Epoch {self.current_epoch} - Average Training Loss: {avg_loss.item()}", len(self.train_epoch_loss))
        self.log('train/ave_loss', avg_loss.item(), prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        # Clear the loss list for the next epoch of training
        self.train_epoch_loss.clear()
        
        # Update current_epoch in EnvPara for conditional freezing
        self.EnvPara["current_epoch"] = self.current_epoch + 1
    
    
    def validation_step(self, batch, batch_idx):
        channel_Antenna_subcarrier_aa, channel_Antenna_subcarrier_ri, channel_delay_angle_ri, UElocation_all, BSconf_all = batch
        channel_Antenna_subcarrier_aa = channel_Antenna_subcarrier_aa.float()
        channel_Antenna_subcarrier_ri = channel_Antenna_subcarrier_ri.float()
        channel_delay_angle_ri = channel_delay_angle_ri.float()
        UElocation_all = UElocation_all.float()
        BSconf_all = BSconf_all.float()

        loss = self.forward(channel_Antenna_subcarrier_aa = channel_Antenna_subcarrier_aa, channel_Antenna_subcarrier_ri = channel_Antenna_subcarrier_ri, channel_delay_angle_ri = channel_delay_angle_ri, UElocation_all = UElocation_all, BSconf_all = BSconf_all)

        self.valepoch_loss.append(loss.detach())
        self.log('val/loss', loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)


    def on_validation_epoch_end(self):
        # calculate average loss
        avg_loss = torch.stack(self.valepoch_loss).mean()
        print(f"Epoch {self.current_epoch} - Average Validation Loss: {avg_loss.item()}", len(self.valepoch_loss))
        self.log('val/ave_loss', avg_loss.item(), prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)

        # Clear the loss list for the next epoch of training
        self.valepoch_loss.clear()

    def configure_optimizers(self):
        self.optim = torch.optim.Adam(self.parameters(), lr=self.EnvPara["lr"], weight_decay=1e-4)#, eps=1e-6)
        # self.schedule = torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=1e-3, total_steps=15625000, pct_start=0.1, final_div_factor=1e2)

        self.schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optim,
            T_0=50000,   # 第一个周期长度（单位：step）
            T_mult=1,    # 每次重启周期是否翻倍（1=固定长度，2=每次周期翻倍）
            eta_min=1e-5 # 最小学习率
        )

        return {
            'optimizer': self.optim, 
            'lr_scheduler': {'scheduler': self.schedule, 'interval': 'step'}
        }