import torch
import math
from torch import nn
from einops import rearrange
import logging
from .build import MODEL_REGISTRY
from timm import create_model
from timm.layers.helpers import to_2tuple
from timm.layers import trunc_normal_
from timm.layers.pos_embed import resample_abs_pos_embed
from timm.models.vision_transformer import Block
from .bot import BNNeck
from copy import deepcopy
from functools import partial
from .transreid import PatchEmbed_overlap
from mamba_ssm.ops.triton.layer_norm import rms_norm_fn, RMSNorm
from timm.models.layers import DropPath

from .mambar import *
from .mambar import create_block, get_cls_idx
import numpy as np
import math
import random
import einops
from copy import deepcopy

import torch.nn.functional as F


logger = logging.getLogger(__name__)


_backbones = {
   'mambar_base_patch16_224': [
       'mambar_base_patch16_224', 
       '/root/data/.cache/models/mambar_base_patch16_224.pth'],
   'mambar_small_patch16_224': [
       'mambar_small_patch16_224', 
        '/root/data/.cache/models/mambar_small_patch16_224.pth'],
}


def get_bb_cls_idxs(old_cls_pos, cls_pos):
    if cls_pos.size(0) > old_cls_pos.size(0):
        # when the number of cls_token is larger than the original one, we need to add some padding to the cls_pos
        assert (cls_pos.size(0) - old_cls_pos.size(0)) % 2 == 0
        t = (cls_pos.size(0) - old_cls_pos.size(0)) // 2
        return t

    old_float_idxs = []
    t = 0
    for i in old_cls_pos:
        old_float_idxs.append((i-0.5-t).item())
        t += 1
        
    new_float_idxs = []
    t = 0
    for i in cls_pos:
        new_float_idxs.append(i-0.5-t)
        t += 1
        
    idxs = []
    old_float_np = np.array(old_float_idxs)
    for i in new_float_idxs:
        idx = np.argmin(np.abs(old_float_np-i.item()))
        idxs.append(idx)
        old_float_np[idx] = np.inf

    return torch.LongTensor(idxs)
        

class GeneralizedMean(nn.Module):
    def __init__(self, norm=3, eps=1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * norm)
        self.eps = eps
        
    def forward(self, x):
        # B N R D
        x = x.clamp(min=self.eps).pow(self.p)
        return x.mean(dim=2).pow(1. / self.p)
        
        
        


def get_oth_pos(num_patches, cls_pos):
    ori_indices = torch.arange(num_patches + cls_pos.size(0))
    pre_i = 0
    other_positions_lists = []
    for i in cls_pos:
        other_positions_lists.append(ori_indices[pre_i:i])
        pre_i = i + 1
    other_positions_lists.append(ori_indices[pre_i:])
    return torch.cat(other_positions_lists)


@MODEL_REGISTRY.register()
class ReIDMambaR(nn.Module):
    def __init__(self, backbone_name='mambar_base_patch16_224', num_classes=751, img_size=224, patch_size=16, stride_size=16,
                 in_chans=3, drop_path_rate=0.1, num_cls_tokens=8, cls_reduce=4, num_branches=1, token_fusion_type='max',
                 use_cid=False, num_cids=0, sie_xishu =3.0,  *args, **kwargs):
        super().__init__()
        
        name, path = _backbones[backbone_name]
        bb = create_model(name)
        bb.load_state_dict(torch.load(path)['model'])
        logger.info('loading backbone from {}'.format(backbone_name))
        logger.info('\t embedding dim is : {}'.format(bb.embed_dim))
        logger.info('\t number of cls_token is : {}'.format(num_cls_tokens))
        logger.info('\t number of layers is : {}'.format(bb.depth))
        logger.info('\t reduction factor is : {}'.format(cls_reduce))
        logger.info('\t finale feature dim is : {}'.format(bb.embed_dim // cls_reduce  * num_cls_tokens * num_branches))
        # logger.info('\t finale feature dim is : {}'.format(bb.embed_dim // cls_reduce  * sum([num_cls_tokens // 2**b for b in range(num_branches)])))
        
        self.num_classes = num_classes
        self.num_features = self.embed_dim = bb.embed_dim  # num_features for consistency with other models
        self.use_cid = use_cid
        self.cls_reduce = cls_reduce
        self.num_cls_tokens = num_cls_tokens
        self.num_branches = num_branches
        self.token_fusion_type = token_fusion_type
        self.stride_size = stride_size
        if token_fusion_type == 'gem':
            self.gems = nn.ModuleList([GeneralizedMean(norm=3) for _ in range(2*(num_branches-1))])
        
        
        
        # patch embedding
        self.patch_embed = PatchEmbed_overlap(
            img_size=img_size, 
            patch_size=patch_size, 
            stride_size=stride_size, 
            in_chans=in_chans, 
            embed_dim=self.embed_dim
        )
        self.patch_embed.proj.load_state_dict(bb.patch_embed.proj.state_dict())
        num_patches = self.patch_embed.num_patches
        # initialize cls token and position embedding from Mamba-R
        _, bb_cls_positions = get_cls_idx(self.patch_embed.num_y, self.patch_embed.num_x, bb.num_cls_tokens)
        self.token_idx, self.cls_idx = get_cls_idx(self.patch_embed.num_y, self.patch_embed.num_x, num_cls_tokens)
        self.cls_token =nn.Parameter(torch.zeros(1, num_cls_tokens, self.embed_dim))
        self.pos_embed_cls = nn.Parameter(torch.zeros(1, num_cls_tokens, self.embed_dim))
        trunc_normal_(self.cls_token.data, std=.02)
        trunc_normal_(self.pos_embed_cls.data, std=.02)
        self.oth_idx = get_oth_pos(num_patches, self.cls_idx)
        # copy cls token from Mamba-R
        with torch.no_grad():
            idxs = get_bb_cls_idxs(bb_cls_positions, self.cls_idx)
            if isinstance(idxs, torch.Tensor):
                self.cls_token.data.copy_(bb.cls_token.data[:, idxs])
                self.pos_embed_cls.data.copy_(bb.pos_embed_cls.data[:, idxs])
            else:
                self.cls_token.data[:,idxs:-idxs] = bb.cls_token.data
                self.pos_embed_cls.data[:,idxs:-idxs] = bb.pos_embed_cls.data
        # copy position embedding from Mamba-R
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        inter_pos_embed = resample_abs_pos_embed(
                            posemb=bb.pos_embed, 
                            new_size=[self.patch_embed.num_y, self.patch_embed.num_x], 
                            num_prefix_tokens=0,
                            verbose=True,
                            )
        with torch.no_grad():
            self.pos_embed.data = inter_pos_embed.data
        
        # Initialize SIE Embedding
        self.cam_num = num_cids
        self.sie_xishu = sie_xishu        
        if self.use_cid:
            self.sie_embed = nn.Parameter(torch.zeros(num_cids, 1, self.embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            logger.info('camera number is : {}'.format(num_cids))
            logger.info('using SIE_Lambda is : {}'.format(sie_xishu))
        # drop path rate
        logger.info('using drop_path rate is : {}'.format(drop_path_rate))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, bb.depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.layers = nn.ModuleList(
            [
                create_block(
                    self.embed_dim,
                    ssm_cfg=None, # None
                    norm_epsilon=1e-5, # 1e-5
                    rms_norm=True, # True
                    residual_in_fp32=True, # True
                    fused_add_norm=True, # True
                    layer_idx=i,
                    drop_path=inter_dpr[i]
                )
                for i in range(bb.depth - 2)
            ]
        )
        for i in range(bb.depth - 2):
            self.layers[i].load_state_dict(bb.layers[i].state_dict())

        base_layer = nn.ModuleList(
            [
                create_block(
                    self.embed_dim,
                    ssm_cfg=None, # None
                    norm_epsilon=1e-5, # 1e-5
                    rms_norm=True, # True
                    residual_in_fp32=True, # True
                    fused_add_norm=True, # True
                    layer_idx=i,
                    drop_path=inter_dpr[i]
                ) for i in range(bb.depth - 2, bb.depth)
            ]
        )
        for i in range(bb.depth - 2, bb.depth):
            base_layer[i-bb.depth+2].load_state_dict(bb.layers[i].state_dict())
        
        self.multi_layers = nn.ModuleList()
        self.norm_fs = nn.ModuleList()
        self.necks = nn.ModuleList()
        self.norm_necks = nn.ModuleList()
        self.bnnecks = nn.ModuleList()
        
        self.down_token_idx = []
        self.down_cls_idx = []
        sampling_rate = 1
        for b in range(num_branches):
            self.multi_layers.append(deepcopy(base_layer))
            self.norm_fs.append(deepcopy(bb.norm_f))
            self.necks.append(nn.Linear(self.embed_dim, self.embed_dim // cls_reduce * sampling_rate, bias=False))
            trunc_normal_(self.necks[b].weight, std=0.02)
            self.norm_necks.append(RMSNorm((self.embed_dim // cls_reduce)  * num_cls_tokens, eps=1e-5))
            self.bnnecks.append(BNNeck((self.embed_dim // cls_reduce)  * num_cls_tokens, num_classes, False, pool=None, neck_feat='before', init_mode=0))

            if b:
                token_idx, cls_idx = get_cls_idx(self.patch_embed.num_y, self.patch_embed.num_x, num_cls_tokens//sampling_rate)
                self.down_token_idx.append(token_idx)
                self.down_cls_idx.append(cls_idx)
                
            sampling_rate *= 2

    def forward_features(self, x, cids=None, get_tokens=False):      
        x = self.patch_embed(x)
        B, _, _ = x.shape

        x = x + self.pos_embed
        
        cls_token = self.cls_token.expand(B, -1, -1) + self.pos_embed_cls
        x = torch.cat([x, cls_token], dim=1)[:, self.token_idx]
        
        if self.use_cid:
            x = x + self.sie_embed[cids] * self.sie_xishu    

        residual = None
        hidden_states = x
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual
            )
        
        hidden_states_cls = []
        sampling_rate = 1
        for b in range(self.num_branches):
            if b == 0:
                _hidden_states, _residual = hidden_states, residual
            else:
                _residual = None
                _hidden_states = residual + self.multi_layers[b][0].drop_path(hidden_states)
                _hidden_states_cls, _hidden_states_oth = _hidden_states[:, self.cls_idx], _hidden_states[:, self.oth_idx]
                if self.token_fusion_type =='max':
                    _hidden_states_cls = torch.max(_hidden_states_cls.view(B, self.num_cls_tokens//sampling_rate, sampling_rate, -1), dim=2)[0]
                elif self.token_fusion_type == 'avg':
                    _hidden_states_cls = torch.mean(_hidden_states_cls.view(B, self.num_cls_tokens//sampling_rate, sampling_rate, -1), dim=2)
                elif self.token_fusion_type == 'gem':
                    _hidden_states_cls = self.gems[2*b-2](_hidden_states_cls.view(B, self.num_cls_tokens//sampling_rate, sampling_rate, -1))
                else:
                    raise NotImplementedError
                _hidden_states = torch.cat([_hidden_states_oth, _hidden_states_cls], dim=1)[:, self.down_token_idx[b-1]]
                
                # hidden_states and residual are fusion respectively!
                # _hidden_states_cls, _residual_cls = hidden_states[:, self.cls_idx], residual[:, self.cls_idx]
                # _hidden_states_oth, _residual_oth = hidden_states[:, self.oth_idx], residual[:, self.oth_idx]
                # if self.token_fusion_type =='max':
                #     _hidden_states_cls = torch.max(_hidden_states_cls.view(B, self.num_cls_tokens//sampling_rate, sampling_rate, -1), dim=2)[0]
                #     _residual_cls = torch.max(_residual_cls.view(B, self.num_cls_tokens//sampling_rate, sampling_rate, -1), dim=2)[0]
                # elif self.token_fusion_type == 'avg':
                #     _hidden_states_cls = torch.mean(_hidden_states_cls.view(B, self.num_cls_tokens//sampling_rate, sampling_rate, -1), dim=2)
                #     _residual_cls = torch.mean(_residual_cls.view(B, self.num_cls_tokens//sampling_rate, sampling_rate, -1), dim=2)
                # elif self.token_fusion_type == 'gem':
                #     _hidden_states_cls = self.gems[2*b-2](_hidden_states_cls.view(B, self.num_cls_tokens//sampling_rate, sampling_rate, -1))
                #     _residual_cls = self.gems[2*b-1](_residual_cls.view(B, self.num_cls_tokens//sampling_rate, sampling_rate, -1))
                # else:
                #     raise NotImplementedError
                # _hidden_states, _residual = torch.cat([_hidden_states_oth, _hidden_states_cls], dim=1)[:, self.down_token_idx[b-1]], torch.cat([_residual_oth, _residual_cls], dim=1)[:, self.down_token_idx[b-1]]
            
            for layer in self.multi_layers[b]:
                _hidden_states, _residual = layer(
                    _hidden_states, _residual
                )
            _hidden_states = rms_norm_fn(
                self.drop_path(_hidden_states),
                self.norm_fs[b].weight,
                self.norm_fs[b].bias,
                eps=self.norm_fs[b].eps,
                residual=_residual,
                prenorm=False, # set to False, not return the residual
                residual_in_fp32=True,
            )
            
            if b == 0:
                _hidden_states_cls = self.necks[b](_hidden_states[:, self.cls_idx])
            else:
                _hidden_states_cls = self.necks[b](_hidden_states[:, self.down_cls_idx[b-1]])
            _hidden_states_cls = self.norm_necks[b](_hidden_states_cls.view(B, -1))
            hidden_states_cls.append(_hidden_states_cls)
            sampling_rate *= 2
            
        return hidden_states_cls
        
    def forward(self, x, cids=None, get_tokens=False, *args, **kwargs):
        fs = self.forward_features(x, cids, get_tokens)
        
        if not self.training:
            return torch.cat([F.normalize(f) for f in fs], dim=1)
            # ff = 0.0
            # for f in fs:
            #     ff += f
            # return ff
            
        else:
            tri = []
            log = []
            for i,f in enumerate(fs):
                res = self.bnnecks[i](f)
                tri.append(res[0][0])
                log.append(res[1][0])
            return tri, log, [[F.normalize(f) for f in fs]], [[F.normalize(f) for f in fs]]
            # return tri, log, [fs,], [fs,]
        
    def get_params(self, *args, **kwargs):
        no_weight_decay_list = {"pos_embed", "cls_token", "sie_embed", "pos_embed_cls"}
        
        params = []
        for k, v in self.named_parameters():
            if not v.requires_grad:
                continue
            
            if k in no_weight_decay_list or k.endswith(".A_log") or k.endswith(".D") or k.endswith(".A_b_log") or k.endswith(".D_b") or v.ndim <= 1 or k.endswith(".bias"):     
                params += [{"params": [v], "weight_decay": 0}]
                
            else:
                params += [{"params": [v]}]
                
                
                # if 'bnneck.' not in k and 'norm_neck.' not in k:
                #     params[-1]["lr"] = 5e-3
                
            # if k.endswith(".bias") or v.ndim <= 1 or k.endswith(".A_log") or k.endswith(".D") or k.endswith(".A_b_log") or k.endswith(".D_b"):
            #     params[-1]["lr"] = 0.04
            #     logger.info(f"lr of {k} is set to {params[-1]['lr']}")
                
                
        return params
    
    def freeze_backbone(self):
        for n, p in self.named_parameters():
            if 'bnneck.' not in n and 'norm_neck.' not in n:
                p.requires_grad_(False)
                
    def unfreeze_backbone(self):
        for n, p in self.named_parameters():
            if 'bnneck.' not in n and 'norm_neck.' not in n:
                p.requires_grad_(True)
                
    def eval_backbone(self):
        for name, child in self.named_children():
            if name != "bnneck":
                child.eval()

    def train_backbone(self):
        for name, child in self.named_children():
            if name != "bnneck":
                child.train()