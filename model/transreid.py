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
from timm.models.vision_transformer import LayerScale, Mlp, DropPath # we can directly use the Block, but it will use the F.scaled_dot_product which will cause randomness in the training!
from .bot import BNNeck
from copy import deepcopy
from functools import partial


logger = logging.getLogger(__name__)   

_backbones = {
   'vit_base_16_224': ['vit_base_patch16_224.orig_in21k_ft_in1k', '/root/data/.cache/models/jx_vit_base_p16_224-80ecf9dd.pth'],
}


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x



def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        logger.info('using stride: {}, and patch number is num_y{} * num_x{}'.format(stride_size, self.num_y, self.num_x))
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2) # B L D
        return x


@MODEL_REGISTRY.register()
class TransReID(nn.Module):
    def __init__(self, backbone_name='vit_base_16_224', num_classes=751, img_size=224, patch_size=16, stride_size=16, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                 in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, 
                 use_cid=True, num_cids=0, sie_xishu =3.0, 
                 local_feature=True, shuffle_groups=2, divide_length=4, rearrange=True, shift_num=5, neck_feat='before', *args, **kwargs):
        super(TransReID, self).__init__()
        
        self.use_cid = use_cid
        
        timm_model_name, timm_pretrained_path = _backbones[backbone_name]
        bb = create_model(timm_model_name, True, pretrained_cfg_overlay={'file': timm_pretrained_path})
         
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.local_feature = local_feature
        self.shuffle_groups = shuffle_groups
        self.divide_length = divide_length
        self.rearrange = rearrange
        self.shift_num = shift_num

        self.patch_embed = PatchEmbed_overlap(
            img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans, embed_dim=embed_dim
            )
        self.patch_embed.load_state_dict(bb.patch_embed.state_dict())
        
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = bb.cls_token # nn.Parameter(torch.zeros(1, 1, embed_dim))
        # trunc_normal_(self.cls_token, std=.02)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        # trunc_normal_(self.pos_embed, std=.02)
        self.pos_embed.data.copy_(
            resample_abs_pos_embed(
                posemb=bb.pos_embed, 
                new_size=[self.patch_embed.num_y, self.patch_embed.num_x], 
                num_prefix_tokens=bb.num_prefix_tokens,
                )
            )
        
        self.cam_num = num_cids
        self.sie_xishu = sie_xishu
        
        # Initialize SIE Embedding
        if self.use_cid:
            self.sie_embed = nn.Parameter(torch.zeros(num_cids, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            logger.info('camera number is : {}'.format(num_cids))
            logger.info('using SIE_Lambda is : {}'.format(sie_xishu))

        logger.info('using drop_out rate is : {}'.format(drop_rate))
        logger.info('using attn_drop_out rate is : {}'.format(attn_drop_rate))
        logger.info('using drop_path rate is : {}'.format(drop_path_rate))
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        blocks = [Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer
            ) for i in range(depth)]
        for bn, bo in zip(blocks, bb.blocks):
            bn.load_state_dict(bo.state_dict())
            
        if self.local_feature:
            self.blocks = nn.ModuleList(blocks[:-1])
            self.b1 = nn.Sequential(
                blocks[-1],
                bb.norm
            )
            self.b2 = deepcopy(self.b1)
            
            logger.info('using local feature')
            logger.info('using shuffle_groups is : {}'.format(shuffle_groups))
            logger.info('using divide_length is : {}'.format(divide_length))
            logger.info('using rearrange is : {}'.format(rearrange))
            logger.info('using shift_num is : {}'.format(shift_num))
            
            self.necks = nn.ModuleList(
                [
                    BNNeck(embed_dim, num_classes, False, pool=None, neck_feat=neck_feat) for _ in range(self.divide_length+1)
                ]
            )
            
        else:
            self.blocks = nn.ModuleList(blocks)
            self.norm = bb.norm
            self.neck = BNNeck(embed_dim, num_classes, False, pool=None, neck_feat=neck_feat)
       
    def forward_features(self, x, cids=None):
        B = x.shape[0]
        x = self.patch_embed(x) # B x L x D

        cls_tokens = self.cls_token.expand(B, -1, -1)  # B x 1 x D
        x = torch.cat((cls_tokens, x), dim=1) # B x (1+L) x D

        if self.use_cid:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[cids]
        else:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x) # B x (1+L) x D
            
        if self.local_feature:
            feats = [self.b1(x)[:, 0]]
            
            feature_length = x.size(1) - 1
            patch_length = feature_length // self.divide_length
            token = x[:, 0:1] # B x D
            
            if self.rearrange:
                x = shuffle_unit(x, self.shift_num, self.shuffle_groups)
            else:
                x = x[:, 1:]
            
            for i in range(self.divide_length):
                feats.append(self.b2(torch.cat([token, x[:, i*patch_length:(i+1)*patch_length]], dim=1))[:,0])
            
            return feats # [B x D, B x D,..., B x D] x divide_length+1
            
        else:
            x = self.norm(x)
            return x[:, 0]
        
    def forward(self, x, cids=None, *args, **kwargs):
        fs = self.forward_features(x, cids) # B x D
        if self.local_feature:
            fs = [self.necks[i](f) for i, f in enumerate(fs)] # [B x C, B x C, ..., B x C]
            if not self.training:
                return torch.cat([fs[0], torch.cat([f/self.divide_length for f in fs[1:]], dim=1)], dim=1) # B x (C*(1+divide_length))
            else:
                # soft_triplet_loss, cross_entropy_loss, [soft_triplet_loss, ...], [cross_entropy_loss, ...]
                return [fs[0][0][0],], [fs[0][1][0],], [f[0][0] for f in fs[1:]], [f[1][0] for f in fs[1:]]
        else:
            return self.neck(fs)
        
    def get_params(self, *args, **kwargs):
        lr = 0.008
        lr_bias = lr * 2
        weight_decay = 0.0001
        
        params = []
        for k, v in self.named_parameters():
            if not v.requires_grad:
                continue
            
            if 'bias' in k:
                params += [{"params": [v], "lr": lr_bias, "weight_decay": 0, "momentum": 0.9}]
            else:
                params += [{"params": [v], "lr": lr, "weight_decay": weight_decay, "momentum": 0.9}]
        
        return params