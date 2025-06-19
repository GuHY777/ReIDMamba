import torch
from torch import nn
from .resnets import ResNet
from einops import rearrange
from timm.layers import trunc_normal_, lecun_normal_
import logging
from .build import MODEL_REGISTRY
import torch.nn.functional as F

logger = logging.getLogger(__name__)  


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight) 


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('NormLinear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class NormLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        
        nn.init.normal_(self.weight, std=0.001)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)
    
    def forward(self, x):
        return F.linear(F.normalize(x, p=2, dim=1), F.normalize(self.weight, p=2, dim=1), self.bias)


class BNNeck(nn.Module):
    def __init__(self, in_planes, num_classes, bias=True, pool='avg', neck_feat='after', init_mode=0, softmax_loss_only=False, normalize=False):
        super(BNNeck, self).__init__()
        
        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            self.pool = nn.Identity()    

        self.bn = nn.BatchNorm1d(in_planes)
        self.bn.apply(weights_init_kaiming)
        self.bn.bias.requires_grad_(False)  # no shift
        
        self.cls = nn.Linear(in_planes, num_classes, bias) if not normalize else NormLinear(in_planes, num_classes, False)
        if init_mode == 0:
            self.cls.apply(weights_init_classifier)
        elif init_mode == 1:
            self.cls.apply(segm_init_weights)
        else:
            raise ValueError('Unsupported initialization mode: {}'.format(init_mode))
        
        self.neck_feat = neck_feat
        self.softmax_loss_only = softmax_loss_only
        
    def forward(self, x):
        f0 = self.pool(x).flatten(1) # B C
        f1 = self.bn(f0)
        if self.training:
            logits = self.cls(f1)
            if self.softmax_loss_only:
                return [[logits, ],]
            return [f0, ], [logits, ]
        else:
            if self.neck_feat == 'before':
                return f0
            elif self.neck_feat == 'after':
                return f1
            else:
                raise ValueError('Unsupported neck feature type: {}'.format(self.neck_feat))
        
@MODEL_REGISTRY.register()
class BoT(nn.Module):
    def __init__(self, backbone_name='resnet50', num_classes=751, cls_bias=True, *args, **kwargs):
        super(BoT, self).__init__()
        
        self.backbone = ResNet(backbone_name, last_stride=1, pretrained=True)
        logger.info('BoT model created with backbone: {}'.format(backbone_name))
        self.bnneck = BNNeck(self.backbone.num_features, num_classes, cls_bias)
        logger.info('BoT model created with BNneck classifier with bias: {}'.format(cls_bias))
        
        self.use_cid = False
        
    def forward(self, x, *args, **kwargs):
        x = self.backbone(x)
        return self.bnneck(x)
    
    def get_params(self, *args, **kwargs):
        return [
            {
                'params': self.parameters(),
                'lr': 3.5e-4,
                'weight_decay': 5e-4
            }
        ]
    
    

        