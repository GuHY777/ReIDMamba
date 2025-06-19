from timm import create_model
from torch import nn


_backbones = {
   'resnet18': ['resnet18.tv_in1k', '/root/data/.cache/models/resnet18-5c106cde.pth'],
   'resnet34': ['resnet34.tv_in1k','/root/data/.cache/models/resnet34-333f7ec4.pth'],
   'resnet50': ['resnet50.tv_in1k','/root/data/.cache/models/resnet50-19c8e357.pth'],
   'resnet101': ['resnet101.tv_in1k','/root/data/.cache/models/resnet101-5d3b4d8f.pth'],
}

class ResNet(nn.Module):
    def __init__(self, model_name, last_stride=1, pretrained=True):
        super().__init__()
        
        self.model_name = model_name
        self.last_stride = last_stride
        self.pretrained = pretrained
        
        timm_model_name, timm_pretrained_path = _backbones[model_name]
        
        model = create_model(timm_model_name, pretrained, pretrained_cfg_overlay={'file': timm_pretrained_path})
        
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.act1 = model.act1
        self.maxpool = model.maxpool
        
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        
        if last_stride == 1:
            if model_name in ['resnet18','resnet34']:
                self.layer4[0].conv1.stride = (1, 1)
                self.layer4[0].downsample[0].stride = (1, 1)
            else:
                self.layer4[0].conv2.stride = (1, 1)
                self.layer4[0].downsample[0].stride = (1, 1)
            
        self.num_features = model.num_features
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x
    
    def __repr__(self):
        return f'\tlast stride={self.last_stride}\n\tpretrained={self.pretrained}'
    
    
if __name__ == '__main__':
    m = ResNet('resnet50', last_stride=1, pretrained=True)