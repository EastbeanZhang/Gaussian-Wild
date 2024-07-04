import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18,resnet34,resnet50,resnet101
import copy

#Unet_model
#-----------------------------------------------------------------------------------------
from collections import OrderedDict
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class OneConv(nn.Module):
    def __init__(self, in_channels, out_channels,not_act=False):
        super(OneConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        if not_act:
            self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return self.conv(x)
    
class IntermediateLayerGetter(nn.ModuleDict):

    def __init__(self,model,return_layers):

        orig_return_layers=return_layers
        return_layers={k: v for k,v in return_layers.items()}
        layers=OrderedDict()
        #
        for name,module in model.named_children():
            layers[name]=module
            #
            if name in return_layers:
                del return_layers[name]
            #
            if not return_layers:
                break

        super(IntermediateLayerGetter,self).__init__(layers)
        self.return_layers=orig_return_layers

    def forward(self,x):
        out=OrderedDict()
        #
        for name,module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                #
                out_name=self.return_layers[name]

                out[out_name]=x
        return out
    
class Unet_model(nn.Module):
    def __init__(self,features_dim=32,backbone="resnet18",use_features_mask=False,use_independent_mask_branch=False):
        super().__init__()
        features_dim=features_dim

        #
        if backbone=="resnet18":
            resnet=resnet18(weights=torchvision.models.ResNet18_Weights)
        elif backbone=="resnet34":
            resnet = resnet34(weights=torchvision.models.ResNet34_Weights)
        elif backbone=="resnet50":
            resnet = resnet50(weights=torchvision.models.ResNet50_Weights)
        elif backbone=="resnet101":
            resnet = resnet101(weights=torchvision.models.ResNet101_Weights)
        #
        backbone=nn.Sequential(*list(resnet.children())[:-2])#-2 16 16  -3 32 32
        return_layers={'0': 'low','4': '64','5': '128','6': '256',"7":"512"}
        self.encoder=IntermediateLayerGetter(backbone,return_layers=return_layers)
        backbone_channels,self.backbone_channels = [ 64,64, 128, 256, 512],[ 64,64, 128, 256, 512]
        
        self.decoder=nn.ModuleList()
        for i in range(len(backbone_channels) - 2, -1, -1):
            self.decoder.append(
                nn.ConvTranspose2d(backbone_channels[i+1], backbone_channels[i], kernel_size=2, stride=2),
                # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                # nn.Conv2d(backbone_channels[i]*2, backbone_channels[i], kernel_size=3, stride=1,padding=1)
            )
            self.decoder.append(DoubleConv(backbone_channels[i]*2, backbone_channels[i]))
        
        self.final_conv = nn.Conv2d(backbone_channels[0], features_dim, kernel_size=1)
        
        self.use_features_mask=use_features_mask
        if use_features_mask:
            self.use_independent_mask_branch=use_independent_mask_branch
            if use_independent_mask_branch:
                print("unet:use_independent_mask_branch")
                self.mask_encoder=copy.deepcopy(self.encoder)
            
            self.mask_decoder=nn.Sequential(
                nn.ConvTranspose2d(backbone_channels[-1], backbone_channels[-2], kernel_size=2, stride=2),
                OneConv(backbone_channels[-2],backbone_channels[-2]),
                nn.ConvTranspose2d(backbone_channels[-2], backbone_channels[-3], kernel_size=2, stride=2),
                OneConv(backbone_channels[-3],backbone_channels[-3]),
                nn.ConvTranspose2d(backbone_channels[-3], backbone_channels[-4], kernel_size=2, stride=2),
                OneConv(backbone_channels[-4],backbone_channels[-4]),
                nn.Conv2d(backbone_channels[-4], 1, kernel_size=1),
                nn.Sigmoid()
            )

    
    def forward(self, inx,only_gloabal_features=False,eval_mode=False):
        out={}
        features=self.encoder(inx)
        x=features["512"]
        if self.use_features_mask:
            infm=features["512"]
            
            if self.use_independent_mask_branch:
                infm=self.mask_encoder(inx)["512"]
            out["mask"]=self.mask_decoder(infm)

        
        for i in range(0, len(self.decoder)-1, 2):
            x = self.decoder[i](x)
            x0=features[str(self.backbone_channels[-i//2-2])]
            x=torch.nn.functional.interpolate(x,size=(x0.shape[-2],x0.shape[-1]))
            x = torch.cat([x,x0] , dim=1)
            x = self.decoder[i+1](x)
        x = self.decoder[-2](x)
        x0=features[str("low")]
        x=torch.nn.functional.interpolate(x,size=(x0.shape[-2],x0.shape[-1]))
        x = torch.cat([x, x0], dim=1)
        x = self.decoder[-1](x)

        out["feature_maps"] = self.final_conv(x)
            
        return out

    
if __name__=="__main__":
    inx = torch.rand((1,3,413,513))
    fcn_model=Unet_model()
    out=fcn_model(inx)
    pass

