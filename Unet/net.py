import torch.nn as nn
from torch.nn import functional as F
import torch

class Conv_Block(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,padding=1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(.2),
            nn.LeakyReLU(inplace=True),
            
            
            nn.Conv2d(out_channels,out_channels,3,padding=1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(.2),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self,x):
        return self.layer(x)

class DownSample(nn.Module):
    def __init__(self,channel) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self,x):
        return self.layer(x)

class UpSample(nn.Module):
    def __init__(self,channel) -> None:
        super().__init__()
        self.layer = nn.Conv2d(channel,channel//2,1,1,padding_mode='reflect',bias=False)

    def forward(self,x,feature_map):
        up = F.interpolate(x,scale_factor=2,mode='nearest')
        out = self.layer(up)
        return torch.cat((out,feature_map),dim=1)


class Unet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = Conv_Block(3,64)
        
        self.down1 = DownSample(64)
        self.conv2 = Conv_Block(64,128)
        
        self.down2 = DownSample(128)
        self.conv3 = Conv_Block(128,256)
        
        self.down3 = DownSample(256)       
        self.conv4 = Conv_Block(256,512)
        
        self.down4 = DownSample(512)    
        self.conv5 = Conv_Block(512,1024)
        
        self.up1 = UpSample(1024)    
        self.conv6 = Conv_Block(1024,512)
        self.up2 = UpSample(512)    
        self.conv7 = Conv_Block(512,256)        
        self.up3 = UpSample(256)    
        self.conv8 = Conv_Block(256,128)        
        self.up4 = UpSample(128)    
        self.conv9 = Conv_Block(128,64)
        
        self.out = nn.Conv2d(64,3,1,1,padding_mode='reflect',bias=False)
        self.Th = nn.Sigmoid()           
    
    def forward(self,x):
        r1 = self.conv1(x)
        # print("r1",r1.shape)
        r2 = self.conv2(self.down1(r1))
        # print("r2",r2.shape)
        r3 = self.conv3(self.down2(r2))
        # print("r3",r3.shape)
        r4 = self.conv4(self.down3(r3))
        # print("r4",r4.shape)
        r5 = self.conv5(self.down4(r4))
        # print("r5",r5.shape)
        out1 = self.conv6(self.up1(r5,r4))
        # print("out1",out1.shape)
        out2 = self.conv7(self.up2(out1,r3))
        # print("out2",out2.shape)
        out3 = self.conv8(self.up3(out2,r2))
        # print("out3",out3.shape)
        out4 = self.conv9(self.up4(out3,r1))
        # print("out4",out4.shape)
        
        out = self.Th(self.out(out4))
        # print("out",out.shape)
        return out

if __name__ == "__main__":
    x = torch.randn(2,3,256,256)
    net = Unet()
