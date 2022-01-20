import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from models import modules, senet
import torchvision


class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()        
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB( self.convB( self.convA( torch.cat([up_x, concat_with], dim=1)  ) )  )

class Decoder(nn.Module):
    def __init__(self, num_features=1024, decoder_width = 1.0):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)

        self.up1 = UpSample(skip_input=features//1 + features//2 +1 , output_features=features//2)
        self.up2 = UpSample(skip_input=features//2 + features//4 +1,  output_features=features//4)
        self.up3 = UpSample(skip_input=features//4 + features//8 +1,  output_features=features//8)
        
        self.conv3 = nn.Conv2d(features//8  , 1, kernel_size=3, stride=1, padding=1)

    def forward(self, i,x_block0, x_block1, x_block2, x_block3,x_block4):

        x_d0 = self.conv2(x_block3)
        
        Generate_depth1= nn.Conv2d(x_d0.size(1) , 1, kernel_size=3, stride=1, padding=1).cuda()
        depth1 = F.relu(Generate_depth1(x_d0))

        max0 = torch.cat([x_d0,depth1], dim=1)

        x_d1 = self.up1(max0,x_block2)

        Generate_depth2= nn.Conv2d(x_d1.size(1) , 1, kernel_size=3, stride=1, padding=1).cuda()
        depth2 = F.relu(Generate_depth2(x_d1))

        max1 = torch.cat([x_d1,depth2], dim=1)

        x_d2 = self.up2(max1,x_block1)

        Generate_depth3 = nn.Conv2d(x_d2.size(1) , 1, kernel_size=3, stride=1, padding=1).cuda()
        depth3 = F.relu(Generate_depth3(x_d2))

        max2 = torch.cat([x_d2,depth3], dim=1)

        x_d3 = self.up3(max2,x_block0)
        
        return F.interpolate(self.conv3(x_d3), (180, 320))

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()       
        self.original_model = senet.senet154(pretrained='imagenet')
        self.base = nn.Sequential(*list(self.original_model.children())[:-3])

    def forward(self, x):
        x = self.base[0](x)
        x_block1 = self.base[1](x)
        x_block2 = self.base[2](x_block1)
        x_block3 = self.base[3](x_block2)
        x_block4 = self.base[4](x_block3)

        return x, x_block1, x_block2, x_block3, x_block4


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self,i,x):
        x_block0, x_block1, x_block2, x_block3 , x_block4= self.encoder(x)

        return self.decoder(i, x_block0, x_block1, x_block2, x_block3, x_block4)
