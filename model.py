import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


def crop_image(image, crop_shape):
    B, C, H, W = image.shape

    height_start = (H - crop_shape[0]) // 2
    height_end = height_start + crop_shape[0]

    width_start = (W - crop_shape[1]) // 2
    width_end = width_start + crop_shape[1]

    return image[:, :, height_start:height_end, width_start:width_end]




class ContractingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContractingBlock, self).__init__()
        self.contracting = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=1, padding=0),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.contracting(x)


class ExpansiveBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExpansiveBlock, self).__init__()

    
        self.transpose =  nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2,2), stride=2)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=1, padding=0)
       
    def forward(self, x, y):
        x = self.transpose(x)
        if x.shape != y.shape:
            y = crop_image(y, x.shape[2:])
        x = torch.cat([x, y], dim=1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        return x


class FeatureBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureBlock, self).__init__()

        self.final = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1,1)),
        )
    
    def forward(self, x):
        return self.final(x)
    



class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.channels = 64
        self.down1 = ContractingBlock(in_channels, self.channels)
        self.down2 = ContractingBlock(self.channels, self.channels * 2)
        self.down3 = ContractingBlock(self.channels * 2, self.channels * 4)
        self.down4 = ContractingBlock(self.channels * 4, self.channels * 8)
        self.pool = nn.MaxPool2d((2,2), 2)  

        self.bottom = ContractingBlock(self.channels*8, self.channels*16)

        self.up1 = ExpansiveBlock(self.channels *16, self.channels * 8)
        self.up2 = ExpansiveBlock(self.channels * 8, self.channels * 4)
        self.up3 = ExpansiveBlock(self.channels * 4, self.channels * 2)
        self.up4 = ExpansiveBlock(self.channels * 2, self.channels)
        
        self.final = FeatureBlock(self.channels, out_channels)

    def forward(self, x):
        down_1 = self.down1(x) # in 1 out 64 
        pool_1 = self.pool(down_1)
        down_2 = self.down2(pool_1) # in 64 out 128 
        pool_2 = self.pool(down_2) 
        down_3 = self.down3(pool_2) # in 128 out 256 
        pool_3 = self.pool(down_3) 
        down_4 = self.down4(pool_3) # in 256 out 512
        pool_4 = self.pool(down_4)
        bottom = self.bottom(pool_4) # in 512 out 1024

        up_1 = self.up1(bottom, down_4) # in 1024 out 512
        up_2 = self.up2(up_1, down_3) #  in 512 out 256
        up_3 = self.up3(up_2, down_2) #  in 256 out 128
        up_4 = self.up4(up_3, down_1) # in 128 out 64

        return self.final(up_4)
    

from torchinfo import summary
net = UNet(1, 1)
summary(net, (1, 1, 512, 512), depth=1)


