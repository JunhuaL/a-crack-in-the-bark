import torch
import torch.nn as nn
import torch.nn.functional as F
import complexPyTorch.complexLayers as cnn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1, dtype=None):
        super(Block, self).__init__()
        block = []
        if dtype is not None:
            norm = cnn.ComplexBatchNorm2d
            relu = cnn.ComplexReLU
            dropout = cnn.ComplexDropout
        else:
            norm = nn.BatchNorm2d
            relu = nn.ReLU
            dropout = nn.Dropout
            
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, dtype=dtype))
        block.append(norm(out_channels))
        block.append(relu())
        block.append(dropout(p=0.1))
        block.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype))
        block.append(norm(out_channels))
        self.block = nn.Sequential(*block)
        self.relu = relu()
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        x = self.block(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x
    
class ResNet_18(nn.Module):
    def __init__(self, image_channels, num_classes, dtype=None):
        super(ResNet_18, self).__init__()
        self.dtype = dtype
        # Entry Block
        entry_block = []
        entry_block.append(nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3,dtype=dtype))
        entry_block.append(nn.BatchNorm2d(64) if dtype is None else cnn.ComplexBatchNorm2d(64))
        entry_block.append(nn.ReLU() if dtype is None else cnn.ComplexReLU())
        entry_block.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if dtype is None else cnn.ComplexMaxPool2d(kernel_size=3, stride=2, padding=1))
        self.entry_block = nn.Sequential(*entry_block)

        # ResNet layers
        nn_list = []
        layer_sizes = [(64,64,1),(64,128,2),(128,256,2),(256,512,2)]
        for size in layer_sizes:#
            nn_list.append(self.__make_layer(size[0], size[1], stride=size[2]))
        
        self.resnet_blocks = nn.Sequential(*nn_list)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes,dtype=dtype)
        
    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
            
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride,dtype=self.dtype), 
            Block(out_channels, out_channels, dtype=self.dtype)
        )
        
    def forward(self, x):
        x = self.entry_block(x)
        x = self.resnet_blocks(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x 
    
    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,dtype=self.dtype), 
            nn.BatchNorm2d(out_channels) if self.dtype is None else cnn.ComplexBatchNorm2d(out_channels)
        )