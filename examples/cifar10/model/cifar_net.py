# @file cifar_net.py
# @brief FP32 model for CIFAR-10: two conv layers that make 64 tokens, one SimA attention layer, and a linear classifier

from torch import nn

import torch

from .sima_layer import SimaLayer



class CifarNet(nn.Module):
    '''
    #* FP32 CNN + attention, small enough to train on a CPU.
    #? main.py trains it first; its weights, with the BatchNorms folded in, start the QAT of QuantCifarNet.
    '''
    
    def __init__(self) -> None:
        '''
        #* Make the layers: 32 x 32 to 16 x 16 with 32 channels, to 8 x 8 with 64 (the tokens), then the attention and the classifier.
        '''
        
        super().__init__()
        self.conv1      = nn.Conv2d(3 , 32, 3, 2, 1, bias=False)
        self.bn1        = nn.BatchNorm2d(32)
        self.conv2      = nn.Conv2d(32, 64, 3, 2, 1, bias=False)
        self.bn2        = nn.BatchNorm2d(64)
        self.block      = SimaLayer(64, 4, 16, 16)
        self.classifier = nn.Linear(64 * 8 * 8, 10)
        return None
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        '''
        #* Return the logits of uint8 images (B, 3, 32, 32).
        '''
        
        x = torch.relu(self.bn1(self.conv1(images.float() / 255)))
        x = self.bn2(self.conv2(x))
        x = self.block(x)
        return self.classifier(x.flatten(1))
