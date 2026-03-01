import torch.nn as nn
import torch


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Adds channel-wise attention to a convolutional block.
    """
    def __init__(self, channels, reduction_ratio=16):
        """
        Initializes the SE Block.
        Args:
            channels (int): Number of input channels.
            reduction_ratio (int): Factor by which to reduce channels
                                   in the intermediate layer. Default: 16.
        """
        super(SEBlock, self).__init__()
        if channels <= reduction_ratio:
             # Avoid reducing channels to zero or negative numbers
             reduced_channels = channels // 2 if channels > 1 else 1
        else:
            reduced_channels = channels // reduction_ratio

        # Squeeze operation: Global Average Pooling
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        # Excitation operation: Two Linear layers
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the SE Block.
        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width).
        Returns:
            torch.Tensor: Output tensor, input scaled by channel-wise attention weights.
        """
        batch_size, num_channels, _, _ = x.size()

        # Squeeze: (batch, channels, height, width) -> (batch, channels, 1, 1)
        squeezed = self.squeeze(x)
        # Reshape for Linear layers: (batch, channels, 1, 1) -> (batch, channels)
        squeezed = squeezed.view(batch_size, num_channels)

        # Excitation: (batch, channels) -> (batch, channels)
        channel_weights = self.excitation(squeezed)
        # Reshape weights for scaling: (batch, channels) -> (batch, channels, 1, 1)
        channel_weights = channel_weights.view(batch_size, num_channels, 1, 1)

        # Scale: Multiply original input by the learned channel weights
        scaled_output = x * channel_weights
        return scaled_output
    



class SEBottleneck(nn.Module):
    """
    Bottleneck block for ResNet-50/101/152
    """
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction_ratio=16):
        super(SEBottleneck, self).__init__()
        
        # 1x1 conv to reduce channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 conv with stride
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 conv to expand channels
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.se_block = SEBlock(out_channels, reduction_ratio)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out
    




    

class ResNetSE(nn.Module):
    """
    ResNet-50 implementation
    """
    def __init__(self, num_classes=3):
        super().__init__()
        
        self.in_channels = 64
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, 
                              padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(SEBottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(SEBottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(SEBottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(SEBottleneck, 512, 3, stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * SEBottleneck.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        
        # Create downsample layer if needed
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
            
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

