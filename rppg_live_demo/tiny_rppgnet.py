import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyRPPGNet(nn.Module):
    """
    Custom lightweight 3D CNN for rPPG-based HR regression.
    Input : (B, 3, T, H, W)
    Output: (B,) BPM
    """
    def __init__(self, in_ch=3):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, 16, kernel_size=(3,5,5),
                               stride=(1,2,2), padding=(1,2,2))
        self.bn1   = nn.BatchNorm3d(16)

        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3,3,3),
                               stride=(1,2,2), padding=(1,1,1))
        self.bn2   = nn.BatchNorm3d(32)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3,3,3),
                               stride=(1,2,2), padding=(1,1,1))
        self.bn3   = nn.BatchNorm3d(64)

        self.conv4 = nn.Conv3d(64, 64, kernel_size=(3,3,3),
                               stride=(1,1,1), padding=(1,1,1))
        self.bn4   = nn.BatchNorm3d(64)

        self.pool  = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc1   = nn.Linear(64, 32)
        self.drop  = nn.Dropout(0.3)
        self.fc2   = nn.Linear(32, 1)

    def forward(self, x):
        # x: (B, 3, T, H, W)
        x = F.relu(self.bn1(self.conv1(x)))     # (B,16,...)
        x = F.relu(self.bn2(self.conv2(x)))     # (B,32,...)
        x = F.relu(self.bn3(self.conv3(x)))     # (B,64,...)
        x = F.relu(self.bn4(self.conv4(x)))     # (B,64,...)
        x = self.pool(x).view(x.size(0), -1)    # (B,64)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x).squeeze(1)              # (B,)
        return x
