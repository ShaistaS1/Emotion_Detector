import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        # Feature extraction layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 3 * 3, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        # Input shape: (batch_size, 1, 48, 48)
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))  # 24x24
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))  # 12x12
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))  # 6x6
        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x))))  # 3x3
        
        # Flatten
        x = x.view(-1, 512 * 3 * 3)
        
        # FC layers
        x = self.dropout(F.leaky_relu(self.bn5(self.fc1(x))))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def initialize_model(device='cpu'):
    model = EmotionCNN(num_classes=7)
    model = model.to(device)
    return model