import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class MNIST_Net(nn.Module):
    '''
    MNIST
    - channel : 1
    - input : 28 * 28
    - output : 10
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2,2)
        self._to_linear = None
        self._get_conv_output()
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 10)
 
    def _get_conv_output(self):
        x = torch.zeros(1,1,28,28)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        self._to_linear = x.view(1,-1).size(1)
 
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CIFAR_Net(nn.Module):
    '''
    CIFAR10
    - channel : 3
    - input : 32 * 32
    - output : 10
    '''

    def __init__(self):
        super().__init__()

        # pretrained ResNet18
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # CIFAR용 수정 (중요)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()

        # output 10 classes
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)