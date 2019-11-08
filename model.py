import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np

class RoofNet(nn.Module):
    def __init__(self):
        super(RoofNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, 5, 1) 
        #self.conv1_bn = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 128, 5, 1)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, 1) 
        self.conv3_bn = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(25*25*256, 500)
        self.fc1_drop = nn.Dropout(p = 0.25)
        self.fc2 = nn.Linear(500, 100)
        self.fc2_drop = nn.Dropout(p = 0.25)
        self.fc3 = nn.Linear(100, 40)
        self.fc3_drop = nn.Dropout(p = 0.25)
        self.fc4 = nn.Linear(40, 5)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.elu(self.conv2(x))
        x = F.max_pool2d(self.conv2_bn(x), 2, 2)
        x = F.elu(self.conv3(x))
        x = F.max_pool2d(self.conv3_bn(x), 2, 2)
        x = x.view(-1, self.num_flat_features(x))        
        x = F.elu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = F.relu(self.fc3(x))
        x = self.fc3_drop(x)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1) 

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension [batch_size, features, width, height]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class RoofEnsemble(nn.Module):
    def __init__(self):
        super(RoofEnsemble, self).__init__()
        self.init_branches()
        self.mlp = nn.Sequential(nn.Linear(2, 64),
                                    nn.BatchNorm1d(64),
                                    nn.LeakyReLU(0.3),
                                    nn.Dropout(0.25),
                                    nn.Linear(64, 64),
                                    nn.BatchNorm1d(64),
                                    nn.LeakyReLU(0.3))#,
                                    # nn.Dropout(0.25),
                                    # nn.Linear(256, 256))
        self.classifier = nn.Sequential(nn.Linear(2048*1 + 64*1, 5))
        # self.classifier = nn.Sequential(nn.Linear(2048*1 + 64*1, 512),
        #                             nn.LeakyReLU(0.3),
        #                             nn.Dropout(0.25),
        #                             nn.Linear(512, 64),
        #                             nn.LeakyReLU(0.3),
        #                             nn.Dropout(0.25),
        #                             nn.Linear(64, 5))
        
    def forward(self, x1, x2, x3):             
        x1 = self.modelA(x1)
        #x2 = self.modelB(x2)
        x3 = self.mlp(x3)
        x1 = torch.flatten(x1, start_dim=1)
        #x2 = torch.flatten(x2, start_dim=1)
        x = torch.cat((x1, x3), dim=1)
        x = self.classifier(x)

        return x #F.log_softmax(x, dim=1) 

    def init_branches(self):
        # Load pretrained models
        model_roof = models.resnet101(pretrained=True)
        model_context = models.resnet101(pretrained=True)
        
        #model_roof.classifier = model_roof.classifier[:-1]
        #model_context.classifier = model_context.classifier[:-1]

        self.modelA = nn.Sequential(*list(model_roof.children()))[:-1]
        self.modelB = nn.Sequential(*list(model_context.children()))[:-1]

        for param in self.modelA.parameters():
            param.requires_grad = False
        for param in self.modelB.parameters():
             param.requires_grad = False

