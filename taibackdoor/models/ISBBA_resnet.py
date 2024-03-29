import torchvision.models as models
import torch


class resnet18_200(torch.nn.Module):
    def __init__(self, num_classes=200):
        super(resnet18_200, self).__init__()
        self.model = models.resnet18(True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.get_features = False

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        features = x
        x = self.model.fc(x)
        if self.get_features:
            return features, x
        return x
