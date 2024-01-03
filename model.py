import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


def set_parameter_requires_grad(model, fine_tuning):
    if not fine_tuning:
        for param in model.parameters():
            param.requires_grad = False


def create_resnet_model(num_classes=2, fine_tuning=True):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    set_parameter_requires_grad(model, fine_tuning)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
