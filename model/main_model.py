import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model


class Identical(nn.Module):
    def __init__(self):
        super(Identical, self).__init__()

    def forward(self, x):
        return x


def load_backbone(args):
    model_name = args.base_model
    bone = create_model(
        model_name,
        pretrained=True,
        num_classes=1000)

    if 'seresnet' in model_name:
        bone.avg_pool = Identical()
        bone.last_linear = Identical()
    elif 'res' in model_name:
        bone.global_pool = Identical()
        bone.fc = Identical()
    elif 'efficient' in model_name:
        bone.global_pool = Identical()
        bone.classifier = Identical()
    elif 'densenet' in model_name:
        bone.global_pool = Identical()
        bone.classifier = Identical()
    elif 'mobilenet' in model_name:
        bone.global_pool = Identical()
        bone.conv_head = Identical()
        bone.act2 = Identical()
        bone.classifier = Identical()
    return bone


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.backbone = load_backbone(args)
        self.num_features = self.backbone.num_features
        self.fc = nn.Linear(self.num_features, args.hash_len)
        self.activatetion = nn.Tanh()
        # self.drop_rate = 0.

    def forward(self, x):
        features = self.backbone(x)
        x = features
        x = F.adaptive_max_pool2d(x, 1).squeeze(-1).squeeze(-1)
        feature = x
        x = self.fc(x)
        return self.activatetion(x), feature