import torch.nn as nn
from torchvision import models

def get_root_model(num_classes=2):
    """Menginisialisasi Vision Transformer (ViT_B_16) untuk Level 1 Root."""
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    num_features = model.heads.head.in_features
    model.heads.head = nn.Linear(num_features, num_classes)
    return model

def get_sub_organic_model(num_classes=2):
    """Menginisialisasi MobileNetV3-Large untuk Sub-Model Organik."""
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, num_classes)
    return model

def get_sub_inorganic_model(num_classes=7):
    """Menginisialisasi ConvNeXt-Tiny untuk Sub-Model Anorganik."""
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_features, num_classes)
    return model