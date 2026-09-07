"""Model definitions available to the experiment configs."""

from fdl_project.models.attention import CBAM, build_attention
from fdl_project.models.baseline_cnn import BaselineCNN, count_trainable_parameters
from fdl_project.models.pretrained import PretrainedClassifier
from fdl_project.models.convnext import ConvNeXtBlock, WaferConvNeXt
from fdl_project.models.densenet import DenseLayer, WaferDenseNet
from fdl_project.models.dilated import DilatedBlock, WaferDilatedCNN
from fdl_project.models.inception import InceptionBlock, WaferInception
from fdl_project.models.vit import WaferViT
from fdl_project.models.wafer_resnet import ResidualBlock, WaferResNet

__all__ = [
    "CBAM",
    "BaselineCNN",
    "ConvNeXtBlock",
    "DenseLayer",
    "DilatedBlock",
    "InceptionBlock",
    "PretrainedClassifier",
    "ResidualBlock",
    "WaferConvNeXt",
    "WaferDenseNet",
    "WaferDilatedCNN",
    "WaferInception",
    "WaferResNet",
    "WaferViT",
    "build_attention",
    "count_trainable_parameters",
]
