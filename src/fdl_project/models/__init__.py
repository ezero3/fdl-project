"""Model definitions available to the experiment configs."""

from fdl_project.models.attention import CBAM, build_attention
from fdl_project.models.baseline_cnn import BaselineCNN, count_trainable_parameters
from fdl_project.models.pretrained import PretrainedClassifier
from fdl_project.models.wafer_resnet import ResidualBlock, WaferResNet

__all__ = [
    "CBAM",
    "BaselineCNN",
    "PretrainedClassifier",
    "ResidualBlock",
    "WaferResNet",
    "build_attention",
    "count_trainable_parameters",
]
