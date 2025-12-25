import torch

from src.models.custom_cnn import CustomCNN
from src.models.utils import count_parameters


def test_count_parameters_returns_positive():
    model = CustomCNN(in_channels=3, num_classes=4)
    params = count_parameters(model)
    assert params > 0


