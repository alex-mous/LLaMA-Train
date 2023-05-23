"""
Test for evaluation metrics
"""

import unittest
import torch
from torch import nn
from src.main.util.metrics import get_number_of_parameters


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear1 = nn.Linear(784, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 10)
        self.sigmoid = nn.Sigmoid()
        self.red_herring1 = nn.Parameter(torch.zeros(10), requires_grad=False)
        self.red_herring2 = 0

    def forward(self, x):
        return self.sigmoid(self.linear2(self.relu(self.linear1(x))))


class TestParameters(unittest.TestCase):
    """
    Test parameter analysis from util/metrics
    """
    def test_get_number_of_parameters_simple(self):
        model = nn.Linear(784, 10)
        expected = 7840 + 10    # (784 * 10) weights and 10 biases
        self.assertEqual(expected, get_number_of_parameters(model))

    def test_get_number_of_parameters_custom(self):
        model = DummyModel()
        expected = 50890        # 50240 from linear1, 650 from linear2
        self.assertEqual(expected, get_number_of_parameters(model))


if __name__ == '__main__':
    unittest.main()
