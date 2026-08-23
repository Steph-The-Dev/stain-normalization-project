"""
Tests for ONNX Model Export module.
"""

import os
import pytest
import torch

from src.models import ResNetGenerator
from src.export import export_generator_to_onnx


def test_export_generator_to_onnx(tmp_path):
    model = ResNetGenerator(input_nc=3, output_nc=3, ngf=16, num_blocks=2)
    onnx_file = str(tmp_path / "cut_generator.onnx")

    exported_path = export_generator_to_onnx(model, output_path=onnx_file, input_shape=(1, 3, 64, 64))

    assert os.path.exists(exported_path)
    assert os.path.getsize(exported_path) > 0
