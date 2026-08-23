"""
ONNX Model Export Module for Stain Normalization Generators.

Serializes PyTorch CUT ResNet Generators to ONNX format with dynamic spatial axes (H, W)
for ultra-fast, framework-agnostic C++/Python deployment in digital pathology pipelines.
"""

import os
from typing import Tuple, Optional
import torch
import torch.nn as nn


def export_generator_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
    opset_version: int = 14,
) -> str:
    """
    Exports a PyTorch stain normalizer Generator to ONNX format.

    Args:
        model: Trained PyTorch Generator nn.Module.
        output_path: Target .onnx file path.
        input_shape: Input dummy tensor dimensions (B, C, H, W).
        opset_version: ONNX opset version.

    Returns:
        Absolute path to the exported ONNX model file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    model.eval()

    dummy_input = torch.randn(*input_shape, device=next(model.parameters()).device)

    # Dynamic axes allow inference on images of arbitrary resolution (H, W)
    dynamic_axes = {
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size", 2: "height", 3: "width"},
    }

    class GeneratorWrapper(nn.Module):
        def __init__(self, gen_model):
            super().__init__()
            self.gen_model = gen_model

        def forward(self, x):
            out, _ = self.gen_model(x)
            return out

    wrapper = GeneratorWrapper(model)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    return os.path.abspath(output_path)
