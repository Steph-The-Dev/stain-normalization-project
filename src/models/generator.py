"""
ResNet Generator Architecture for Image-to-Image Translation (CUT / CycleGAN).

Features Residual Identity Mapping and feature extraction hooks for PatchNCE loss.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    """Residual Block with Instance Normalization and Reflection Padding."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """
    ResNet-based Generator with Residual Identity Shortcut for Stain Normalization.
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        num_blocks: int = 6,
        use_residual_shortcut: bool = True,
    ) -> None:
        """
        Args:
            input_nc: Number of input image channels (3 for RGB).
            output_nc: Number of output image channels (3 for RGB).
            ngf: Number of generator filters in the first conv layer.
            num_blocks: Number of ResNet residual blocks.
            use_residual_shortcut: If True, adds residual input shortcut x + delta for zero-noise output.
        """
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1.")

        self.use_residual_shortcut = use_residual_shortcut

        # Layer 0: Initial Padding & Conv
        self.layer0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        )

        # Layer 1 & 2: Downsampling
        self.layer1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
        )

        # Bottleneck: Residual Blocks
        res_blocks = [ResNetBlock(ngf * 4) for _ in range(num_blocks)]
        self.bottleneck = nn.Sequential(*res_blocks)

        # Upsampling (Nearest-Neighbor + Conv2d eliminates ConvTranspose2d checkerboard grid artifacts)
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 2, ngf, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        )

        # Final Output Layer
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0, bias=True),
            nn.Tanh(),
        )

        # Zero-initialize final layer weights for zero-noise identity start
        if self.use_residual_shortcut:
            with torch.no_grad():
                self.final[1].weight.zero_()
                if self.final[1].bias is not None:
                    self.final[1].bias.zero_()

    def forward(
        self, x: torch.Tensor, layers: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with optional intermediate feature map extraction for PatchNCE loss.
        """
        feats = []

        out = self.layer0(x)
        if layers and 0 in layers:
            feats.append(out)

        out = self.layer1(out)
        if layers and 1 in layers:
            feats.append(out)

        out = self.layer2(out)
        if layers and 2 in layers:
            feats.append(out)

        out = self.bottleneck(out)
        if layers and 3 in layers:
            feats.append(out)

        out = self.upsample1(out)
        if layers and 4 in layers:
            feats.append(out)

        out = self.upsample2(out)
        if layers and 5 in layers:
            feats.append(out)

        final_delta = self.final(out)
        if self.use_residual_shortcut:
            final_out = torch.clamp(x + final_delta, -1.0, 1.0)
        else:
            final_out = final_delta

        return final_out, feats
