"""
Differentiable Structural Similarity (SSIM) Loss Module for PyTorch.

Used as an auxiliary loss in Contrastive Unpaired Translation (CUT) to prevent
generative hallucinations and enforce histological structural preservation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_gaussian_window(window_size: int, channel: int) -> torch.Tensor:
    """Creates a 1D Gaussian kernel and expands it to 2D for 4D Tensor convolution."""
    def _gaussian(size, sigma):
        gauss = torch.tensor([math.exp(-(x - size // 2) ** 2 / (float(2 * sigma ** 2))) for x in range(size)])
        return gauss / gauss.sum()

    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim_tensor(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channel: int,
    size_average: bool = True,
) -> torch.Tensor:
    """Computes SSIM map between two 4D Tensors (B, C, H, W)."""
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(nn.Module):
    """
    Differentiable SSIM Loss Module: Loss = 1.0 - SSIM(img1, img2)
    """

    def __init__(self, window_size: int = 11, size_average: bool = True) -> None:
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.register_buffer("window", create_gaussian_window(window_size, self.channel))

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img1: Tensor of shape (B, C, H, W) normalized to [0, 1] or [-1, 1]
            img2: Tensor of shape (B, C, H, W)
        """
        # Ensure inputs are scaled to [0, 1] for stable SSIM constant calculation
        if img1.min() < 0:
            img1 = (img1 + 1.0) / 2.0
            img2 = (img2 + 1.0) / 2.0

        _, channel, _, _ = img1.size()

        if channel != self.channel or self.window.device != img1.device:
            window = create_gaussian_window(self.window_size, channel).to(img1.device)
            self.window = window
            self.channel = channel
        else:
            window = self.window

        ssim_val = ssim_tensor(img1, img2, window, self.window_size, channel, self.size_average)
        return 1.0 - ssim_val
