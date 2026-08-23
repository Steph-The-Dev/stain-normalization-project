"""
Tests for Contrastive Unpaired Translation (CUT) Deep Learning Model & Losses.
"""

import numpy as np
import pytest
import torch

from src.models import ResNetGenerator, SSIMLoss, PatchNCELoss
from src.normalizers import CUTStainNormalizer


def test_resnet_generator_forward_and_feature_hooks():
    model = ResNetGenerator(input_nc=3, output_nc=3, ngf=16, num_blocks=2)
    x = torch.randn(2, 3, 64, 64)

    # Standard forward
    out, feats = model(x)
    assert out.shape == (2, 3, 64, 64)
    assert len(feats) == 0

    # Forward with feature map extraction
    out, feats = model(x, layers=[0, 2, 4])
    assert out.shape == (2, 3, 64, 64)
    assert len(feats) == 3
    assert feats[0].shape[0] == 2  # Batch size


def test_ssim_loss_identical_tensors():
    loss_fn = SSIMLoss()
    x = torch.rand(1, 3, 64, 64)
    loss_val = loss_fn(x, x)
    assert pytest.approx(loss_val.item(), abs=1e-4) == 0.0


def test_patch_nce_loss_computation():
    nce_fn = PatchNCELoss(num_patches=16)
    f1 = [torch.randn(2, 32, 16, 16)]
    f2 = [torch.randn(2, 32, 16, 16)]

    loss_val = nce_fn(f1, f2)
    assert isinstance(loss_val, torch.Tensor)
    assert loss_val.item() > 0.0


def test_cut_stain_normalizer_not_fitted_by_default():
    normalizer = CUTStainNormalizer(ngf=16, num_blocks=2)
    assert not normalizer.is_fitted

    src = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="must be fit"):
        normalizer.transform(src)


def test_cut_stain_normalizer_fit_transform():
    np.random.seed(42)
    src = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
    target = np.random.randint(80, 220, (64, 64, 3), dtype=np.uint8)

    normalizer = CUTStainNormalizer(ngf=16, num_blocks=2, lr=1e-3)
    normalizer.fit(target, num_epochs=1, batch_size=1)

    assert normalizer.is_fitted
    result = normalizer.transform(src)
    assert result.shape == src.shape
    assert result.dtype == np.uint8
