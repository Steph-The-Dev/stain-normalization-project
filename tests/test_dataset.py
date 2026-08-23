"""
Tests for PyTorch Unpaired Dataset Loader module.
"""

import os
from PIL import Image
import numpy as np
import pytest
import torch

from src.dataset import (
    is_image_file,
    get_image_paths,
    get_default_transform,
    UnpairedStainDataset,
)


def test_is_image_file():
    assert is_image_file("slide.png")
    assert is_image_file("slide.JPG")
    assert is_image_file("slide.tif")
    assert not is_image_file("notes.txt")
    assert not is_image_file("script.py")


def test_get_image_paths(tmp_path):
    img_dir = tmp_path / "domain_a"
    img_dir.mkdir()

    p1 = img_dir / "img1.png"
    p2 = img_dir / "img2.jpg"
    txt = img_dir / "ignore.txt"

    Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8)).save(p1)
    Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8)).save(p2)
    txt.write_text("hello")

    paths = get_image_paths(str(img_dir))
    assert len(paths) == 2
    assert str(p1) in paths
    assert str(p2) in paths


def test_unpaired_dataset_loading_and_shapes(tmp_path):
    dir_a = tmp_path / "domain_a"
    dir_b = tmp_path / "domain_b"
    dir_a.mkdir()
    dir_b.mkdir()

    # Domain A: 3 images
    for i in range(3):
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(arr).save(dir_a / f"a_{i}.png")

    # Domain B: 2 images
    for i in range(2):
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(arr).save(dir_b / f"b_{i}.png")

    transform = get_default_transform(image_size=(64, 64), normalize=True)
    dataset = UnpairedStainDataset(str(dir_a), str(dir_b), transform=transform, random_seed=42)

    # Length should be max(3, 2) = 3
    assert len(dataset) == 3

    sample = dataset[0]
    assert "A" in sample and "B" in sample
    assert "path_A" in sample and "path_B" in sample

    assert isinstance(sample["A"], torch.Tensor)
    assert isinstance(sample["B"], torch.Tensor)

    assert sample["A"].shape == (3, 64, 64)
    assert sample["B"].shape == (3, 64, 64)

    # Tensor values normalized with (mean=0.5, std=0.5) should be bounded in [-1.0, 1.0]
    assert torch.all(sample["A"] >= -1.0) and torch.all(sample["A"] <= 1.0)
    assert torch.all(sample["B"] >= -1.0) and torch.all(sample["B"] <= 1.0)


def test_unpaired_dataset_empty_domain_raises():
    empty_dir = "/tmp/non_existent_domain_test_12345"
    with pytest.raises(FileNotFoundError):
        UnpairedStainDataset(empty_dir, empty_dir)
