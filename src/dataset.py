"""
PyTorch Unpaired Dataset Loader for Histological Stain Normalization (CycleGAN / CUT).

Loads unpaired images/patches from Domain A (Source) and Domain B (Target),
applying GPU/CPU tensor transformations for neural network training.
"""

import os
import random
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff")


def is_image_file(filename: str) -> bool:
    """Checks if a file path is a valid image based on extension."""
    return filename.lower().endswith(IMG_EXTENSIONS)


def get_image_paths(dir_or_paths: Union[str, List[str]]) -> List[str]:
    """
    Returns a sorted list of absolute image file paths from a directory or list.
    """
    if isinstance(dir_or_paths, list):
        return sorted([os.path.abspath(p) for p in dir_or_paths if is_image_file(p)])

    if not os.path.exists(dir_or_paths):
        raise FileNotFoundError(f"Image directory path does not exist: '{dir_or_paths}'")

    if os.path.isfile(dir_or_paths):
        return [os.path.abspath(dir_or_paths)] if is_image_file(dir_or_paths) else []

    paths = []
    for root, _, fnames in os.walk(dir_or_paths):
        for fname in sorted(fnames):
            if is_image_file(fname):
                paths.append(os.path.join(root, fname))

    return sorted(paths)


def get_default_transform(
    image_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
) -> transforms.Compose:
    """
    Generates standard PyTorch transformations for GAN inputs.

    Normalizes images to range [-1.0, 1.0] required for Tanh activation generators.
    """
    transform_list = []
    if image_size is not None:
        transform_list.append(transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC))
    transform_list.append(transforms.ToTensor())
    if normalize:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return transforms.Compose(transform_list)


class UnpairedStainDataset(Dataset):
    """
    Unpaired PyTorch Dataset for Domain A (Source) and Domain B (Target) stain translation.
    """

    def __init__(
        self,
        domain_a: Union[str, List[str]],
        domain_b: Union[str, List[str]],
        transform: Optional[transforms.Compose] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            domain_a: Directory path or list of image paths for Domain A (Source).
            domain_b: Directory path or list of image paths for Domain B (Target).
            transform: Optional PyTorch torchvision transform. Defaults to get_default_transform().
            random_seed: Optional seed for reproducible domain B sampling.
        """
        self.paths_a = get_image_paths(domain_a)
        self.paths_b = get_image_paths(domain_b)

        if len(self.paths_a) == 0:
            raise ValueError(f"No valid images found for Domain A in '{domain_a}'.")
        if len(self.paths_b) == 0:
            raise ValueError(f"No valid images found for Domain B in '{domain_b}'.")

        self.size_a = len(self.paths_a)
        self.size_b = len(self.paths_b)
        self.transform = transform if transform is not None else get_default_transform()
        
        if random_seed is not None:
            self.rng = random.Random(random_seed)
        else:
            self.rng = random.Random()

    def __len__(self) -> int:
        """Returns the maximum dataset size across both domains."""
        return max(self.size_a, self.size_b)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Returns an unpaired sample from Domain A and Domain B.

        Returns:
            Dict containing:
                - 'A': Tensor image from Domain A (3, H, W)
                - 'B': Tensor image from Domain B (3, H, W)
                - 'path_A': File path of image A
                - 'path_B': File path of image B
        """
        path_a = self.paths_a[index % self.size_a]

        # Random sampling for unpaired domain B to break pairing correlation
        index_b = self.rng.randint(0, self.size_b - 1)
        path_b = self.paths_b[index_b]

        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")

        tensor_a = self.transform(img_a)
        tensor_b = self.transform(img_b)

        return {
            "A": tensor_a,
            "B": tensor_b,
            "path_A": path_a,
            "path_B": path_b,
        }
