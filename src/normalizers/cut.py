"""
CUTStainNormalizer: Strategy Pattern Wrapper for Contrastive Unpaired Translation (CUT).

Wraps PyTorch deep learning Generator, PatchNCE Loss, and SSIM Loss under the
BaseStainNormalizer interface for seamless plug-and-play integration with UI & CLI pipelines.
"""

from typing import Optional, Union, List
import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.normalizers.base import BaseStainNormalizer
from src.models.generator import ResNetGenerator
from src.models.patch_nce import PatchNCELoss
from src.models.ssim_loss import SSIMLoss
from src.dataset import UnpairedStainDataset, get_default_transform
from src.reinhard import get_tissue_mask_hsv


def rgb_to_lab_torch(rgb: torch.Tensor) -> torch.Tensor:
    """
    Converts RGB tensor in range [-1.0, 1.0] to CIELAB (L*, a*, b*) space.
    """
    rgb01 = (rgb + 1.0) / 2.0
    matrix = torch.tensor([
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227]
    ], device=rgb.device)
    
    rgb_perm = rgb01.permute(0, 2, 3, 1)
    xyz = torch.matmul(rgb_perm, matrix.T)
    
    xyz_ref = torch.tensor([0.95047, 1.00000, 1.08883], device=rgb.device)
    xyz_norm = xyz / xyz_ref
    
    mask = xyz_norm > 0.008856
    f_xyz = torch.where(mask, torch.pow(torch.clamp(xyz_norm, min=1e-5), 1.0 / 3.0), 7.787 * xyz_norm + 16.0 / 116.0)
    
    L = 116.0 * f_xyz[..., 1:2] - 16.0
    a = 500.0 * (f_xyz[..., 0:1] - f_xyz[..., 1:2])
    b = 200.0 * (f_xyz[..., 1:2] - f_xyz[..., 2:3])
    
    lab = torch.cat([L, a, b], dim=-1).permute(0, 3, 1, 2)
    return lab


class CUTStainNormalizer(BaseStainNormalizer):
    """
    Stain Normalizer using Contrastive Unpaired Translation (CUT) with SSIM Structural Loss.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        ngf: int = 16,
        num_blocks: int = 3,
        lr: float = 2e-3,
        lambda_nce: float = 1.0,
        lambda_ssim: float = 2.0,
        lambda_color: float = 10.0,
        lambda_bg: float = 10.0,
        saturation_threshold: int = 15,
    ) -> None:
        """
        Args:
            device: 'cpu' or 'cuda'. Defaults to GPU if available else CPU.
            ngf: Generator filter base count.
            num_blocks: Number of residual blocks in generator.
            lr: Learning rate for Adam optimizer.
            lambda_nce: Weight for PatchNCE contrastive loss.
            lambda_ssim: Weight for SSIM structural loss.
            lambda_color: Weight for target stain color matching loss.
            lambda_bg: Weight for background identity preservation loss.
            saturation_threshold: Threshold for tissue mask extraction during blending.
        """
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Residual shortcut preserves 100% full original resolution & sharp cell nuclei boundaries
        self.generator = ResNetGenerator(
            input_nc=3, output_nc=3, ngf=ngf, num_blocks=num_blocks, use_residual_shortcut=True
        ).to(self.device)

        self.nce_loss_fn = PatchNCELoss().to(self.device)
        self.ssim_loss_fn = SSIMLoss().to(self.device)

        self.lr = lr
        self.lambda_nce = lambda_nce
        self.lambda_ssim = lambda_ssim
        self.lambda_color = lambda_color
        self.lambda_bg = lambda_bg
        self.saturation_threshold = saturation_threshold
        self.feature_layers = [0, 2, 4]

    def fit(
        self,
        target_image_or_dataset: Union[npt.NDArray[np.uint8], UnpairedStainDataset],
        source_image: Optional[npt.NDArray[np.uint8]] = None,
        num_epochs: int = 10,
        batch_size: int = 4,
    ) -> "CUTStainNormalizer":
        """
        Fits/fine-tunes the generator to map source image stain distribution to target standard.

        Args:
            target_image_or_dataset: Either target reference image array (H, W, 3) in BGR
                                     or UnpairedStainDataset instance.
            source_image: Optional source image array (H, W, 3) in BGR to translate from.
            num_epochs: Number of training epochs.
            batch_size: DataLoader batch size.
        """
        self.generator.train()
        optimizer = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        transform = get_default_transform()

        if isinstance(target_image_or_dataset, np.ndarray):
            target_rgb = cv2.cvtColor(target_image_or_dataset, cv2.COLOR_BGR2RGB)
            target_pil = Image.fromarray(target_rgb)

            src_img = source_image if source_image is not None else target_image_or_dataset
            source_rgb = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
            source_pil = Image.fromarray(source_rgb)

            h, w = source_rgb.shape[:2]
            crop_size = max(64, min(h, w, 256))

            patch_transform = transforms.Compose([
                transforms.RandomCrop((crop_size, crop_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            full_transform = get_default_transform(normalize=True)

            dataloader = []
            for _ in range(8):
                s_t = patch_transform(source_pil).unsqueeze(0).to(self.device)
                t_t = patch_transform(target_pil).unsqueeze(0).to(self.device)
                dataloader.append((s_t, t_t))

            s_full = full_transform(source_pil).unsqueeze(0).to(self.device)
            t_full = full_transform(target_pil).unsqueeze(0).to(self.device)
            dataloader.append((s_full, t_full))
        elif isinstance(target_image_or_dataset, UnpairedStainDataset):
            dataset_loader = DataLoader(target_image_or_dataset, batch_size=batch_size, shuffle=True)
            dataloader = [(batch["A"].to(self.device), batch["B"].to(self.device)) for batch in dataset_loader]
        else:
            raise TypeError("target_image_or_dataset must be a numpy image array or UnpairedStainDataset.")

        total_steps = num_epochs * len(dataloader)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=self.lr * 0.01)

        for epoch in range(num_epochs):
            for src_batch, tgt_batch in dataloader:
                optimizer.zero_grad()

                gen_out, gen_feats = self.generator(src_batch, layers=self.feature_layers)

                with torch.no_grad():
                    _, src_feats = self.generator(src_batch, layers=self.feature_layers)

                loss_nce = self.nce_loss_fn(src_feats, gen_feats)
                loss_ssim = self.ssim_loss_fn(src_batch, gen_out)

                # Continuous smooth tissue masking to prevent sharp edge boundary halos
                src_mean_channel = torch.mean(src_batch, dim=1, keepdim=True)
                tgt_mean_channel = torch.mean(tgt_batch, dim=1, keepdim=True)

                # Smooth sigmoid tissue probability mask (0.0 for white background, 1.0 for tissue)
                src_mask = torch.sigmoid(12.0 * (0.6 - src_mean_channel))
                tgt_mask = torch.sigmoid(12.0 * (0.6 - tgt_mean_channel))
                bg_mask = 1.0 - src_mask

                gen_lab = rgb_to_lab_torch(gen_out)
                tgt_lab = rgb_to_lab_torch(tgt_batch)

                g_tissue_lab = gen_lab * src_mask
                t_tissue_lab = tgt_lab * tgt_mask

                src_mask_sum = src_mask.sum(dim=[2, 3]) + 1e-5
                tgt_mask_sum = tgt_mask.sum(dim=[2, 3]) + 1e-5

                g_m = g_tissue_lab.sum(dim=[2, 3]) / src_mask_sum
                t_m = t_tissue_lab.sum(dim=[2, 3]) / tgt_mask_sum

                g_var = ((g_tissue_lab - g_m.unsqueeze(-1).unsqueeze(-1)) ** 2 * src_mask).sum(dim=[2, 3]) / src_mask_sum
                t_var = ((t_tissue_lab - t_m.unsqueeze(-1).unsqueeze(-1)) ** 2 * tgt_mask).sum(dim=[2, 3]) / tgt_mask_sum

                # Lightness L* (channel 0) weighted gently to preserve dark cell nuclei details; chromaticity a*, b* (channels 1, 2) matched
                loss_color = (
                    0.1 * torch.mean(torch.abs(g_m[:, 0] - t_m[:, 0]))
                    + 1.0 * torch.mean(torch.abs(g_m[:, 1:] - t_m[:, 1:]))
                    + 0.1 * torch.mean(torch.abs(torch.sqrt(g_var[:, 0] + 1e-5) - torch.sqrt(t_var[:, 0] + 1e-5)))
                    + 1.0 * torch.mean(torch.abs(torch.sqrt(g_var[:, 1:] + 1e-5) - torch.sqrt(t_var[:, 1:] + 1e-5)))
                )
                loss_bg = torch.mean(torch.abs((gen_out - src_batch) * bg_mask))

                total_loss = (
                    self.lambda_color * loss_color 
                    + self.lambda_nce * loss_nce 
                    + self.lambda_ssim * loss_ssim
                    + self.lambda_bg * loss_bg
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

        self.generator.eval()
        self.is_fitted = True
        return self

    def fit_transform(
        self, 
        source_image: npt.NDArray[np.uint8], 
        target_image: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """
        Convenience method: fits generator on (source, target) pair and transforms source image.
        """
        return self.fit(target_image, source_image=source_image, num_epochs=10).transform(source_image)

    def transform(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Normalizes a source BGR image using the trained CUT generator.
        """
        if not self.is_fitted:
            raise RuntimeError("CUTStainNormalizer must be fit before calling transform().")

        original_h, original_w = image.shape[:2]

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = get_default_transform(normalize=True)
        pil_img = Image.fromarray(img_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(self.device)

        self.generator.eval()
        with torch.no_grad():
            output_tensor, _ = self.generator(input_tensor)

        output_tensor = (output_tensor.squeeze(0).cpu() + 1.0) / 2.0
        output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

        output_np = output_tensor.permute(1, 2, 0).numpy()
        output_rgb = (output_np * 255.0).astype(np.uint8)

        if output_rgb.shape[:2] != (original_h, original_w):
            output_rgb = cv2.resize(output_rgb, (original_w, original_h), interpolation=cv2.INTER_CUBIC)

        output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

        # Continuous soft tissue-background alpha matte with wide Gaussian feathering (15x15)
        # Prevents hard step transition lines between normalized tissue and background slide
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Tissue probability combining saturation and brightness distance from white glass slide (255)
        sat_prob = 1.0 / (1.0 + np.exp(-0.2 * (sat - float(self.saturation_threshold))))
        luma_prob = 1.0 / (1.0 + np.exp(-0.1 * (220.0 - gray)))
        tissue_prob = np.maximum(sat_prob, luma_prob)

        # Smooth wide Gaussian feathering kernel (15x15) for imperceptible gradient transitions
        mask_float = cv2.GaussianBlur(tissue_prob, (15, 15), 0)[:, :, None]
        blended_bgr = (output_bgr.astype(np.float32) * mask_float + image.astype(np.float32) * (1.0 - mask_float)).astype(np.uint8)

        return blended_bgr
