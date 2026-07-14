from __future__ import annotations

import os

import cv2
import numpy as np


class AnyLocEmbedding:
    """AnyLoc-style place descriptor from DINOv2 patch tokens.

    This is intentionally a simple x64 evaluation backend. It uses a pretrained
    DINOv2 model to extract patch tokens, then aggregates local patch features
    with mean+max pooling into one global place descriptor.

    It does not implement VLAD yet. The goal is to replace TinyNav's current
    CLS-token DINO retrieval with a patch-token place-recognition descriptor and
    evaluate whether the direction improves map/query retrieval before adding a
    map-specific VLAD vocabulary.
    """

    def __init__(
        self,
        model_name: str | None = None,
        image_size: int | None = None,
        device: str | None = None,
    ):
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "AnyLocEmbedding requires torch. "
                "Install them in the x64 test environment before using this branch."
            ) from exc

        self.torch = torch
        self.model_name = model_name or os.environ.get("TINYNAV_ANYLOC_MODEL", "dinov2_vitb14")
        self.image_size = int(image_size or os.environ.get("TINYNAV_ANYLOC_IMAGE_SIZE", "224"))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        hub_repo = os.environ.get("TINYNAV_ANYLOC_HUB_REPO", "facebookresearch/dinov2")
        self.model = torch.hub.load(hub_repo, self.model_name).to(self.device)
        self.model.eval()

    def _to_rgb(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("image is None")
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 1:
            rgb = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] >= 3:
            # TinyNav images are generally OpenCV BGR when 3-channel.
            rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"unsupported image shape: {image.shape}")
        if self.image_size > 0:
            rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        return rgb

    async def infer(self, image: np.ndarray) -> np.ndarray:
        tokens = await self.infer_patch_tokens(image)
        mean_desc = tokens.mean(axis=0, keepdims=True)
        max_desc = tokens.max(axis=0, keepdims=True)
        desc = np.concatenate([mean_desc, max_desc], axis=-1)
        norm = np.linalg.norm(desc, axis=-1, keepdims=True)
        desc = desc / np.maximum(norm, 1e-8)
        return desc.squeeze(0).astype(np.float32)

    async def infer_patch_tokens(self, image: np.ndarray) -> np.ndarray:
        rgb = self._to_rgb(image)
        tensor = self.torch.from_numpy(rgb).to(self.device)
        tensor = tensor.permute(2, 0, 1).float().div(255.0)
        mean = self.torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        std = self.torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0)

        with self.torch.inference_mode():
            features = self.model.forward_features(tensor)
            tokens = features["x_norm_patchtokens"]  # patch tokens, [1, N, C]
            tokens = self.torch.nn.functional.normalize(tokens, dim=-1)

        return tokens.squeeze(0).detach().cpu().numpy().astype(np.float32)
