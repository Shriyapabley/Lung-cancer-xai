# ml/datasets/ct_dataset.py

import os
from typing import Callable, Optional, Tuple, List

from PIL import Image
import torch
from torch.utils.data import Dataset


class CTImageDataset(Dataset):
    """
    CT lung images from a directory structure:
        root/
            normal/
                img1.png
                ...
            cancer/
                img2.png
                ...
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.root_dir = root_dir
        self.transform = transform

        if class_names is None:
            # Infer classes from subfolders
            class_names = sorted(
                [
                    d
                    for d in os.listdir(root_dir)
                    if os.path.isdir(os.path.join(root_dir, d))
                ]
            )
        self.class_names = class_names
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

        self.samples = self._make_dataset()

    def _make_dataset(self) -> List[Tuple[str, int]]:
        samples = []
        for cls_name in self.class_names:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                fpath = os.path.join(cls_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                label = self.class_to_idx[cls_name]
                samples.append((fpath, label))
        if len(samples) == 0:
            raise RuntimeError(f"No images found in {self.root_dir}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")  # CT often grayscale

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
