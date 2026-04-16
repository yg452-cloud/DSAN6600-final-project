from pathlib import Path
from typing import List, Dict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


def read_rgb_image(path: str) -> np.ndarray:
    """Read an RGB image and normalize to [0, 1]."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def read_mask(path: str) -> np.ndarray:
    """Read a binary mask and convert it to 0/1."""
    mask = Image.open(path).convert("L")
    mask = np.array(mask, dtype=np.float32)
    mask = (mask > 0).astype(np.float32)
    return mask


class SYSUCDDataset(Dataset):
    """
    Dataset for SYSU-CD style folder structure:
    data/
        train/
            time1/
            time2/
            label/
        val/
            time1/
            time2/
            label/
        test/
            time1/
            time2/
            label/
    """

    def __init__(self, root_dir: str, split: str):
        self.root_dir = Path(root_dir)
        self.split = split

        self.split_dir = self.root_dir / split
        self.time1_dir = self.split_dir / "time1"
        self.time2_dir = self.split_dir / "time2"
        self.label_dir = self.split_dir / "label"

        # Use filenames from time1 as the master list
        self.sample_ids = sorted([p.name for p in self.time1_dir.glob("*.png")])

        # Optional safety check
        for file_name in self.sample_ids:
            if not (self.time2_dir / file_name).exists():
                raise FileNotFoundError(f"Missing time2 file: {file_name}")
            if not (self.label_dir / file_name).exists():
                raise FileNotFoundError(f"Missing label file: {file_name}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_name = self.sample_ids[idx]

        img_t1 = read_rgb_image(self.time1_dir / file_name)
        img_t2 = read_rgb_image(self.time2_dir / file_name)
        mask = read_mask(self.label_dir / file_name)

        # Concatenate two RGB images into 6 channels
        image_6ch = np.concatenate([img_t1, img_t2], axis=-1)  # (H, W, 6)

        image_tensor = torch.tensor(image_6ch).permute(2, 0, 1).float()  # (6, H, W)
        mask_tensor = torch.tensor(mask).unsqueeze(0).float()            # (1, H, W)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "id": file_name.replace(".png", "")
        }


def create_dataloader(root_dir: str, split: str, batch_size: int = 8,
                      shuffle: bool = False, num_workers: int = 0):
    dataset = SYSUCDDataset(root_dir=root_dir, split=split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataset, loader