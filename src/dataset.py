import os
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in IMG_EXTENSIONS


class FruitQualityDataset(Dataset):
    """Dataset that reads images recursively from a base directory organized as:

    base_dir/
      <quality_name>/
        <fruit_name>/
          image files (possibly nested deeper)

    The fruit label is inferred from the folder name immediately under the quality folder.
    The quality label is the top-level folder under `base_dir`.
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform

        self.samples: List[Tuple[str, str, str]] = []  # (filepath, quality_name, fruit_name)

        for dirpath, _, filenames in os.walk(self.root_dir):
            for fname in filenames:
                if not is_image_file(fname):
                    continue
                fpath = os.path.join(dirpath, fname)
                rel = os.path.relpath(fpath, self.root_dir)
                parts = rel.split(os.sep)
                # Need at least quality/fruit/file
                if len(parts) < 3:
                    # skip files not in expected depth
                    continue
                quality_name = parts[0]
                fruit_name = parts[1]
                self.samples.append((fpath, quality_name, fruit_name))

        if not self.samples:
            raise RuntimeError(f"No image files found under {self.root_dir}. Expected structure: <quality>/<fruit>/images.")

        # build mappings
        qualities = sorted({q for _, q, _ in self.samples})
        fruits = sorted({f for _, _, f in self.samples})

        self.quality2idx: Dict[str, int] = {q: i for i, q in enumerate(qualities)}
        self.fruit2idx: Dict[str, int] = {f: i for i, f in enumerate(fruits)}

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        # Data augmentation for better accuracy
        import torchvision.transforms as T
        import random
        
        # Augmentation transforms
        if self.transform is None:  # Training mode - apply augmentation
            transform = T.Compose([
                T.Resize((256, 256)),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(0.5),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:  # Inference mode - no augmentation
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        return transform(img)

    def __getitem__(self, idx: int):
        path, quality_name, fruit_name = self.samples[idx]
        img = self._load_image(path)
        tensor = self._preprocess(img)

        fruit_label = self.fruit2idx[fruit_name]
        quality_label = self.quality2idx[quality_name]

        return tensor, torch.tensor(fruit_label, dtype=torch.long), torch.tensor(quality_label, dtype=torch.long)


if __name__ == "__main__":
    # quick test (update path as needed)
    import sys
    if len(sys.argv) > 1:
        ds = FruitQualityDataset(sys.argv[1])
        print('Found', len(ds), 'samples')
        print('Fruits:', ds.fruit2idx)
        print('Qualities:', ds.quality2idx)
