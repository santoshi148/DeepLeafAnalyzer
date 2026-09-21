from pathlib import Path
from typing import Optional
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class LeafClassificationDataset(Dataset):
    """CSV-backed image dataset.

    Required CSV columns: image_path, class_id. Optional: mask_path, bbox_path.
    Paths may be absolute or relative to root_dir.
    """
    def __init__(self, csv_file, root_dir='.', transform=None):
        self.df = pd.read_csv(csv_file)
        required = {'image_path', 'class_id'}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f'Missing required columns: {sorted(missing)}')
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _resolve(self, p):
        p = Path(str(p))
        return p if p.is_absolute() else self.root_dir / p

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self._resolve(row.image_path)
        image = np.array(Image.open(path).convert('RGB'))
        if self.transform:
            out = self.transform(image=image)
            image = out['image']
        else:
            image = torch.from_numpy(image).permute(2,0,1).float()/255.0
        return {
            'image': image,
            'label': torch.tensor(int(row.class_id), dtype=torch.long),
            'path': str(path)
        }
