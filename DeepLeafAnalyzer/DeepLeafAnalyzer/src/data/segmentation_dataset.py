from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2

class SegmentationDataset(Dataset):
    def __init__(self,csv_file,root_dir='.',image_size=224):
        self.df=pd.read_csv(csv_file); self.root=Path(root_dir); self.size=image_size
        if not {'image_path','mask_path'} <= set(self.df.columns): raise ValueError('CSV must contain image_path,mask_path')
    def _r(self,p):
        p=Path(str(p)); return p if p.is_absolute() else self.root/p
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; im=np.array(Image.open(self._r(r.image_path)).convert('RGB')); m=np.array(Image.open(self._r(r.mask_path)).convert('L'))
        im=cv2.resize(im,(self.size,self.size),interpolation=cv2.INTER_LINEAR); m=cv2.resize(m,(self.size,self.size),interpolation=cv2.INTER_NEAREST)
        im=torch.from_numpy(im).permute(2,0,1).float()/255.; m=torch.from_numpy((m>127).astype(np.float32))[None]
        return {'image':im,'mask':m}
