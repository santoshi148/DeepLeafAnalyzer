#!/usr/bin/env python
import argparse,sys,torch
from pathlib import Path
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.segmentation_dataset import SegmentationDataset
from src.models.unet import UNet
from src.training.segmentation_trainer import train_unet
from src.utils.seed import set_seed

def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',required=True); p.add_argument('--val',required=True); p.add_argument('--root',default='.'); p.add_argument('--outdir',default='outputs/unet'); p.add_argument('--epochs',type=int,default=50); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--image-size',type=int,default=224); a=p.parse_args(); set_seed(42)
    tr=SegmentationDataset(a.train,a.root,a.image_size); va=SegmentationDataset(a.val,a.root,a.image_size); dl1=DataLoader(tr,a.batch_size,shuffle=True); dl2=DataLoader(va,a.batch_size)
    d=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); m=UNet(base=32).to(d); train_unet(m,dl1,dl2,d,a.outdir,a.epochs,1e-4)
if __name__=='__main__': main()
