#!/usr/bin/env python
import argparse, yaml, sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.dataset import LeafClassificationDataset
from src.data.preprocessing import build_transforms
from src.models.hybrid_leaf_disease_net import HybridLeafDiseaseNet
from src.training.trainer import fit
from src.utils.seed import set_seed


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train',required=True); ap.add_argument('--val',required=True); ap.add_argument('--root',default='.'); ap.add_argument('--config',default='configs/default.yaml'); ap.add_argument('--outdir',default='outputs/classifier'); ap.add_argument('--epochs',type=int); args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config)); set_seed(cfg['seed'])
    ncls=int(cfg.get('num_classes', pd.read_csv(args.train)['class_id'].nunique()))
    tr=LeafClassificationDataset(args.train,args.root,build_transforms(cfg['image_size'],True)); va=LeafClassificationDataset(args.val,args.root,build_transforms(cfg['image_size'],False))
    dltr=DataLoader(tr,batch_size=cfg['training']['batch_size'],shuffle=True,num_workers=0); dlva=DataLoader(va,batch_size=cfg['training']['batch_size'],shuffle=False,num_workers=0)
    mcfg=cfg['model']; model=HybridLeafDiseaseNet(ncls, channels=tuple(mcfg['backbone_channels']), fpn_channels=mcfg['fpn_channels'], heads=mcfg['attention_heads'], lstm_hidden=mcfg['lstm_hidden'], dropout=mcfg['dropout'])
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(dev)
    fit(model,dltr,dlva,args.epochs or cfg['training']['epochs'],cfg['training']['learning_rate'],cfg['training']['weight_decay'],dev,args.outdir,cfg['training']['scheduler_factor'],cfg['training']['scheduler_patience'],cfg['training']['early_stopping_patience'])

if __name__=='__main__': main()
