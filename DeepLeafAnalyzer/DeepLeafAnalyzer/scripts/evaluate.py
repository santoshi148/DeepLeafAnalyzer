#!/usr/bin/env python
import argparse,yaml,sys
from pathlib import Path
import pandas as pd, numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.dataset import LeafClassificationDataset
from src.data.preprocessing import build_transforms
from src.models.hybrid_leaf_disease_net import HybridLeafDiseaseNet
from src.evaluation.classification_metrics import classification_metrics


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--test',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--root',default='.'); ap.add_argument('--config',default='configs/default.yaml'); args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config)); ncls=int(cfg['num_classes']); ds=LeafClassificationDataset(args.test,args.root,build_transforms(cfg['image_size'],False)); dl=DataLoader(ds,batch_size=cfg['training']['batch_size'],shuffle=False,num_workers=0)
    m=cfg['model']; model=HybridLeafDiseaseNet(ncls,channels=tuple(m['backbone_channels']),fpn_channels=m['fpn_channels'],heads=m['attention_heads'],lstm_hidden=m['lstm_hidden'],dropout=m['dropout'])
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ck=torch.load(args.checkpoint,map_location=dev); model.load_state_dict(ck['model'] if 'model' in ck else ck); model.to(dev).eval()
    yt=[]; yp=[]
    with torch.no_grad():
        for b in dl:
            p=model(b['image'].to(dev)).argmax(1).cpu().numpy(); yp.extend(p); yt.extend(b['label'].numpy())
    print(classification_metrics(yt,yp))

if __name__=='__main__': main()
