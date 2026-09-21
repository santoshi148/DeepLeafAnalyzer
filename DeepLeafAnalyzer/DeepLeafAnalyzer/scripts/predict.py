#!/usr/bin/env python
import argparse,sys,yaml,torch,numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models.hybrid_leaf_disease_net import HybridLeafDiseaseNet
from src.data.preprocessing import build_transforms

def main():
    p=argparse.ArgumentParser(); p.add_argument('--image',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--config',default='configs/default.yaml'); a=p.parse_args(); cfg=yaml.safe_load(open(a.config)); m=cfg['model']
    model=HybridLeafDiseaseNet(cfg['num_classes'],channels=tuple(m['backbone_channels']),fpn_channels=m['fpn_channels'],heads=m['attention_heads'],lstm_hidden=m['lstm_hidden'],dropout=m['dropout']); d=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ck=torch.load(a.checkpoint,map_location=d); model.load_state_dict(ck['model'] if 'model' in ck else ck); model.to(d).eval()
    im=np.array(Image.open(a.image).convert('RGB')); x=build_transforms(cfg['image_size'],False)(image=im)['image'][None].to(d)
    with torch.no_grad(): prob=torch.softmax(model(x),1)[0]; idx=int(prob.argmax()); conf=float(prob[idx])
    names=cfg.get('class_names',[str(i) for i in range(cfg['num_classes'])]); label=names[idx] if idx<len(names) else str(idx); level='High' if conf>=0.8 else ('Moderate' if conf>=0.5 else 'Low')
    print({'predicted_class':label,'class_id':idx,'confidence':round(conf,6),'confidence_category':level})
if __name__=='__main__': main()
