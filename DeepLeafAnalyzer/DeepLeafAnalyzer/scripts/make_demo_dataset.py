#!/usr/bin/env python
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

def main():
    root=Path('demo_data'); (root/'images').mkdir(parents=True,exist_ok=True); rows=[]; rng=np.random.default_rng(42)
    for c in range(3):
        for i in range(30):
            a=np.zeros((96,96,3),dtype=np.uint8)+40
            # class-specific synthetic lesion pattern
            yy,xx=np.ogrid[:96,:96]; cx=28+20*c; cy=48; rad=10+2*c; mask=(xx-cx)**2+(yy-cy)**2<rad**2
            a[:,:,1]=100+rng.integers(0,30,size=(96,96)); a[mask,0]=180+20*c; a[mask,1]=40; a[mask,2]=30
            a=np.clip(a+rng.integers(0,15,size=a.shape,dtype=np.uint8),0,255).astype(np.uint8); f=root/'images'/f'c{c}_{i:03d}.png'; Image.fromarray(a).save(f); rows.append({'image_path':str(f.relative_to('.')),'class_id':c,'class_name':f'class_{c}'})
    df=pd.DataFrame(rows); tr,tmp=train_test_split(df,test_size=.2,random_state=42,stratify=df.class_id); va,te=train_test_split(tmp,test_size=.5,random_state=42,stratify=tmp.class_id); (root/'splits').mkdir(exist_ok=True); tr.to_csv(root/'splits/train.csv',index=False); va.to_csv(root/'splits/val.csv',index=False); te.to_csv(root/'splits/test.csv',index=False); print('Demo dataset created in demo_data/')
if __name__=='__main__': main()
