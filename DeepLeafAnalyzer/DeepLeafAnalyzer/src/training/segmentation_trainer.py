from pathlib import Path
import torch
import torch.nn.functional as F

def dice_loss(logits,target,eps=1e-6):
    p=torch.sigmoid(logits); inter=(p*target).sum((1,2,3)); den=p.sum((1,2,3))+target.sum((1,2,3)); return 1-((2*inter+eps)/(den+eps)).mean()

def seg_loss(logits,target): return F.binary_cross_entropy_with_logits(logits,target)+dice_loss(logits,target)

def train_unet(model,train_loader,val_loader,device,outdir,epochs=50,lr=1e-4):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True); opt=torch.optim.Adam(model.parameters(),lr=lr); best=1e9
    for ep in range(1,epochs+1):
        model.train(); tl=0
        for b in train_loader:
            x,y=b['image'].to(device),b['mask'].to(device); opt.zero_grad(); z=model(x); l=seg_loss(z,y); l.backward(); opt.step(); tl+=l.item()*x.size(0)
        model.eval(); vl=0
        with torch.no_grad():
            for b in val_loader:
                x,y=b['image'].to(device),b['mask'].to(device); vl+=seg_loss(model(x),y).item()*x.size(0)
        vl/=max(len(val_loader.dataset),1); print(f'epoch={ep:03d} train_loss={tl/max(len(train_loader.dataset),1):.4f} val_loss={vl:.4f}')
        if vl<best: best=vl; torch.save({'model':model.state_dict(),'val_loss':vl},out/'best_unet.pt')
