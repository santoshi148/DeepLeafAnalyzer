from pathlib import Path
import torch

class EarlyStopping:
    def __init__(self, patience=10): self.patience=patience; self.best=float('inf'); self.bad=0
    def step(self, val):
        if val < self.best-1e-8: self.best=val; self.bad=0; return False, True
        self.bad+=1; return self.bad>=self.patience, False


def run_epoch(model, loader, criterion, device, optimizer=None):
    train=optimizer is not None; model.train(train)
    total_loss=0.0; correct=0; n=0
    for batch in loader:
        x=batch['image'].to(device); y=batch['label'].to(device)
        if train: optimizer.zero_grad(set_to_none=True)
        logits=model(x); loss=criterion(logits,y)
        if train: loss.backward(); optimizer.step()
        total_loss += float(loss.item())*x.size(0); correct += int((logits.argmax(1)==y).sum()); n += x.size(0)
    return {'loss': total_loss/max(n,1), 'accuracy': correct/max(n,1)}


def fit(model, train_loader, val_loader, epochs, lr, weight_decay, device, outdir, scheduler_factor=0.5, scheduler_patience=5, early_patience=10):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    criterion=torch.nn.CrossEntropyLoss(); opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=scheduler_factor, patience=scheduler_patience)
    es=EarlyStopping(early_patience); hist=[]
    for epoch in range(1,epochs+1):
        tr=run_epoch(model,train_loader,criterion,device,opt); va=run_epoch(model,val_loader,criterion,device,None)
        sch.step(va['loss']); hist.append({'epoch':epoch,'train_loss':tr['loss'],'train_acc':tr['accuracy'],'val_loss':va['loss'],'val_acc':va['accuracy']})
        stop,improved=es.step(va['loss'])
        if improved: torch.save({'model':model.state_dict(),'epoch':epoch,'val_loss':va['loss']}, outdir/'best.pt')
        print(f"epoch={epoch:03d} train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} val_acc={va['accuracy']:.4f}")
        if stop: break
    return hist
