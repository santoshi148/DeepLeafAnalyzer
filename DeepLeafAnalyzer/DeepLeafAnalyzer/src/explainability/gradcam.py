import torch
import torch.nn.functional as F
import numpy as np

class SimpleGradCAM:
    def __init__(self, model, target_layer):
        self.model=model; self.activations=None; self.gradients=None
        target_layer.register_forward_hook(self._fh)
        target_layer.register_full_backward_hook(self._bh)
    def _fh(self,m,i,o): self.activations=o
    def _bh(self,m,gi,go): self.gradients=go[0]
    def __call__(self,x,class_idx=None):
        self.model.zero_grad(set_to_none=True); logits=self.model(x)
        if class_idx is None: class_idx=logits.argmax(1)
        score=logits[torch.arange(x.size(0),device=x.device),class_idx].sum(); score.backward()
        w=self.gradients.mean(dim=(2,3),keepdim=True)
        cam=(w*self.activations).sum(1,keepdim=True); cam=F.relu(cam); cam=F.interpolate(cam,size=x.shape[-2:],mode='bilinear',align_corners=False)
        cam=cam.squeeze(1)
        cam=(cam-cam.amin((1,2),keepdim=True))/(cam.amax((1,2),keepdim=True)-cam.amin((1,2),keepdim=True)+1e-8)
        return cam.detach().cpu().numpy(), logits.detach()
