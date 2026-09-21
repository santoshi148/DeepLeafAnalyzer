import random
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps, ImageFilter


class LeafTransform:
    def __init__(self, image_size=224, train=False, use_clahe=False):
        self.image_size=image_size; self.train=train; self.use_clahe=use_clahe
    def __call__(self, image):
        if isinstance(image, np.ndarray): im=Image.fromarray(image.astype(np.uint8))
        elif isinstance(image, Image.Image): im=image.convert('RGB')
        else: raise TypeError('image must be numpy array or PIL image')
        im=im.resize((self.image_size,self.image_size), Image.BILINEAR)
        if self.train:
            if random.random()<0.5: im=ImageOps.mirror(im)
            if random.random()<0.2: im=ImageOps.flip(im)
            if random.random()<0.5: im=im.rotate(random.uniform(-15,15), resample=Image.BILINEAR)
            if random.random()<0.4: im=ImageEnhance.Brightness(im).enhance(random.uniform(0.8,1.2))
            if random.random()<0.4: im=ImageEnhance.Contrast(im).enhance(random.uniform(0.8,1.2))
            if random.random()<0.15:
                arr=np.array(im).astype(np.float32)+np.random.normal(0,5,np.array(im).shape)
                im=Image.fromarray(np.clip(arr,0,255).astype(np.uint8))
        arr=np.array(im,dtype=np.float32)/255.0
        ten=torch.from_numpy(arr).permute(2,0,1).contiguous()
        return {'image':ten}


def build_transforms(image_size=224, train=False, use_clahe=False):
    return LeafTransform(image_size,train,use_clahe)
