import numpy as np
import random
import socket
import torch

hostname = socket.gethostname()
if 'ubuntu' == hostname: # ubuntu
    dataset_path = '/home/nagejacob/Documents/datasets'
else: # hpc
    dataset_path = '/mnt/ssd0/anaconda3/lijunyi/datasets'

# c, h, w numpy
def _aug_np3(img, flip_h, flip_w, transpose):
    if flip_h:
        img = img[:, ::-1, :]
    if flip_w:
        img = img[:, :, ::-1]
    if transpose:
        img = np.transpose(img, (0, 2, 1))
    return img

def _aug_torch3(img, flip_h, flip_w, transpose):
    if flip_h:
        img = torch.flip(img, dims=[1])
    if flip_w:
        img = torch.flip(img, dims=[2])
    if transpose:
        img = img.permute(0, 2, 1)
    return img

def _aug_3(img, flip_h, flip_w, transpose):
    if type(img) is np.ndarray:
        img = _aug_np3(img, flip_h, flip_w, transpose)
    elif type(img) is torch.Tensor:
        img = _aug_torch3(img, flip_h, flip_w, transpose)
    else:
        raise TypeError('img is neither np.ndarray nor torch.Tensor !')
    return img

def aug_3(img_L, img_H=None):
    flip_h = random.random() > 0.5
    flip_w = random.random() > 0.5
    transpose = random.random() > 0.5

    img_L = _aug_3(img_L, flip_h, flip_w, transpose)
    if img_H is not None:
        img_H = _aug_3(img_H, flip_h, flip_w, transpose)
        return img_L, img_H
    else:
        return img_L

def crop_3(patch_size, img_L, img_H=None):
    C, H, W = img_L.shape
    position_H = random.randint(0, H - patch_size)
    position_W = random.randint(0, W - patch_size)
    patch_L = img_L[:, position_H:position_H+patch_size, position_W:position_W+patch_size]
    if img_H is not None:
        patch_H = img_H[:, position_H:position_H+patch_size, position_W:position_W+patch_size]
        return patch_L, patch_H
    else:
        return patch_L