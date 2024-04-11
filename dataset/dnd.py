from dataset.base_function import dataset_path, crop_3
import h5py
import glob
import numpy as np
import os
import scipy.io as sio
from torch.utils.data import Dataset

dnd_path = os.path.join(dataset_path, 'DND')

class DNDBenchmarkTrainDataset(Dataset):
    def __init__(self, patch_size, pin_memory=True):
        super().__init__()
        self.patch_size = patch_size
        self.pin_memory = pin_memory

        self._img_paths = self._get_img_paths()
        if self.pin_memory:
            self._imgs = self._open_images()

    def __getitem__(self, index):
        index = index % len(self._img_paths)

        if self.pin_memory:
            img_L = self.imgs[index]['L']
        else:
            img_path = self._img_paths[index]
            img_L = self._open_image(img_path['L'])

        patch_L = crop_3(self.patch_size, img_L)

        return {'L': patch_L.copy()}

    def __len__(self):
        return len(self._img_paths) * 100

    def _get_img_paths(self):
        L_pattern = os.path.join(dnd_path, 'images_srgb/00*.mat')
        L_paths = sorted(glob.glob(L_pattern))

        img_paths = []
        for L_path in L_paths:
            img_paths.append({'L':L_path})
        return img_paths

    def _open_images(self):
        self.imgs = []
        for img_path in self._img_paths:
            img_L = self._open_image(img_path['L'])
            self.imgs.append({'L': img_L})

    def _open_image(self, path):
        img = h5py.File(path)
        img = np.float32(np.array(img['InoisySRGB']).T)
        img = np.transpose(img, (2, 0, 1))
        return img

# use PNGAN results as pseudo GT
class DNDBenchmarkPNGANDataset(Dataset):
    def __init__(self, length=None):
        super(DNDBenchmarkPNGANDataset, self).__init__()
        self.length = length
        self.imgs = []
        infos = h5py.File(os.path.join(dnd_path, 'info.mat'), 'r')
        info = infos['info']
        bb = info['boundingboxes']
        for i in range(50):
            filename = os.path.join(dnd_path, 'images_srgb', '%04d.mat' % (i + 1))
            img = h5py.File(filename, 'r')
            Inoisy = np.float32(np.array(img['InoisySRGB']).T)
            ref = bb[0][i]
            boxes = np.array(info[ref]).T
            for k in range(20):
                idx = [int(boxes[k, 0] - 1), int(boxes[k, 2]), int(boxes[k, 1] - 1), int(boxes[k, 3])]
                Inoisy_crop = Inoisy[idx[0]:idx[1], idx[2]:idx[3], :].copy()
                Inoisy_crop = np.transpose(Inoisy_crop, (2, 0, 1))
                Iclean_crop = sio.loadmat(os.path.join(dnd_path, 'PNGAN/dnd_srgb_mat', '%04d_%02d.mat' % (i + 1, k + 1)))['Idenoised_crop']
                Iclean_crop = np.transpose(Iclean_crop[0], (2, 0, 1))

                self.imgs.append({'L':Inoisy_crop, 'H': Iclean_crop})

    def __getitem__(self, index):
        return self.imgs[index]

    def __len__(self):
        if self.length is not None:
            return self.length
        return 1000