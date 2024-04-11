import datetime
import imageio
import numpy as np
import os
from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS) and (not filename.startswith('.'))

def get_image_paths(dataroot):
    paths = None  # return None if dataroot is None
    if isinstance(dataroot, str):
        paths = sorted(_get_paths_from_images(dataroot))
    elif isinstance(dataroot, list):
        paths = []
        for i in dataroot:
            paths += sorted(_get_paths_from_images(i))
    return paths

def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images

def open_image_uint8(image_file, to_gray=False):
    if to_gray:
        image = np.asarray(Image.open(image_file).convert('L'))
    else:
        image = imageio.imread(image_file).astype(np.uint8)

    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=0)
        image = np.repeat(image, 3, axis=0)
    elif len(image.shape) == 3:
        image = np.transpose(image, (2, 0, 1))
        image = image[:3, :, :]

    return image


def date_time():
    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d, %H:%M:%S")
    return date_time


def log(log_file, str, also_print=True, with_time=True):
    with open(log_file, 'a+') as F:
        if with_time:
            F.write(date_time() + '  ')
        F.write(str)
    if also_print:
        if with_time:
            print(date_time(), end='  ')
        print(str, end='')


# save numpy image in shape 3xHxW
def np2image(image, image_file):
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0., 1.)
    image = image * 255.
    image = image.astype(np.uint8)
    if 1 == image.shape[2]:
        image = image[:, :, 0]
    imageio.imwrite(image_file, image)

# save tensor image in shape 1x3xHxW
def tensor2image(image, image_file):
    image = image.detach().cpu().squeeze(0).numpy()
    np2image(image, image_file)