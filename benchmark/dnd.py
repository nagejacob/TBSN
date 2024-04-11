import sys
sys.path.append('..')
import argparse
from benchmark.ensemble_wrapper import EnsembleWrapper
from dataset.dnd import DNDBenchmarkPNGANDataset
import numpy as np
import os
import scipy.io as sio
import shutil
from skimage.metrics import peak_signal_noise_ratio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from util.option import parse, recursive_print

def bundle_submissions_srgb(submission_folder):
    '''
    Bundles submission data for sRGB denoising

    submission_folder Folder where denoised images reside

    Output is written to <submission_folder>/bundled/. Please submit
    the content of this folder.
    '''
    out_folder = os.path.join(submission_folder, "bundled/")
    try:
        os.mkdir(out_folder)
    except:
        pass
    israw = False
    eval_version = "1.0"

    for i in range(50):
        Idenoised = np.zeros((20,), dtype=object)
        for bb in range(20):
            filename = '%04d_%02d.mat' % (i + 1, bb + 1)
            s = sio.loadmat(os.path.join(submission_folder, filename))
            Idenoised_crop = s["Idenoised_crop"]
            Idenoised[bb] = Idenoised_crop
        filename = '%04d.mat' % (i + 1)
        sio.savemat(os.path.join(out_folder, filename),
                    {"Idenoised": Idenoised,
                     "israw": israw,
                     "eval_version": eval_version},
                    )

def main(opt):
    test_set = DNDBenchmarkPNGANDataset()
    test_loader = DataLoader(test_set, batch_size=1)

    if os.path.exists(opt['mat_dir']):
        shutil.rmtree(opt['mat_dir'])
    os.makedirs(opt['mat_dir'])

    Model = getattr(__import__('model'), opt['model'])
    model = Model(opt)
    model.data_parallel()
    if 'resume_from' in opt:
        model.load_model(opt['resume_from'])
    if opt['ensemble']:
        model = EnsembleWrapper(model)

    psnrs, count = 0, 0
    for data in tqdm(test_loader):
        output = model.validation_step(data)
        output = torch.clamp(output, 0, 1)

        output = output.permute(0, 2, 3, 1).cpu().numpy()
        i = count // 20
        k = count % 20
        save_file = os.path.join(opt['mat_dir'], '%04d_%02d.mat' % (i + 1, k + 1))
        sio.savemat(save_file, {'Idenoised_crop': output})
        count += 1

        output = output[0]
        gt = data['H'].squeeze(0).permute(1, 2, 0).numpy()
        psnr = peak_signal_noise_ratio(output, gt, data_range=1)
        psnrs += psnr

    print('%s, psnr: %6.4f' % (test_set.__class__.__name__, psnrs / count))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the denoiser")
    parser.add_argument("--config_file", type=str, default='../option/tbsn_sidd.json')
    argspar = parser.parse_args()

    opt = parse(argspar.config_file)
    opt['mat_dir'] = 'dnd_mat'
    opt['ensemble'] = True
    recursive_print(opt)

    main(opt)
    bundle_submissions_srgb(opt['mat_dir'])