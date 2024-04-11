import sys
sys.path.append('..')
import argparse
from dataset.sidd import SIDDBenchmarkDataset
import numpy as np
import os
import scipy.io as sio
from benchmark.ensemble_wrapper import EnsembleWrapper
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from util.option import parse, recursive_print

def main(opt):
    test_set = SIDDBenchmarkDataset()
    test_loader = DataLoader(test_set, batch_size=1)

    if os.path.exists(opt['mat_path']):
        os.remove(opt['mat_path'])

    Model = getattr(__import__('model'), opt['model'])
    model = Model(opt)
    model.data_parallel()
    if opt['ensemble']:
        model = EnsembleWrapper(model)

    count = 0
    denoised_block = np.zeros_like(test_set.noisy_block)
    for data in tqdm(test_loader):
        output = model.validation_step(data)
        output = torch.floor(output * 255. + 0.5)
        output = torch.clamp(output, 0, 255)
        output = output.cpu().squeeze(0).permute(1, 2, 0).numpy()

        index_n = count // test_set.noisy_block.shape[1]
        index_k = count % test_set.noisy_block.shape[1]
        output = np.uint8(output)
        denoised_block[index_n, index_k] = output
        count += 1

    save_dict = {}
    save_dict['__header__'] = b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Thu Jan 10 13:08:11 2019'
    save_dict['__version__'] = 1.0
    save_dict['__globals__'] = []
    save_dict['DenoisedBlocksSrgb'] = denoised_block
    sio.savemat(opt['mat_path'], save_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the denoiser")
    parser.add_argument("--config_file", type=str, default='../option/tbsn_sidd.json')
    argspar = parser.parse_args()

    opt = parse(argspar.config_file)
    opt['mat_path'] = 'SubmitSrgb.mat'
    opt['ensemble'] = True
    recursive_print(opt)

    main(opt)