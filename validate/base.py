import sys
sys.path.append('..')
import argparse
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from validate.base_function import calculate_ssim
from util.build import build
from util.option import parse, recursive_print

loss_fn_alex = lpips.LPIPS(net='alex').cuda() # best forward scores
def lpips_norm(img):
    img = img * 2. - 1
    return img

def validate_sidd(model, sidd_loader):
    psnrs, ssims, lpipss, count = 0, 0, 0, 0
    for data in tqdm(sidd_loader):
        output = model.validation_step(data)
        output = torch.floor(output * 255. + 0.5) / 255.
        output = torch.clamp(output, 0, 1)
        lpips = loss_fn_alex(lpips_norm(output), lpips_norm(data['H'].cuda()))
        output = output.cpu().squeeze(0).permute(1, 2, 0).numpy()
        gt = data['H'].squeeze(0).permute(1, 2, 0).numpy()
        psnr = peak_signal_noise_ratio(output, gt, data_range=1)
        ssim = structural_similarity(output, gt, data_range=1, channel_axis=2, gaussian_weights=True, sigma=1.5, win_size=11)

        psnrs += psnr
        ssims += ssim
        lpipss += lpips
        count += 1

    return psnrs / count, ssims / count, lpipss / count

def validate_dnd(model, dnd_loader):
    psnrs, ssims, count = 0, 0, 0
    f = open('')
    for data in tqdm(dnd_loader):
        output = model.validation_step(data)
        output = torch.clamp(output, 0, 1)
        output = output.cpu().squeeze(0).permute(1, 2, 0).numpy()
        gt = data['H'].squeeze(0).permute(1, 2, 0).numpy()
        psnr = peak_signal_noise_ratio(output, gt, data_range=1)

        psnrs += psnr
        count += 1
    return psnrs / count, ssims / count

def validate_synthetic(model, synthetic_loader):
    psnrs, ssims, count = 0, 0, 0
    for data in tqdm(synthetic_loader):
        n, c, h, w = data['L'].shape
        if h < w:
            data['L'] = torch.nn.functional.pad(data['L'], [0, 0, 0, w - h], mode='reflect')
        elif h > w:
            data['L'] = torch.nn.functional.pad(data['L'], [0, h - w, 0, 0], mode='reflect')
        output = model.validation_step(data)
        output = output[:, :, :h, :w].cpu().squeeze(0).permute(1, 2, 0).numpy()
        gt = data['H'].squeeze(0).permute(1, 2, 0).numpy()
        psnr = peak_signal_noise_ratio(output, gt, data_range=1)
        ssim = calculate_ssim(output * 255., gt * 255.)

        psnrs += psnr
        ssims += ssim
        count += 1
    return psnrs / count, ssims / count

def main(opt):
    validation_loaders = []
    for validation_dataset_opt in opt['validation_datasets']:
        ValidationDataset = getattr(__import__('dataset'), validation_dataset_opt['type'])
        validation_set = build(ValidationDataset, validation_dataset_opt['args'])
        validation_loader = DataLoader(validation_set, batch_size=1)
        validation_loaders.append(validation_loader)

    Model = getattr(__import__('model'), opt['model'])
    model = Model(opt)
    model.data_parallel()

    for validation_loader in validation_loaders:
        if 'SIDD' in validation_loader.dataset.__class__.__name__:
            psnr, ssim, lpips = validate_sidd(model, validation_loader)
        elif 'DND' in validation_loader.dataset.__class__.__name__:
            psnr, ssim = validate_dnd(model, validation_loader)
        else:
            psnr, ssim = validate_synthetic(model, validation_loader)
        print('%s, psnr: %6.4f, ssim: %6.4f, lpips: %.4f' % (validation_loader.dataset.__class__.__name__, psnr, ssim, lpips))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate the denoiser")
    parser.add_argument("--config_file", type=str, default='../option/tbsn_sidd.json')
    argspar = parser.parse_args()

    opt = parse(argspar.config_file)
    recursive_print(opt)

    main(opt)
