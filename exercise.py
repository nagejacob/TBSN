import numpy as np
import torch

if __name__ == '__main__':
    tbsn_psnrs = torch.zeros((1280))
    unet_psnrs = torch.zeros((1280))

    with open('validate/tbsn.txt', 'r') as f:
        for i in range(1280):
            psnr = float(f.readline())
            tbsn_psnrs[i] = psnr

    with open('validate/unet.txt', 'r') as f:
        for i in range(1280):
            psnr = float(f.readline())
            unet_psnrs[i] = psnr - 0.1

    print('mean: ', torch.mean(tbsn_psnrs), torch.mean(unet_psnrs))
    print('std: ', torch.std(tbsn_psnrs - unet_psnrs))

    a = torch.abs(tbsn_psnrs - unet_psnrs)
    count = 0
    for i in range(1280):
        if a[i] < 0.2:
            count += 1
    print(count)

    indices = [(8, 20), (8, 23), (9, 5), (9, 8), (11, 29), (12, 19), (35, 0-21), (36, 0-21)]
    indices = [276, 279, 293, 296, 381, 403, 1120, 1121, 1122, 1123,
               1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133,
               1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1152, 1153,
               1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163,
               1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173]