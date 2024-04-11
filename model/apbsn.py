from model.base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F

def pixel_shuffle_down_sampling(x:torch.Tensor, f:int, train=False):
    b,c,w,h = x.shape
    unshuffled = F.pixel_unshuffle(x, f)
    unshuffled = unshuffled.view(b, c, f * f, w // f, h // f).permute(0, 2, 1, 3, 4)
    if train:
        unshuffled = unshuffled[:, 0]
        unshuffled = unshuffled.reshape(b, c, w // f, h // f)
    else:
        unshuffled = unshuffled.reshape(b * f * f, c, w // f, h // f)
    return unshuffled

def pixel_shuffle_up_sampling(x:torch.Tensor, f:int):
    b,c,h,w = x.shape
    before_shuffle = x.view(b // (f * f), f * f, c, h, w).permute(0, 2, 1, 3, 4)
    before_shuffle = before_shuffle.reshape(b // (f * f), c * f * f, h, w)
    return F.pixel_shuffle(before_shuffle, f)

class APBSNModel(BaseModel):
    def __init__(self, opt):
        super(APBSNModel, self).__init__(opt)
        self.pd_a = opt['pd_a']
        self.pd_b = opt['pd_b']
        self.R3 = opt['R3']
        self.R3_T = opt['R3_T']
        self.R3_p = opt['R3_p']
        self.criteron = nn.L1Loss(reduction='mean')
        self.optimizer = torch.optim.AdamW(self.networks['bsn'].parameters(), lr=opt['lr'])

        milestones = [int(opt['num_iters'] * 0.4), int(opt['num_iters'] * 0.8)]
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)

    def train_step(self, data):
        input = data['L'].to(self.device) * 255.

        self.networks['bsn'].train()
        input = pixel_shuffle_down_sampling(input, f=self.pd_a, train=True)
        output = self.networks['bsn'](input)

        self.loss = self.criteron(output, input)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.iter += 1

    def validation_step(self, data):
        input = data['L'].to(self.device) * 255.
        b, c, h, w = input.shape
        input_pd = pixel_shuffle_down_sampling(input, f=self.pd_b)

        self.networks['bsn'].eval()
        with torch.no_grad():
            output_pd = self.networks['bsn'](input_pd)

        output = pixel_shuffle_up_sampling(output_pd, f=self.pd_b)
        if not self.R3:
            ''' Directly return the result (w/o R3) '''
            denoised = output[:, :, :h, :w]
        else:
            denoised = torch.empty(*(input.shape), self.R3_T, device=input.device)
            # torch.manual_seed(0)
            for t in range(self.R3_T):
                indice = torch.rand_like(input)
                mask = indice < self.R3_p

                tmp_input = torch.clone(output).detach()
                tmp_input[mask] = input[mask]
                with torch.no_grad():
                    tmp_output = self.networks['bsn'](tmp_input)
                denoised[..., t] = tmp_output

            denoised = torch.mean(denoised, dim=-1)

        return denoised / 255.
