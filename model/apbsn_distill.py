from model.base import BaseModel
from model.apbsn import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling
import torch
import torch.nn as nn


class APBSNDistillModel(BaseModel):
    def __init__(self, opt):
        super(APBSNDistillModel, self).__init__(opt)
        self.pd_a = opt['pd_a']
        self.pd_b = opt['pd_b']
        self.R3 = opt['R3']
        self.R3_T = opt['R3_T']
        self.R3_p = opt['R3_p']
        self.criteron = nn.L1Loss(reduction='mean')
        self.optimizer = torch.optim.AdamW(self.networks['network'].parameters(), lr=opt['lr'])

        milestones = [int(opt['num_iters'] * 0.4), int(opt['num_iters'] * 0.8)]
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)


    def train_step(self, data):

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

        target = denoised / 255.

        input = data['L'].to(self.device)
        self.networks['network'].train()
        output = self.networks['network'](input)

        self.loss = self.criteron(output, target)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.iter += 1


    def validation_step(self, data):
        input = data['L'].to(self.device)

        self.networks['network'].eval()
        with torch.no_grad():
            output = self.networks['network'](input)

        return output