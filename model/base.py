from abc import abstractmethod
import os
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
from util.build import build
from util.io import log

class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.iter = 0 if 'iter' not in opt else opt['iter']
        self.networks = {}
        for network_opt in opt['networks']:
            Net = getattr(__import__('network'), network_opt['type'])
            net = build(Net, network_opt['args'])
            if 'path' in network_opt.keys():
                self.load_net(net, network_opt['path'])
            self.networks[network_opt['name']] = net

    @abstractmethod
    def train_step(self, data):
        pass

    @abstractmethod
    def validation_step(self, data):
        pass

    def data_parallel(self):
        self.device = torch.device('cuda')
        for name in self.networks.keys():
            net = self.networks[name]
            net = net.cuda()
            net = DataParallel(net)
            self.networks[name] = net

    def distributed_parallel(self, rank):
        self.device = torch.device('cuda:%d' % rank)
        torch.cuda.set_device(rank)
        for name in self.networks.keys():
            net = self.networks[name]
            net = net.to(torch.device('cuda', rank))
            net = DistributedDataParallel(net, device_ids=[rank], output_device=rank)
            self.networks[name] = net

    def save_net(self):
        for name, net in self.networks.items():
            if isinstance(net, (DataParallel, DistributedDataParallel)):
                net = net.module
            torch.save(net.state_dict(), os.path.join(self.opt['log_dir'], '%s_iter_%08d.pth' % (name, self.iter)))

    def load_net(self, net, path):
        state_dict = torch.load(path, map_location='cpu')
        if 'model_weight' in state_dict:
            state_dict = state_dict['model_weight']['denoiser']
        if 'bsn' in list(state_dict.keys())[0]:
            for key in list(state_dict.keys()):
                state_dict[key.replace('bsn.', '')] =  state_dict.pop(key)
        net.load_state_dict(state_dict)


    def log(self):
        with torch.no_grad():
            log(self.opt['log_file'], 'iter: %d, loss: %f\n' % (self.iter, self.loss.item()))
