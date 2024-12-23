import sys
sys.path.append('.')
sys.path.append('..')
import argparse
import datetime
from util.option import parse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the denoiser")
    parser.add_argument("--config", type=str, default='')
    argspar = parser.parse_args()

    opt = parse(argspar.config)

    now = datetime.datetime.now()
    date = now.strftime("%m%d-%H%M%S")

    model = opt['model']
    network = opt['networks'][0]['type']

    name = date + '_' + model + '_' + network
    print(name)