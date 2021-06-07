import os
from train import train
import argparse
from datetime import datetime

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action = 'store_true')
    parser.add_argument('--learning_rate', default = 1e-3, type = float)
    parser.add_argument('--epochs', default = 100, type = int)
    parser.add_argument('--name', default = None)
    parser.add_argument('--load', default = None)
    parser.add_argument('--range', nargs = '+', type = int)
    parser.add_argument('--dir', default = './')
    return parser
    
if __name__ == '__main__':
    args = get_parser().parse_args()
    curr_dir = args.dir
    os.makedirs(curr_dir, exist_ok = True)
    
    rangess = [20, 784, 50]
    if args.range:
        for i in range(min(len(args.range), 3)):
            rangess[i] = args.range[i]

    name = f'{datetime.now()}' if args.name is None else args.name
            
    for d in range(*rangess):
        train(curr_dir, args.epochs, args.learning_rate, args.gpu, name + f'_{d}', args.load, d)
    
