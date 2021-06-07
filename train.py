import argparse
import numpy as np
import os
import torch

from datetime import datetime
from pathlib import Path
from sklearn import metrics

from model import nnMDS
from model import load_data
import torch.utils.data as data
import torch.nn as nn

def train(rundir, epochs, learning_rate, use_gpu, file_name, load_state, hidden_dim):

    model = nnMDS(784, hidden_dim, 7840)
    if not load_state is None:
        state_dict = torch.load(load_state, map_location = (None if use_gpu else 'cpu'))
        model.load_state_dict(state_dict)

    if use_gpu:
        model = model.cuda()
    loss_fn = nn.MSELoss(reduction = 'sum')

    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay = 0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0, patience = 8, cooldown = 0, min_lr = 1e-5, verbose = True)
    start_time = datetime.now()

    for epoch in range(epochs):
        change = datetime.now() - start_time
        print(f'starting epoch {epoch + 1}. Time passed: {change}')
        train_loader = load_data(database = 'MNIST')

        model.train()

        total_loss = 0
        num_batch = 0

        for (batch, _) in train_loader:
            batch = batch.view(-1, 784)
            optimizer.zero_grad()

            if use_gpu:
                batch = batch.cuda()

            res = model.encode(batch)
            loss = loss_fn(torch.pdist(batch), torch.pdist(res))

            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            num_batch += 1

        scheduler.step(total_loss)
        avg_loss = total_loss / num_batch
        print(f'avg_loss: {avg_loss:0.4f} total_loss: {total_loss:0.4f}')

    if file_name is None:
        file_name = f'train{total_loss:0.4f}_epoch{epoch+1}'
    save_path = Path(rundir) / file_name
    torch.save(model.state_dict(), save_path)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rundir', type = str, default = './')
    parser.add_argument('--gpu', action = 'store_true')
    parser.add_argument('--learning_rate', default = 1e-03, type = float)
    parser.add_argument('--weight_decay', default = 0.01, type = float)
    parser.add_argument('--epochs', default = 50, type = int)
    parser.add_argument('--name', default = None)
    parser.add_argument('--load', default = None)
    parser.add_argument('--hidden_dim', type = int, default = 20)
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    os.makedirs(args.rundir, exist_ok = True)

    train(args.rundir, args.epochs, args.learning_rate, args.gpu, args.name, args.load, args.hidden_dim)

    
