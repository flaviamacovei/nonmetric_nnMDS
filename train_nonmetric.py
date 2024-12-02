import argparse
import random

import numpy as np
import os
import torch
from torch.utils.data import TensorDataset

from datetime import datetime
from pathlib import Path
import wandb
from sklearn.manifold import MDS

from model import nonmetric_nnMDS

from torch.utils.data import DataLoader

from nonmetric_loss import process_loss, compute_loss
from colour_data import dataset
# from data import dataset
from utils import visualise_embeddings, visualise
import random

import matplotlib.pyplot as plt

# seed = random.randint(0, 10000)
# seed = 3803
seed = 5633
print(f"seed number is: {seed}")
torch.manual_seed(seed)


def train(model, rundir, epochs, learning_rate, batch_size, device, file_name, load_state, wandb_log):
    if not load_state is None:
        state_dict = torch.load(load_state, map_location=(None if device != 'cpu' else 'cpu'))
        model.load_state_dict(state_dict)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0, patience=8, cooldown=0, min_lr=1e-5)#,
                                                           #verbose=True)
    start_time = datetime.now()

    final_loss = 0

    for epoch in range(epochs):
        change = datetime.now() - start_time
        # print(f'starting epoch {epoch + 1}. Time passed: {change}')
        # train_loader = load_data(database='MNIST')
        train_loader = DataLoader(dataset, batch_size= batch_size, shuffle = False)

        model.train() # set model to training mode
        loss = 0

        for i, (batch,) in enumerate(train_loader):
            # if batch.shape[0] < batch_size:
            #     batch = batch.to('cpu')
            #     pad = torch.zeros((batch_size - batch.shape[0], batch.shape[1]))
            #     batch = torch.cat((batch, pad))
            batch = batch.to(device)

            optimizer.zero_grad()

            # print(f"batch shape: {batch.shape}")

            # print(f"batch:\n{batch.cpu().detach().numpy()}")

            batch_cdist = torch.cdist(batch, batch)
            print(f"batch cdist:\n{batch_cdist.cpu().detach().numpy()}")

            res = model.encode(batch)
            # print(f"res:\n{res.cpu().detach().numpy()}")

            res_cdist = torch.cdist(res, res)
            print(f"res cdist:\n{res_cdist.cpu().detach().numpy()}")

            # print(f"encode shape: {res.shape}")
            # print(f"encode tensor:\n{res.cpu().detach().numpy()}")
            batch_loss = compute_loss(batch_cdist, res_cdist)
            loss += batch_loss.item()
            # with torch.no_grad():
            #     visualise(res, f"model_epoch_{epoch}_{i}")

            batch_loss.backward()

            optimizer.step()

            # with torch.no_grad():
            #     visualise(res, f"model_epoch_{epoch + 1}_{i}")

        scheduler.step(loss)

        # loss = process_loss(model = model, dataloader = train_loader, batch_size = batch_size, device = device, optimizer = optimizer, scheduler = scheduler, use_grad = True)
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        # print(f"grad: {str([param.grad.norm().item() for param in model.parameters()])}")

        # print(f'loss: {loss:0.4f}')
        print(f'loss: {loss}')
        if wandb_log:
            wandb.log({"loss": loss})
        final_loss = loss

    if file_name is None:
        file_name = f'train{loss:0.4f}_epoch{epoch + 1}'
    save_path = Path(rundir) / file_name
    torch.save(model.state_dict(), save_path)
    return final_loss


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rundir', type=str, default='./')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--learning_rate', default=1e-03, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--name', default=None)
    parser.add_argument('--load', default=None)
    parser.add_argument('--out_dim', type=int, default=2)
    parser.add_argument('--wandb_log', action='store_true')
    parser.add_argument('--visualise', action='store_true')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    os.makedirs(args.rundir, exist_ok=True)
    device = "cuda" if args.gpu else "cpu"

    if args.wandb_log:
        wandb.init(
            project="nnMDS_experiments",

            config={
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
            }
        )

    in_dim = dataset.tensors[0].shape[1]

    losses_training = []
    losses_eval = []

    for i in range(1):
        print(f"starting iteration {i}")
        # seed = random.randint(0, 10000)
        # torch.manual_seed(seed)

        model = nonmetric_nnMDS(in_dim, args.out_dim)

        train_time_start = datetime.now()
        training_loss = train(model, args.rundir, args.epochs, args.learning_rate, args.batch_size, device, args.name,
                              args.load, args.wandb_log)
        # train(model, args.rundir, args.epochs, args.learning_rate, len(dataset), device, args.name, args.load, args.wandb_log)
        train_time = datetime.now() - train_time_start

        model.eval()

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)

        # print("\n\n\n\n-------\n\n\n\n")

        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        res_model = model.encode(dataset.tensors[0].to(device))
        print(f"batch:\n{torch.cdist(dataset.tensors[0], dataset.tensors[0]).cpu().detach().numpy()}")
        print(f"res:\n{torch.cdist(res_model, res_model).cpu().detach().numpy()}")
        # print(f"res model:\n{res_model.cpu().detach().numpy()}")
        # print(f"res model device: {res_model.device}")
        res_dataset = TensorDataset(res_model)
        refs_loader = DataLoader(res_dataset, batch_size=args.batch_size, shuffle=False)
        # print(f"length: {len(data_loader)}")
        diss_matrix = torch.cdist(dataset.tensors[0], dataset.tensors[0]).to(device)
        final_loss_model = process_loss(dataloader=data_loader, refs_dataloader=refs_loader, batch_size=args.batch_size,
                                        device=device, use_grad=False)
        # final_loss_model = compute_loss(diss_matrix, torch.cdist(res_model, res_model))

        # print(f"final loss model: {final_loss_model}")
        # print(f"training time: {train_time}")
        losses_training.append(training_loss)
        losses_eval.append(final_loss_model)

        if abs(final_loss_model - training_loss) > 1:
            print(f"error on iteration {i}: {seed}\ntraining loss: {training_loss}\neval loss: {final_loss_model}")
        print("")

    # for i, (lt, le) in enumerate(zip(losses_training, losses_eval)):
    #     if lt != le:
    #         print(f"found error on iteration {i}: {lt} != {le}")

    # diss_matrix = 0.5 * (diss_matrix + diss_matrix.T)

    # mds = MDS(
    #     n_components=args.out_dim,
    #     metric=False,
    #     dissimilarity="precomputed",
    #     random_state=42,
    # )
    #
    # smacof_time_start = datetime.now()
    # diss_matrix = diss_matrix.to('cpu')
    # res_smacof = torch.from_numpy(mds.fit_transform(diss_matrix))
    # diss_matrix = diss_matrix.to(device)
    # res_smacof = res_smacof.to(device)
    # smacof_time = datetime.now() - smacof_time_start
    # res_smacof_dataset = TensorDataset(res_smacof)
    # smacof_loader = DataLoader(res_smacof_dataset, args.batch_size, shuffle = False)
    # final_loss_smacof = process_loss(dataloader = data_loader, refs_dataloader = smacof_loader, batch_size = args.batch_size, device = device, use_grad = False)
    # # final_loss_smacof = compute_loss(diss_matrix, torch.cdist(res_smacof, res_smacof))
    #
    # print(f"final loss smacof: {final_loss_smacof}")
    # print(f"smacof time: {smacof_time}")
    #
    # if args.visualise:
    #     with torch.no_grad():
    #         visualise_embeddings(dataset.tensors[0], res_model, res_smacof)
    #
    # if args.wandb_log:
    #     wandb.log({
    #         "final loss model": final_loss_model,
    #         "final loss smacof": final_loss_smacof
    #     })
    # print(f"seed: {seed}")



