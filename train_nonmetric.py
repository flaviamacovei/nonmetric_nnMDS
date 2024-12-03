import argparse
import os
import torch
from torch.utils.data import TensorDataset

from datetime import datetime
import wandb
from sklearn.manifold import MDS

from model import nonmetric_nnMDS

from torch.utils.data import DataLoader

from nonmetric_loss import process_loss, compute_loss
from colour_data import dataset
from utils import visualise_embeddings, visualise


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
        print(f'starting epoch {epoch + 1}. Time passed: {change}')
        train_loader = DataLoader(dataset, batch_size= batch_size, shuffle = False)

        model.train() # set model to training mode
        loss = 0

        for i, (batch,) in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            res = model.encode(batch)
            batch_cdist = torch.cdist(batch, batch)
            res_cdist = torch.cdist(res, res)
            batch_loss = compute_loss(batch_cdist, res_cdist)
            loss += batch_loss.item()

            batch_loss.backward()
            optimizer.step()

            with torch.no_grad():
                visualise(res, f"model_epoch_{epoch + 1}_{i}")

        scheduler.step(loss)
        print(f'loss: {loss:0.4f}')
        if wandb_log:
            wandb.log({"loss": loss})
        final_loss = loss

    # if file_name is None:
    #     file_name = f'train{loss:0.4f}_epoch{epoch + 1}'
    # save_path = Path(rundir) / file_name
    # torch.save(model.state_dict(), save_path)


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
    model = nonmetric_nnMDS(in_dim, args.out_dim)

    train_time_start = datetime.now()
    train(model, args.rundir, args.epochs, args.learning_rate, args.batch_size, device, args.name,
                          args.load, args.wandb_log)
    train_time = datetime.now() - train_time_start

    model.eval()

    refs_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    preds_model = model.encode(dataset.tensors[0].to(device))
    preds_dataset = TensorDataset(preds_model)
    preds_dataloader = DataLoader(preds_dataset, batch_size=args.batch_size, shuffle=False)
    final_loss_model = process_loss(preds_dataloader=preds_dataloader, refs_dataloader=refs_dataloader, device=device)
    print(f"final loss model: {final_loss_model:0.4f}")
    print(f"training time: {train_time}")

    mds = MDS(
        n_components=args.out_dim,
        metric=False,
        dissimilarity="precomputed",
        random_state=42,
    )

    diss_matrix = torch.cdist(dataset.tensors[0], dataset.tensors[0]).to('cpu')

    smacof_time_start = datetime.now()
    res_smacof = torch.from_numpy(mds.fit_transform(diss_matrix))
    smacof_time = datetime.now() - smacof_time_start

    res_smacof = res_smacof.to(device)
    res_smacof_dataset = TensorDataset(res_smacof)
    smacof_loader = DataLoader(res_smacof_dataset, args.batch_size, shuffle = False)
    final_loss_smacof = process_loss(preds_dataloader = refs_dataloader, refs_dataloader = smacof_loader, device = device)


    print(f"final loss smacof: {final_loss_smacof:0.4f}")
    print(f"smacof time: {smacof_time}")


    if args.visualise:
        with torch.no_grad():
            visualise_embeddings(dataset.tensors[0], preds_model, res_smacof)

    if args.wandb_log:
        wandb.log({
            "final loss model": final_loss_model,
            "final loss smacof": final_loss_smacof
        })



