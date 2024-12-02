import numpy as np
import torch

def compute_loss(diss_matrix, embed_matrix):
    assert diss_matrix.shape == embed_matrix.shape

    # print(f"diss matrix:\n{diss_matrix.cpu().detach().numpy()}")
    # print(f"embed matrix:\n{embed_matrix.cpu().detach().numpy()}")

    n = diss_matrix.shape[0]

    # print(f"ab matrix:\n{ab.cpu().detach().numpy()}")
    # print(f"bc matrix:\n{bc.cpu().detach().numpy()}")

    pairwise_diff_diss = diss_matrix[:, :, None].repeat(1, 1, n) - diss_matrix[:, None, :].repeat(1, n, 1)
    pairwise_diff_embed = embed_matrix[:, :, None].repeat(1, 1, n) - embed_matrix[:, None, :].repeat(1, n, 1)

    # print(f"pairwise diff diss:\n{pairwise_diff_diss.cpu().detach().numpy()}")
    # print(f"pairwise diff embed:\n{pairwise_diff_embed.cpu().detach().numpy()}")

    # ord_1 = torch.tanh(100000 * pairwise_diff_diss)
    # ord_2 = torch.tanh(100000 * pairwise_diff_diss)
    ord_1 = torch.tril(torch.tanh(500 * pairwise_diff_diss))
    ord_2 = torch.tril(torch.tanh(500 * pairwise_diff_embed))

    # print(f"ord 1:\n{ord_1.cpu().detach().numpy()}")
    # print(f"ord 2:\n{ord_2.cpu().detach().numpy()}")
    # print("         ord1                    ord2")
    # for matrix1, matrix2 in zip(ord_1, ord_2):
    #     for row1, row2 in zip(matrix1, matrix2):
    #         print(f"{row1.cpu().detach().numpy()} | {row2.cpu().detach().numpy()}")
    #     print("------------------------")

    loss = torch.abs(ord_1 - ord_2).sum() / 2
    # print(f"loss from inner: {loss}")
    return loss

def process_loss(model=None, dataloader=None, refs_dataloader=None, batch_size=None, optimizer=None, scheduler=None, device='cpu', use_grad=True):
    assert (model is not None and dataloader is not None) or (dataloader is not None and refs_dataloader is not None)

    loss = 0

    if refs_dataloader is None:
        for (batch,) in dataloader:
            if batch.shape[0] < batch_size:
                batch = batch.to('cpu')
                pad = torch.zeros((batch_size - batch.shape[0], batch.shape[1]))
                batch = torch.cat((batch, pad))
            batch = batch.to(device)

            if optimizer:
                optimizer.zero_grad()

            # print(f"batch shape: {batch.shape}")

            # print(f"batch:\n{batch.cpu().detach().numpy()}")

            res = model.encode(batch)

            # print(f"embedded:\n{res.cpu().detach().numpy()}")

            # print(f"encode shape: {res.shape}")
            # print(f"encode tensor:\n{res.cpu().detach().numpy()}")
            batch_loss = compute_loss(torch.cdist(batch, batch), torch.cdist(res, res))
            loss += batch_loss.item()

            if use_grad:
                batch_loss.backward()
                if optimizer:
                    optimizer.step()

        if scheduler:
            scheduler.step(loss)

    else:
        for (batch_data,), (batch_refs,) in zip(dataloader, refs_dataloader):
            # print("called")
            if batch_data.shape[0] < batch_size:
                batch_data = batch_data.to('cpu')
                batch_data = torch.cat(
                    [batch_data, torch.zeros((batch_size - batch_data.shape[0], batch_data.shape[1]))]
                )
                batch_refs = batch_refs.to('cpu')
                batch_refs = torch.cat(
                    [batch_refs, torch.zeros((batch_size - batch_refs.shape[0], batch_refs.shape[1]))]
                )
            batch_data = batch_data.to(device)
            batch_refs = batch_refs.to(device)

            # print(f"batch_refs:\n{torch.cdist(batch_refs, batch_refs).cpu().detach().numpy()}")
            # print(f"batch_data:\n{torch.cdist(batch_data, batch_data).cpu().detach().numpy()}")

            batch_loss = compute_loss(torch.cdist(batch_data, batch_data), torch.cdist(batch_refs, batch_refs))
            # print(f"loss: {batch_loss.item()}")
            loss += batch_loss.item()

    return loss
