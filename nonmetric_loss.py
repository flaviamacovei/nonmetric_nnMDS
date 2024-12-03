import numpy as np
import torch

def compute_loss(diss_matrix, embed_matrix):
    assert diss_matrix.shape == embed_matrix.shape

    n = diss_matrix.shape[0]

    pairwise_diff_diss = diss_matrix[:, :, None].repeat(1, 1, n) - diss_matrix[:, None, :].repeat(1, n, 1)
    pairwise_diff_embed = embed_matrix[:, :, None].repeat(1, 1, n) - embed_matrix[:, None, :].repeat(1, n, 1)
    ord_1 = torch.tril(torch.tanh(500 * pairwise_diff_diss))
    ord_2 = torch.tril(torch.tanh(500 * pairwise_diff_embed))

    loss = torch.abs(ord_1 - ord_2).sum() / 2
    return loss

def process_loss(preds_dataloader = None, refs_dataloader = None, device='cpu'):

    loss = 0
    for (batch_preds,), (batch_refs,) in zip(preds_dataloader, refs_dataloader):
        batch_preds = batch_preds.to(device)
        batch_refs = batch_refs.to(device)

        batch_loss = compute_loss(torch.cdist(batch_preds, batch_preds), torch.cdist(batch_refs, batch_refs))
        loss += batch_loss.item()

    return loss
