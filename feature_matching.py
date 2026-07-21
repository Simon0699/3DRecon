# x = list of images each with shape (# features, feature description) eg 400, 128
import torch
def exhaustive_match(x, ratio = 0.8):
    # {(i, j): (idx_a, idx_b)} - idx_a[k] in image i matches idx_b[k] in image j
    pair_to_matches = {}
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            d1 = x[i]
            d2 = x[j]
            # descriptors are unit norm, so ||a-b||^2 = 2 - 2*(a.b)
            match_mat = torch.matmul(d1, d2.T)
            top = torch.topk(match_mat, dim = 1, k = 2)
            vals = top.values
            indices = top.indices
            # Lowe: d1 < ratio * d2, rewritten in similarity space
            keep = (1 - vals[:,0]) < (ratio ** 2) * (1 - vals[:,1])
            idx_a = torch.nonzero(keep).squeeze(1)
            idx_b = indices[keep, 0]
            pair_to_matches[(i,j)] = (idx_a, idx_b)
    return pair_to_matches