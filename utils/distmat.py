"""
Rerank code based on bag-of-tricks repo:
https://github.com/michuanhaohao/reid-strong-baseline
"""
import numpy as np
import torch


def compute_distmat(qf: torch.Tensor, gf: torch.Tensor):
    """
    Args:
        qf: Tensor(q, m)
        gf: Tensor(g, m)

    Returns:
        Tensor(q, g)
        Euclidean squared distance
    """
    assert (qf.shape[1] == gf.shape[1])
    q, g = qf.shape[0], gf.shape[0]

    qq = qf.pow(2).sum(dim=1, keepdim=True).expand(q, g)
    gg = gf.pow(2).sum(dim=1, keepdim=True).expand(g, q)

    distmat = (qq + gg.t()).addmm(mat1=qf, mat2=gf.t(), alpha=-2, beta=1)

    return distmat


def compute_inner_distmat(features: torch.Tensor):
    """
    Used to obtain `gallery_distmat` or `all_distmat`

    Args:
        features: Tensor(x, m) (x = g for gallery_distmat, x = q + g for all_distmat)

    Returns:
        Tensor(x, x)
        Euclidean squared distance
    """
    n, m = features.shape
    ff = features.pow(2).sum(dim=1, keepdim=True).expand(n, n)
    distmat = (ff + ff.t()).addmm(mat1=features, mat2=features.t(), beta=1, alpha=-2)

    return distmat


def rerank_distmat(all_distmat: torch.Tensor, q, k1=20, k2=6, lambda_value=0.3, cut=True):
    """
    Args:
        all_distmat: Tensor(q + g, q + g)

    Returns:
        Tensor(q, g) if cut else Tensor(q + g, q + g)
        Euclidean squared distance
    """
    all_distmat = all_distmat.cpu().numpy()  # TODO
    assert(all_distmat.shape[0] == all_distmat.shape[1])
    all_num = all_distmat.shape[0]
    query_num = q
    gallery_num = all_num  # simply matched to bot code. don't ask me.

    original_dist = all_distmat
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    if cut:
        final_dist = final_dist[:query_num, query_num:]
    return final_dist

