"""
NOT TESTED
"""
import torch


def compute_distmat(qf, gf):
    """
    Args:
        qf: Tensor(q, m)
        gf: Tensor(g, m)

    Returns:
        Tensor(q, g)
        Euclidean squared distance
    """
    assert(qf.shape[1] == gf.shape[1])
    q = qf.shape[0]
    g = gf.shape[1]

    qq = qf.pow(2).sum(axis=1, keep_dims=True).expand(q, g)
    gg = gf.pow(2).sum(axis=1, keep_dims=True).expand(g, q)

    distmat = torch.addmm(beta=1, self=qq + gg.t(), alpha=-2, mat1=qf, mat2=gf.t())

    return distmat


def compute_gallery_distmat(gf):
    """
    Args:
        gf: Tensor(g, m)

    Returns:
        Tensor(g, g)
        Euclidean squared distance
    """
    g, m = gf.shape
    gg = gf.pow(2).sum(axis=1, keep_dims=True).expand(g, g)
    distmat = torch.addmm(beta=1, self=gg + gg.t(), alpha=-2, mat1=gf, mat2=gf.t())

    return distmat
