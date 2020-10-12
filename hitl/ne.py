"""
Neighborhood Expansion Algorithm

"""
import torch
from tqdm import tqdm

from hitl.feedback import init_feedback_indices_qg, greedy_feedback_qg
from utils.distmat import compute_distmat
from utils.evaluation import evaluate


def compute_neighborhood_distmat(distmat_qg, qf, gf, positive_indices_qg, negative_indices_qg, method="min",
                                 device=None, verbose=0):
    """
    :param distmat_qg: (q + g) * g
    :param qf:
    :param gf:
    :param positive_indices_qg:
    :param negative_indices_qg:
    :param method: "min" | "mean"
    :param verbose:
    :param device:
    :return: distmat q x g
    """
    q, g = qf.shape[0], gf.shape[0]
    assert (qf.shape[1] == gf.shape[1])
    assert (tuple(distmat_qg.shape) == (q + g, g))
    assert (tuple(positive_indices_qg.shape) == (q, q + g))
    assert (tuple(negative_indices_qg.shape) == (q, q + g))

    distmat = torch.empty(q, g, dtype=torch.float32, device=device)
    tqdm_if_verbose = tqdm if verbose else (lambda e: e)
    for i in tqdm_if_verbose(range(q)):
        positives = positive_indices_qg[i]
        d = distmat_qg[positives, :]
        if method == "min":
            final_distances, indices = d.min(axis=0)
        elif method == "mean":
            final_distances = d.mean(axis=0)
        else:
            raise ValueError()
        distmat[i] = final_distances

    positive_indices = positive_indices_qg[:, q:]
    negative_indices = negative_indices_qg[:, q:]
    distmat[positive_indices] = 0
    distmat[negative_indices] = float("inf")

    return distmat


def compute_distmat_qg(qf, gf):
    """
    :param qf:
    :param gf:
    :return: (q + g) * g
    """
    return compute_distmat(torch.cat([qf, gf]), gf)


def ne_round(qf, gf, q_pids, g_pids, positive_indices=None, negative_indices=None, distmat_qg=None, method="min",
             device=None, verbose=1):
    """
    Only inplace is supported
    :param qf: q * m
    :param gf: g * m
    :param q_pids: q
    :param g_pids: g
    :param positive_indices: q * (q + g)
    :param negative_indices: q * (q + g)
    :param distmat_qg: (q + g) * g
    :param device: CUDA device. Other Tensor arguments must also be on this device, if specified.
    :return:
    """
    q, g = qf.shape[0], gf.shape[0]
    assert (qf.shape[1] == gf.shape[1])

    if distmat_qg is None:
        distmat_qg = compute_distmat_qg(qf, gf)

    if positive_indices is None: positive_indices = init_feedback_indices_qg(q, g, True, device=device)
    if negative_indices is None: negative_indices = init_feedback_indices_qg(q, g, False, device=device)

    positive_indices, negative_indices = greedy_feedback_qg(distmat_qg, q_pids, g_pids, positive_indices,
                                                            negative_indices)
    if verbose:
        print("Computing min neighborhood distmat")
    distmat = compute_neighborhood_distmat(distmat_qg, qf, gf, positive_indices, negative_indices, method=method,
                                           device=device, verbose=verbose)

    return distmat, positive_indices, negative_indices, distmat_qg


def run(qf, gf, q_pids, g_pids, q_camids, g_camids, t=5, method="min", device=None):
    positive_indices = None
    negative_indices = None
    distmat = None
    distmat_qg = None
    for _ in tqdm(range(t)):
        res = ne_round(qf, gf, q_pids, g_pids, positive_indices, negative_indices, distmat_qg, method=method,
                       verbose=0, device=device)
        distmat, positive_indices, negative_indices, distmat_qg = res
    result = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, device=device)
    print("Results after {} rounds of neighborhood expansion ({}):".format(t, method), "mAP", result[1], "mINP",
          result[2])
