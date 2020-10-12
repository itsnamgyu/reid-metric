from tqdm import tqdm

from post_rank.feedback import init_feedback_indices, greedy_feedback
from utils.distmat import compute_distmat
from utils.evaluation import evaluate


def adjust_qf(qf, gf, positive_indices, negative_indices, alpha=1, beta=0.65, gamma=0.35):
    assert (qf.shape[1] == gf.shape[1])
    mean_positive_gf = positive_indices.float().mm(gf) / positive_indices.float().sum(dim=1, keepdim=True)
    mean_negative_gf = negative_indices.float().mm(gf) / negative_indices.float().sum(dim=1, keepdim=True)
    mean_positive_gf[mean_positive_gf.isnan()] = 0
    mean_negative_gf[mean_negative_gf.isnan()] = 0
    qf_adjusted = qf * alpha + mean_positive_gf * beta - mean_negative_gf * gamma
    return qf_adjusted


def rocchio(qf, gf, q_pids, g_pids, positive_indices=None, negative_indices=None,
            inplace=True, previous_distmat=None, direct_feedback=True, alpha=1, beta=0.65, gamma=0.35, device=None):
    """
    Args:
        qf: q * m
        gf: g * m
        q_pids: q
        g_pids: g
        positive_indices: q * g
        negative_indices: q * g
        inplace:
        previous_distmat: distmat for adjusted_qf (== compute_distmat(qf, gf) only at init)
        direct_feedback: Whether to apply feedback to distmat directly (pos=0, neg=inf)
        alpha:
        beta:
        gamma:
        device: CUDA device. Other Tensor arguments must also be on this device, if specified.
    Returns:
    """
    q, g = qf.shape[0], gf.shape[0]
    assert (qf.shape[1] == gf.shape[1])

    if positive_indices is None: positive_indices = init_feedback_indices(q, g, device=device)
    if negative_indices is None: negative_indices = init_feedback_indices(q, g, device=device)

    if previous_distmat is None:
        qf_adjusted = adjust_qf(qf, gf, positive_indices, negative_indices)
        distmat = compute_distmat(qf_adjusted, gf)
    else:
        distmat = previous_distmat
        if device:
            distmat = distmat.cuda(device)

    positive_indices, negative_indices = greedy_feedback(distmat, q_pids, g_pids, positive_indices,
                                                         negative_indices, inplace=inplace)
    qf_adjusted = adjust_qf(qf, gf, positive_indices, negative_indices, alpha=alpha, beta=beta, gamma=gamma)
    distmat = compute_distmat(qf_adjusted, gf)

    if direct_feedback:
        # Can still be used as previous_distmat in subsequent function call
        distmat[positive_indices] = 0
        distmat[negative_indices] = float("inf")

    return distmat, positive_indices, negative_indices


def example_usage(qf, gf, q_pids, g_pids, q_camids, g_camids):
    # These will be initialized automatically
    positive_indices = None
    negative_indices = None
    distmat = None
    for _ in tqdm(range(5)):
        res = rocchio(qf, gf, q_pids, g_pids, positive_indices, negative_indices, previous_distmat=distmat)
        distmat, positive_indices, negative_indices = res
    result = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    print("mAP", result[1], "mINP", result[2])
