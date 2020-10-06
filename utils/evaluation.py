"""
Code based on bag-of-tricks repo:
https://github.com/michuanhaohao/reid-strong-baseline
"""
import numpy as np
from tqdm import tqdm


def evaluate(distmat: np.ndarray, q_pids: np.ndarray, g_pids: np.ndarray, q_camids: np.ndarray,
             g_camids: np.ndarray, max_rank=50):
    distmat = np.array(distmat, copy=False)
    q_pids, g_pids = np.array(q_pids, copy=False), np.array(g_pids, copy=False)
    q_camids, g_camids = np.array(q_camids, copy=False), np.array(g_camids, copy=False)
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    order = indices
    remove = (g_pids[order] == q_pids.reshape(-1, 1)) & (g_camids[order] == q_camids.reshape(-1, 1))
    keep = np.invert(remove)
    kept = keep.cumsum(axis=1)

    q, g = len(q_pids), len(g_pids)

    valid_matches = matches * keep
    valid_query = (valid_matches.sum(axis=1) > 0)  # at least one matchable (== matched) gallery image
    assert (valid_matches.sum() != 0)  # error: all query identities do not appear in gallery

    final_rank_positions = np.argmax(valid_matches * np.arange(1, g + 1), axis=1)
    final_rank_valid = kept[np.arange(q), final_rank_positions]
    all_INP = valid_matches.sum(axis=1) / final_rank_valid.astype("float")

    # `kept` is analogous to index within only-valid instances
    cum_precision = (valid_matches.cumsum(axis=1) / kept.astype("float"))
    cum_precision[np.isnan(cum_precision)] = 1
    all_AP = (cum_precision * valid_matches).sum(axis=1) / valid_matches.sum(axis=1)

    # Compute CMC (need to go query-by-query) (assume up to ~50 invalid gallery images)
    keep = keep[:, :max_rank * 2 + 50]
    matches = matches[:, :max_rank * 2 + 50]
    all_cmc = []
    for i in range(q):
        cmc = matches[i][keep[i]].cumsum()
        # E.g., 0 1 x x x x ... to 0 1 1 1 1 1 ...
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / valid_query.astype("int").sum()

    mAP = np.mean(all_AP[valid_query])
    mINP = np.mean(all_INP[valid_query])

    return all_cmc, mAP, mINP


def evaluate_old(distmat: np.ndarray, q_pids: np.ndarray, g_pids: np.ndarray, q_camids: np.ndarray,
                 g_camids: np.ndarray, max_rank=50, test_ratio=1):
    """
    Evaluation with market1501 metric (used for Market, Duke, MSMT)
    Key: for each query identity, its gallery images from the same camera view are discarded.

    Returns all_cnc, mAP, mINP
    """
    if test_ratio > 1 or test_ratio < 0:
        raise ValueError("test_ratio must by in [0, 1]")

    q_pids, g_pids = np.array(q_pids, copy=False), np.array(g_pids, copy=False)
    q_camids, g_camids = np.array(q_camids, copy=False), np.array(g_camids, copy=False)
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in tqdm(range(int(num_q * test_ratio))):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        # E.g., 0 1 1 0 1 1 0 ...
        orig_cmc = matches[q_idx][keep]
        # orig_cmc = matches[q_idx]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        final_rank = np.argmax(orig_cmc * np.arange(1, orig_cmc.shape[0] + 1)).astype(float) + 1
        INP = orig_cmc.sum() / final_rank
        all_INP.append(INP)

        # E.g., 0 1 x x x x ... to 0 0 1 1 1 1 ...
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    return all_cmc, mAP, mINP

