import argparse
import os

import numpy as np

import data
from hitl import *
from project import BASE_DIR
from utils.distmat import *
from utils.evaluation import *

keys = data.get_output_keys()
methods = ["rocchio", "rocchio_positive", "ne_mean", "ne_max", "ne_min", "re_ne_mean", "re_ne_max", "re_ne_min"]
parser = argparse.ArgumentParser()
parser.add_argument("--output", "-O", help="Output key. Omit to display choices.")
parser.add_argument("--method", "-M", help="Method. Omit to display choices.")
parser.add_argument("--device", "-D", type=int, default=None, help="GPU device.")
args = parser.parse_args()

run = True
if args.output not in keys:
    run = False
    print("Choose output key from:")
    print("-" * 80)
    for key in keys:
        print("    {}".format(key))
    print("-" * 80)

if args.method not in methods:
    run = False
    print("Choose method from:")
    print("-" * 80)
    for method in methods:
        print("    {}".format(method))
    print("-" * 80)

dc = torch.cuda.device_count()
if args.device is not None and args.device >= dc:
    run = False
    print("Invalid device index {}. {} devices are available.".format(args.device, dc))
if dc and args.device is None:
    print("Running on CPU. Btw, you have a GPU.")

if not run:
    print("Check arguments plz")
    parser.print_help()
    exit()


def save_match_matrix(match_matrix):
    output_path = os.path.join(BASE_DIR, "output", "match_matrix", "{}_{}.npy".format(args.output, args.method))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, match_matrix)


output = data.load_output(args.output)
qf = torch.as_tensor(output["qf"], device=args.device)
gf = torch.as_tensor(output["gf"], device=args.device)
q_pids = torch.as_tensor(output["q_pids"], device=args.device)
g_pids = torch.as_tensor(output["g_pids"], device=args.device)
q_camids = torch.as_tensor(output["q_camids"], device=args.device)
g_camids = torch.as_tensor(output["g_camids"], device=args.device)
q, g = qf.shape[0], gf.shape[0]

rounds = gf.shape[0]
reduction_rounds = [10, 25, 50, 75, 100, 200, 500, 1000, 2000, 5000]

print("Extracting match matrix on {} using {} method".format(args.output, args.method))

if args.method == "rocchio":
    positive_indices = None
    negative_indices = None
    distmat = compute_distmat(qf, gf)
    incomplete = np.ones(q, dtype=bool)
    match_matrix = np.zeros((q, g), dtype=bool)
    for i in tqdm(range(rounds)):
        distmat, positive_indices, negative_indices, matches = rocchio.rocchio_round(
            qf, gf, q_pids, g_pids, positive_indices, negative_indices, previous_distmat=distmat, device=args.device)
        match_matrix[incomplete, i] = matches.cpu().numpy()
        if i in reduction_rounds:
            matches = match_matrix.sum(axis=1)[incomplete]
            total_matches = (g_pids == q_pids.reshape(-1, 1)).sum(dim=1).cpu().numpy()
            local_incomplete = matches != total_matches
            distmat = distmat[local_incomplete, :]
            positive_indices = positive_indices[local_incomplete, :]
            negative_indices = negative_indices[local_incomplete, :]
            q_pids = q_pids[local_incomplete]
            qf = qf[local_incomplete, :]
            incomplete[incomplete] = local_incomplete

    save_match_matrix(match_matrix)
    del positive_indices, negative_indices

elif args.method == "rocchio_positive":
    positive_indices = None
    negative_indices = None
    distmat = compute_distmat(qf, gf)
    incomplete = np.ones(q, dtype=bool)
    match_matrix = np.zeros((q, g), dtype=bool)
    for i in tqdm(range(rounds)):
        distmat, positive_indices, negative_indices, matches = rocchio.rocchio_round(
            qf, gf, q_pids, g_pids, positive_indices, negative_indices, previous_distmat=distmat, device=args.device,
            gamma=0)
        match_matrix[incomplete, i] = matches.cpu().numpy()
        if i in reduction_rounds:
            matches = match_matrix.sum(axis=1)[incomplete]
            total_matches = (g_pids == q_pids.reshape(-1, 1)).sum(dim=1).cpu().numpy()
            local_incomplete = matches != total_matches
            distmat = distmat[local_incomplete, :]
            positive_indices = positive_indices[local_incomplete, :]
            negative_indices = negative_indices[local_incomplete, :]
            q_pids = q_pids[local_incomplete]
            qf = qf[local_incomplete, :]
            incomplete[incomplete] = local_incomplete

    save_match_matrix(match_matrix)
    del positive_indices, negative_indices

elif args.method[:3] == "ne_":
    method = args.method[3:]

    positive_indices = None
    negative_indices = None
    distmat = None
    distmat_qg = None
    incomplete = np.ones(q, dtype=bool)
    match_matrix = np.zeros((q, g), dtype=bool)
    for i in tqdm(range(rounds)):
        res = ne.ne_round(qf, gf, q_pids, g_pids,
                          positive_indices=positive_indices,
                          negative_indices=negative_indices,
                          distmat=distmat,
                          distmat_qg=distmat_qg, method=method,
                          device=args.device, verbose=0)
        distmat, positive_indices, negative_indices, distmat_qg, matches = res
        del res
        match_matrix[incomplete, i] = matches.cpu().numpy()
        if i in reduction_rounds:
            matches = match_matrix.sum(axis=1)[incomplete]
            total_matches = (g_pids == q_pids.reshape(-1, 1)).sum(dim=1).cpu().numpy()
            local_incomplete = matches != total_matches
            distmat = distmat[local_incomplete, :]
            local_incomplete_qg = np.concatenate([local_incomplete, np.ones(g, dtype=bool)])
            distmat_qg = distmat_qg[local_incomplete_qg, :]
            positive_indices = positive_indices[local_incomplete, :]
            positive_indices = positive_indices[:, local_incomplete_qg]
            negative_indices = negative_indices[local_incomplete, :]
            negative_indices = negative_indices[:, local_incomplete_qg]
            q_pids = q_pids[local_incomplete]
            qf = qf[local_incomplete, :]
            incomplete[incomplete] = local_incomplete
    save_match_matrix(match_matrix)
    del distmat_qg, positive_indices, negative_indices

elif args.method[:6] == "re_ne_":
    method = args.method[6:]

    positive_indices = None
    negative_indices = None
    distmat_all = compute_inner_distmat(torch.cat([qf, gf], dim=0))
    distmat_all = rerank_distmat(distmat_all, q, cut=False)
    distmat = torch.as_tensor(distmat_all[:q, q:], device=args.device)
    distmat_qg = torch.as_tensor(distmat_all[:, q:], device=args.device)
    del distmat_all
    incomplete = np.ones(q, dtype=bool)
    match_matrix = np.zeros((q, g), dtype=bool)
    for i in tqdm(range(rounds)):
        res = ne.ne_round(qf, gf, q_pids, g_pids,
                          positive_indices=positive_indices,
                          negative_indices=negative_indices,
                          distmat=distmat,
                          distmat_qg=distmat_qg, method=method,
                          device=args.device, verbose=0)
        distmat, positive_indices, negative_indices, distmat_qg, matches = res
        del res
        match_matrix[incomplete, i] = matches.cpu().numpy()
        if i in reduction_rounds:
            matches = match_matrix.sum(axis=1)[incomplete]
            total_matches = (g_pids == q_pids.reshape(-1, 1)).sum(dim=1).cpu().numpy()
            local_incomplete = matches != total_matches
            distmat = distmat[local_incomplete, :]
            local_incomplete_qg = np.concatenate([local_incomplete, np.ones(g, dtype=bool)])
            distmat_qg = distmat_qg[local_incomplete_qg, :]
            positive_indices = positive_indices[local_incomplete, :]
            positive_indices = positive_indices[:, local_incomplete_qg]
            negative_indices = negative_indices[local_incomplete, :]
            negative_indices = negative_indices[:, local_incomplete_qg]
            q_pids = q_pids[local_incomplete]
            qf = qf[local_incomplete, :]
            incomplete[incomplete] = local_incomplete
    save_match_matrix(match_matrix)
    del distmat_qg, positive_indices, negative_indices

else:
    raise AssertionError("wut?")
