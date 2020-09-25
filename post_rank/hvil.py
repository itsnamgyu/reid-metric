import numpy as np
import torch


class HVIL:
    """
    HVIL post-ranking method, based on paper:
    Human-In-The-Loop Person Re-Identification, Hanxiao Wang, et al. 2018.
    https://arxiv.org/abs/1612.01345
    """

    def __init__(self, gf, qf):
        """
        qf: ndarray(q=q_count, m=n_features)
        gf: ndarray(g=g_count, m=n_features)
        """
        assert (gf.shape[1] == qf.shape[1])
        self.m = gf.shape[1]  # size of feature dimension
        self.qf = qf  # query features
        self.gf = gf  # gallery features
        self.q = qf.shape[0]  # number of query images
        self.g = gf.shape[0]  # number of gallery images
        self.model = np.identity(self.m, dtype="float32")  # M from original paper

    def compute_distance_matrix(self):
        qf = torch.from_numpy(self.qf)
        gf = torch.from_numpy(self.gf)
        model = torch.from_numpy(self.model)
        qfr = qf.reshape(self.q, 1, self.m).repeat(1, self.g, 1)
        gfr = gf.reshape(1, self.g, self.m).repeat(self.q, 1, 1)
        z = qfr - gfr
        zm = torch.matmul(z, model)
        zmz = torch.mul(zm, z)
        zmz = zmz.sum(2)

        return zmz.cpu().numpy()

    def compute_distance_vector(self, q_index):
        """
        Calculate distance vector for given query

        :param q_index:
        Query index within given qf matrix
        :return:
        ndarray(g)
        """
        qf = torch.from_numpy(self.qf[q_index])
        gf = torch.from_numpy(self.gf)
        model = torch.from_numpy(self.model)
        qfr = qf.repeat(self.g, 1)
        z = qfr - gf
        zm = torch.matmul(z, model)
        zmz = torch.mul(zm, z)
        zmz = zmz.sum(1)

        return zmz.cpu().numpy()

    def feedback(self, q_index, g_index, match):
        """
        Provide human feedback (a single annotation) to update model

        :param q_index:
        Index of query image (within given qf matrix)
        :param g_index:
        Index of gallery image (within given gf matrix)
        :param match:
        Boolean, whether the two images match
        :return:
        None
        """
