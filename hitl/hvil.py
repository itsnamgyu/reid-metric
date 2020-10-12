import numpy as np
import torch


class HVIL:
    """
    HVIL post-ranking method, based on paper:
    Human-In-The-Loop Person Re-Identification, Hanxiao Wang, et al. 2018.
    https://arxiv.org/abs/1612.01345
    """
    n = 0.2

    def __init__(self, qf, gf):
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

        # l(rank) = l_*[rank] s.t. rank >= 0
        # Note that these are ndarrays
        self.l_match = np.cumsum(np.concatenate([np.array([0]), 1 / np.arange(1, self.g)])).astype("float32")
        self.l_mismatch = np.linspace(1, 0, self.g).astype("float32")

    def compute_distance_matrix(self, numpy=True):
        qf = torch.from_numpy(self.qf)
        gf = torch.from_numpy(self.gf)
        model = torch.from_numpy(self.model)
        qfr = qf.reshape(self.q, 1, self.m).repeat(1, self.g, 1)
        gfr = gf.reshape(1, self.g, self.m).repeat(self.q, 1, 1)
        z = qfr - gfr
        zm = torch.matmul(z, model)
        zmz = torch.mul(zm, z)
        zmz = zmz.sum(2)

        if numpy:
            zmz = zmz.cpu().numpy()
        return zmz

    def compute_distance_vector(self, q_index, numpy=True):
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

        if numpy:
            zmz = zmz.cpu().numpy()
        return zmz

    def _compute_gallery_distance_vector(self, g_index, numpy=True):
        """
        Calculate distance vector for given gallery image (distance to other gallery images

        :param q_index:
        Query index within given qf matrix
        :return:
        ndarray(g)
        """
        sf = torch.from_numpy(self.gf[g_index])  # selected gallery image
        gf = torch.from_numpy(self.gf)
        model = torch.from_numpy(self.model)
        sfr = sf.repeat(self.g, 1)
        z = sfr - gf
        assert(tuple(z.shape) == (self.g, self.m))
        zm = torch.matmul(z, model)
        zmz = torch.mul(zm, z)
        zmz = zmz.sum(1)

        if numpy:
            zmz = zmz.cpu().numpy()
        return zmz

    def feedback(self, q_index, g_index, match):
        """
        Provide human feedback (a single annotation) to update model.
        All RHS variables are constants or torch tensors.

        :param q_index:
        Index of query image (within given qf matrix)
        :param g_index:
        Index of gallery image (within given gf matrix)
        :param match:
        Boolean, whether the two images match
        :return:
        None
        """
        n = torch.tensor(self.n).float()
        dv = self.compute_distance_vector(q_index, numpy=False)
        # gdv = self._compute_gallery_distance_vector(g_index, numpy=False)
        model = torch.from_numpy(self.model)

        f_hat = dv[g_index]
        b_t = 1 if match else -1

        v = (1 + (dv - f_hat) * b_t).argmax()  # index of max hinge-loss gallery image
        f_v = dv[v]

        ranked = dv.argsort()
        rank = torch.where(ranked.eq(g_index))
        print('rank', rank)
        l_hat = torch.tensor(self.l_match[rank] if match else self.l_mismatch[rank]).float()
        print('l_hat', l_hat)

        _part = n * l_hat * (f_v + b_t) * f_hat - 1
        print('_part', _part)
        f_t_top = _part + (_part * _part + 4 * n * l_hat * f_hat * f_hat).sqrt()
        f_t_bot = 2 * n * l_hat * f_hat
        f_t = (f_t_top / f_t_bot).float()  # scalar
        print('f_hat', f_hat)
        print('f_t', f_t)

        z = torch.from_numpy(self.qf[q_index] - self.gf[g_index]).reshape(-1, 1)
        print(z.T.mm(model).mm(z).cpu().numpy().item())
        print(f_hat.cpu().numpy().item())
        assert(tuple(z.shape) == (self.m, 1))
        delta_m_top = l_hat * (f_t - f_v - b_t) * model.mm(z).mm(z.T).mm(model)
        delta_m_bot = 1 + n * l_hat * (f_t - f_v - b_t) * z.T.mm(model).mm(z)  # scalar
        model -= n * delta_m_top / delta_m_bot  # mind the m
        self.model = model.numpy()


