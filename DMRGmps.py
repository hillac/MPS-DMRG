import numpy as np
import scipy.sparse.linalg
import scipy.sparse as sparse
from time import time as tm
from einsum_tools import *


# a class to store the operators with the bottom and left indices,
# and bottom and right indices moved to the right for contraction
# while sparse
class SparseOperator():
    def __init__(self, Op):
        self.shape = Op.shape
        self.dtype = Op.dtype
        # l_form mujd
        self.l_form = NDSparse(Op, [0, 2])
        # r_form jumd
        self.r_form = NDSparse(Op, [1, 2])


# contracts terms into the left tensor
def contract_left(Op, Md, Mu, L):
    # Mu = Mu.conj()
    L = einsum("dil,jik->dljk", Md, L)
    if type(Op) == SparseOperator:
        L = einsum("mujd,dljk->lkmu", Op.l_form, L)
    else:
        L = einsum("dljk,jmdu->lkmu", L, Op)
    L = einsum("lkmu,ukn->mln", L, Mu)
    return L


# contracts terms into the right tensor
def contract_right(Op, Md, Mu, R):
    # Mu = Mu.conj()
    R = einsum("dil,mln->dimn", Md, R)
    if type(Op) == SparseOperator:
        R = einsum("jumd,dimn->inju", Op.r_form, R)
    else:
        R = einsum("dimn,jmdu->inju", R, Op)
    R = einsum("inju,ukn->jik", R, Mu)
    return R


# returns the expectation of an MPO
def expectation(MPS, MPO):
    E = np.array([[[1]]])
    for i in range(len(MPS)):
        E = contract_left(MPO[i], MPS[i], MPS[i], E)
    return E[0, 0, 0]


# contracts two mpos by the common indices
def contract_mpo(MPO1, MPO2):
    MPO = []
    for i in range(len(MPO1)):
        MPO += [einsum("ijqu,kldq->ikjldu", MPO1[i], MPO2[i])
                    .reshape(MPO1[i].shape[0] * MPO2[i].shape[0],
                             MPO1[i].shape[1] * MPO2[i].shape[1],
                             MPO2[i].shape[2], MPO1[i].shape[3])]
    return MPO


# A liear operator for the sparse eigenvalue problem
class SparseHamProd(sparse.linalg.LinearOperator):
    def __init__(self, L, OL, OR, R):
        self.L = L
        self.OL = OL
        self.OR = OR
        self.R = R
        self.dtype = OL.dtype
        self.issparse = type(OL) == SparseOperator
        self.req_shape = [OL.shape[2], OR.shape[2], L.shape[1], R.shape[1]]
        self.req_shape2 = [OL.shape[2] * OR.shape[2], L.shape[1], R.shape[1]]
        self.size = prod(self.req_shape)
        self.shape = [self.size, self.size]

    # return the output of H*B
    def _matvec(self, B):
        L = einsum("jik,adil->jkadl", self.L, np.reshape(B, self.req_shape))
        if self.issparse: # for sparse
            L = einsum("cbja,jkadl->kdlcb", self.OL.l_form, L)
            L = einsum("mucd,kdlcb->klbmu", self.OR.l_form, L)
        else:
            L = einsum("jkadl,jcab->kdlcb", L, self.OL)
            L = einsum("kdlcb,cmdu->klbmu", L, self.OR)
        L = einsum("klbmu,mln->bukn", L, self.R)

        return np.reshape(L, -1)


# truncates the svd output by m
def trunacte_svd(u, s, v, m):
    if len(s) < m: m = len(s)
    truncation = s[m:].sum()
    u = u[:, :, :m]
    s = s[:m]
    v = v[:m, :, :]
    return u, s, v, truncation, m


# optimises the current site
def optimize_sites(M1, M2, O1, O2, L, R, m, heading=True, tol=0):
    # generate intial guess B
    B = einsum("aiz,dzl->adil", M1, M2)
    # create sparse operator
    H = SparseHamProd(L, O1, O2, R)
    # solve for lowest energy state
    E, V = sparse.linalg.eigsh(H, 1, v0=B, which='SA', tol=tol)
    V = V[:, 0].reshape(H.req_shape)

    # re-arange output so the indices are in the correct location
    V = np.moveaxis(V, 1, 2)  # aidl
    V = V.reshape(O1.shape[2] * L.shape[1], O2.shape[2] * R.shape[1])

    # truncate
    u, s, v = np.linalg.svd(V)
    u = u.reshape(O1.shape[2], L.shape[1], -1)
    v = v.reshape(-1, O2.shape[2], R.shape[1])
    u, s, v, trunc, m_i = trunacte_svd(u, s, v, m)

    # if going right, contract s into the right unitary, else left
    if heading:
        # v = einsum_with_str("ij,djl->dil", np.diag(s), v)
        v = s[:, None] * v.reshape(-1, O2.shape[2] * R.shape[1])  # broadcasting should be faster
        v = v.reshape(-1, O2.shape[2], R.shape[1])
    else:
        # u = einsum_with_str("dik,kl->dil", u, np.diag(s))
        u = u.reshape(O1.shape[2] * L.shape[1], -1) * s
        u = u.reshape(O1.shape[2], L.shape[1], -1)
    v = np.moveaxis(v, 0, 1)
    return E[0], u, v, trunc, m_i


def two_site_DMRG(MPS, MPO, m, num_sweeps, verbose=1):
    N = len(MPS)
    # get first Rj tensor
    R = [np.array([[[1.0]]])]
    # find Rj tensors starting from the right
    for j in range(N - 1, 1, -1):
        R += [contract_right(MPO[j], MPS[j], MPS[j], R[-1])]
    L = [np.array([[[1.0]]])]

    # lists for storing outputs
    t = [];
    E_s = [];
    E_j = []

    for i in range(num_sweeps):
        t0 = tm()
        # sweep right
        for j in range(0, N - 2):
            # optimise going right
            E, MPS[j], MPS[j + 1], trunc, m_i = optimize_sites(MPS[j], MPS[j + 1], MPO[j], MPO[j + 1], L[-1], R[-1], m,
                                                               tol=0, heading=True)
            R = R[:-1]  # remove leftmost R tensor
            L += [contract_left(MPO[j], MPS[j], MPS[j], L[-1])]  # add L tensor
            E_j += [E]
            if verbose >= 3: print(E, "sweep right", i, "sites:", (j, j + 1), "m:", m_i)

        # sweep left
        for j in range(N - 2, 0, -1):

            E, MPS[j], MPS[j + 1], trunc, m_i = optimize_sites(MPS[j], MPS[j + 1], MPO[j], MPO[j + 1], L[-1], R[-1], m,
                                                               tol=0, heading=False)
            R += [contract_right(MPO[j + 1], MPS[j + 1], MPS[j + 1], R[-1])]  # add R tensor
            L = L[:-1]  # remove L tensor
            E_j += [E]
            if verbose >= 3: print(E, "sweep left", i, "sites:", (j, j + 1), "m:", m_i)

        t1 = tm()
        t += [t1 - t0]
        E_s += [E]
        if verbose >= 2: print("sweep", i, "complete")

    if verbose >= 1: print("N:", N, "m:", m, "time for", num_sweeps, "sweeps:", *t)
    return MPS, t, E_j, E_s


# create |0101..> state
def construct_init_state(d, N):
    down = np.zeros((d, 1, 1))
    down[0, 0, 0] = 1
    up = np.zeros((d, 1, 1))
    up[1, 0, 0] = 1
    # state 0101...
    return [down, up] * (N // 2) + [down] * (N % 2)


def construct_MPO(N, type="heisenberg", h=1, issparse=False):
    # operators
    I = np.identity(2)
    Z = np.zeros([2, 2])
    Sz = np.array([[0.5, 0], [0, -0.5]])
    Sp = np.array([[0, 0], [1, 0]])
    Sm = np.array([[0, 1], [0, 0]])
    sz = np.array([[0, 1], [1, 0]])
    sx = np.array([[0, -1j], [1j, 0]])

    # heisenberg MPO
    if type == "h":
        W = np.array([[I, Sz, 0.5 * Sp, 0.5 * Sm, Z],
                      [Z, Z, Z, Z, Sz],
                      [Z, Z, Z, Z, Sm],
                      [Z, Z, Z, Z, Sp],
                      [Z, Z, Z, Z, I]])

        W0 = np.array([[I, Sz, 0.5 * Sp, 0.5 * Sm, Z]])
        Wn = np.array([[Z], [Sz], [Sm], [Sp], [I]])

    else:  # ising model mpo
        assert (type == "i")
        W = np.array([[I, sz, h * sx],
                      [Z, Z, sz],
                      [Z, Z, I]])

        W0 = np.array([[I, sz, h * sx]])
        Wn = np.array([[h * sx], [sz], [I]])

    # create H^2 terms
    [W02, W2, Wn2] = contract_mpo([W0, W, Wn], [W0, W, Wn])

    if issparse: # convert to sparse
        W = SparseOperator(W)
        W0 = SparseOperator(W0)
        Wn = SparseOperator(Wn)

    MPO = [W0] + ([W] * (N - 2)) + [Wn]
    MPO2 = [W02] + ([W2] * (N - 2)) + [Wn2]

    return MPO, MPO2


d = 2  # visible index dimension
N_list = [10, 20, 40, 80]  # number of sites
m_list = [2 ** i for i in range(7, 8)]  # truncation size / bond dimensionality
# N_list = [10]
# m_list = [20, 50]
model = "h"  # model type, h heis, i ising
num_sweeps = 6  # full sweeps
reps = 5  # repetitions
vb = 2  # verbosity
use_sparse = False

t = []
E = []
Var = []
E_sweeps = []
E_steps = []
t_sweeps = []

# run for all configurations
for N in N_list:
    MPO, MPO2 = construct_MPO(N, type=model, issparse=use_sparse)
    E_steps2 = []
    for m in m_list:
        for r in range(reps):
            MPS = construct_init_state(d, N)
            t0 = tm()
            MPS, t_s, E_j, E_s = two_site_DMRG(MPS, MPO, m, num_sweeps, verbose=vb)
            t1 = tm()
            E1 = np.real(expectation(MPS, MPO))
            E2 = np.real(expectation(MPS, MPO2))

            E += [E1]
            Var += [E2 - E1 * E1]
            t += [t1 - t0]
            E_sweeps += [E_s]
            E_steps += [E_j]
            t_sweeps += [t_s]
            E_steps2 += [E_j]

            print("N", N, "m", m, "rep", r, "time:", t1 - t0, "energy:", E1, "var", Var[-1])

E = np.array(E).reshape(len(N_list), len(m_list), reps)
Var = np.array(Var).reshape(len(N_list), len(m_list), reps)
t = np.array(t).reshape(len(N_list), len(m_list), reps)
E_steps2 = np.array(E_steps2).reshape(len(m_list), reps, -1)

import csv
# print the outputs of each trial
file = open(model + "out.csv", 'w', newline='')
f = csv.writer(file)
f.writerow(["N", "m", "reps", "E", "var", "t", "dt"])
for i in range(len(N_list)):
    for j in range(len(m_list)):
        f.writerow([N_list[i], m_list[j], reps, E[i, j, 0], Var[i, j, 0], t[i, j, :].mean(), t[i, j, :].std()])
file.close()

# print all times for each rpeetition for more detailed analysis later
file = open(model + "tout.csv", 'w', newline='')
f = csv.writer(file)
f.writerow(["N", "m", "rep", "t"])
for i in range(len(N_list)):
    for j in range(len(m_list)):
        for r in range(reps):
            f.writerow([N_list[i], m_list[j], r, *t_sweeps[len(m_list) * reps * i + reps * j + r]])
file.close()

# print the energy found for each iteration
file = open(model + "Eout.csv", 'w', newline='')
f = csv.writer(file)
f.writerow(["m", "E"])
for j in range(len(m_list)):
    for i in range((N_list[-1] - 2) * 2 * num_sweeps):
        f.writerow([m_list[j], E_steps2[j, 0, i]])
file.close()
