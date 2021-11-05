# Lachie Hill 26/10/2021
# simple parser and einsum contraction using matmul or tensordot
# does no checking for errors in index string, only works for contractions

import numpy as np
import scipy.sparse as sparse


# faster than np.prod for small stuff
def prod(arr):
    d = arr[0]
    for i in arr[1:]:
        d *= i
    return d


# create sparse matrix with desired contracting inds at the end
# current einsum only uses the right form, ie contracting
# terms on the right
class NDSparse():
    def __init__(self, M, end_ind, side='right'):
        if side == 'right':
            loc = list(range(-len(end_ind), 0))
            sh = (-1, prod([M.shape[i] for i in end_ind]))
        else:
            loc = list(range(len(end_ind)))
            sh = (prod([M.shape[i] for i in end_ind]), -1)

        Mtemp = np.moveaxis(M, end_ind, loc)
        self.sparse = sparse.csr_matrix(Mtemp.reshape(sh))
        self.shape = Mtemp.shape


# get einsum index format from a string
def parse_ind_str_to_num(ind_str):
    ab, c = ind_str.split("->")
    a, b = ab.split(',')

    ind_a = [ord(i) for i in a]
    ind_b = [ord(i) for i in b]
    ind_out = [ord(i) for i in c]
    return ind_a, ind_b, ind_out


# get ordered inds for contracting
def axe_order_inds(ind_a, ind_b, ind_out):
    c_a_inds = [i for i, d in enumerate(ind_a) if d in ind_b]
    c_b_inds = [i for f in c_a_inds for i, d in enumerate(ind_b) if ind_a[f] == d]

    final_order_a = [i for d in ind_a for i, f in enumerate(ind_out) if d == f]
    final_order_b = [i for d in ind_b for i, f in enumerate(ind_out) if d == f]
    out_ord = final_order_a + final_order_b

    return c_a_inds, c_b_inds, out_ord


# calculate einsum using the ordered contracting indices
# option to use tensordot, otherwise it defaults to matmul
def einsum__(A, c_a_inds, B, c_b_inds, out_ord, tensd=False):
    c_len = len(c_a_inds)
    c_dim_sz = prod([A.shape[i] for i in c_a_inds])

    r_shape = [A.shape[i] for i in range(len(A.shape)) if i not in c_a_inds] \
              + [B.shape[i] for i in range(len(B.shape)) if i not in c_b_inds]

    if tensd:
        return np.moveaxis(np.tensordot(A, B, axes=(c_a_inds, c_b_inds)).reshape(r_shape), list(range(len(out_ord))),
                           out_ord)

    if type(A) == NDSparse:
        A = A.sparse
    else:
        A = np.moveaxis(A, c_a_inds, list(range(-c_len, 0))).reshape(-1, c_dim_sz)
    B = np.moveaxis(B, c_b_inds, list(range(c_len))).reshape(c_dim_sz, -1)
    R = np.moveaxis((A @ B).reshape(r_shape), list(range(len(out_ord))), out_ord)

    return R


# calculate einsum contraction using a string
ind_map = {}


# TODO: changes to allow a sparse matrix on the right
def einsum(ind_str, A, B, tensd=False):
    # get ordered inds from a string, cache results in map
    if ind_str in ind_map:
        c_a_inds, c_b_inds, out_ord = ind_map[ind_str]
    else:
        c_a_inds, c_b_inds, out_ord = axe_order_inds(*parse_ind_str_to_num(ind_str))
        ind_map[ind_str] = (c_a_inds, c_b_inds, out_ord)

    return einsum__(A, c_a_inds, B, c_b_inds, out_ord, tensd=tensd)
