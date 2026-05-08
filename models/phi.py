"""
Structural phi feature precomputation for GAT-with-phi.

Given an adjacency matrix, computes per-edge features:
  - phi_deg[i, j] = deg(j) / max_deg          (normalised target-node degree)
  - phi_jac[i, j] = |N_i n N_j| / |N_i u N_j|  (Jaccard similarity of neighbourhoods)

These are PURE STRUCTURAL signals: they cannot be inferred from node features,
so injecting them into GAT's attention score gives the model information it
otherwise has no access to.
"""
import numpy as np
import scipy.sparse as sp


def compute_phi(adj):
    """
    Compute dense phi feature tensor.

    Args:
        adj: [N, N] adjacency matrix (numpy array or scipy sparse).
             Should already include self-loops if you want them.

    Returns:
        phi: [N, N, 2] float32 array. Last axis: [phi_deg, phi_jac].
             Suitable for the dense attn_head.
    """
    if sp.issparse(adj):
        A = adj.tocsr().astype(np.float32)
    else:
        A = sp.csr_matrix(adj).astype(np.float32)

    # binarise just in case
    A.data = np.ones_like(A.data, dtype=np.float32)

    N = A.shape[0]
    degrees = np.asarray(A.sum(axis=1)).flatten()  # [N]
    deg_max = max(float(degrees.max()), 1.0)

    # phi_deg[i, j] = deg(j) / deg_max  (depends only on j -> broadcast across rows)
    phi_deg = np.tile((degrees / deg_max).astype(np.float32), (N, 1))  # [N, N]

    # Jaccard similarity. For undirected binary A:
    #   |N_i n N_j| = (A A^T)[i, j]
    #   |N_i u N_j| = deg(i) + deg(j) - |N_i n N_j|
    inter = (A @ A.T).toarray().astype(np.float32)  # [N, N]
    union = degrees[:, None] + degrees[None, :] - inter
    phi_jac = np.divide(
        inter, union,
        out=np.zeros_like(inter, dtype=np.float32),
        where=(union > 0)
    ).astype(np.float32)

    phi = np.stack([phi_deg, phi_jac], axis=-1)  # [N, N, 2]
    return phi


def compute_phi_sparse(adj):
    """
    Sparse variant: only return phi values for edges present in adj.

    Returns:
        indices: [E, 2] int64 array of edge indices, sorted by (row, col)
        values:  [E, 2] float32 array of [phi_deg, phi_jac] per edge
    """
    if sp.issparse(adj):
        A = adj.tocsr().astype(np.float32)
    else:
        A = sp.csr_matrix(adj).astype(np.float32)
    A.data = np.ones_like(A.data, dtype=np.float32)

    degrees = np.asarray(A.sum(axis=1)).flatten()
    deg_max = max(float(degrees.max()), 1.0)

    A_coo = A.tocoo()
    rows = A_coo.row
    cols = A_coo.col

    # Sort by (row, col) to match TF SparseTensor convention
    order = np.lexsort((cols, rows))
    rows = rows[order]
    cols = cols[order]

    # Compute Jaccard only for edges that exist
    AAT = (A @ A.T).toarray()  # still O(N^2) memory; for very large graphs use a smarter approach
    inter = AAT[rows, cols].astype(np.float32)
    union = degrees[rows] + degrees[cols] - inter
    phi_jac = np.where(union > 0, inter / union, 0.0).astype(np.float32)

    phi_deg = (degrees[cols] / deg_max).astype(np.float32)

    indices = np.stack([rows, cols], axis=-1).astype(np.int64)
    values = np.stack([phi_deg, phi_jac], axis=-1).astype(np.float32)
    return indices, values
