import numpy as np
import tensorflow as tf

def _conv1d(x, out_sz, use_bias=True):
    # Keras Conv1D works in modern TF and avoids deprecated tf.layers.* APIs.
    return tf.keras.layers.Conv1D(out_sz, 1, use_bias=use_bias)(x)


def _dropout(x, drop_prob):
    if drop_prob == 0.0:
        return x
    try:
        # TF2 signature
        return tf.nn.dropout(x, rate=drop_prob)
    except TypeError:
        # TF1 signature
        return tf.nn.dropout(x, keep_prob=1.0 - drop_prob)


def attn_head(seq, out_sz, bias_mat, activation, phi_mat=None,
              in_drop=0.0, coef_drop=0.0, residual=False):
    """
    GAT attention head, optionally augmented with structural phi features.

    Args:
        seq: [batch, N, F] node features
        out_sz: output feature dimension
        bias_mat: [batch, N, N] mask (-inf for non-edges, 0 for edges)
        activation: nonlinearity applied at the end
        phi_mat: [batch, N, N, K] structural features per edge, or None for vanilla GAT
        in_drop, coef_drop, residual: as in original GAT
    """
    with tf.name_scope('my_attn'):
        seq = _dropout(seq, in_drop)

        seq_fts = _conv1d(seq, out_sz, use_bias=False)

        # standard GAT scoring: e_ij = a1^T (W h_i) + a2^T (W h_j)
        f_1 = _conv1d(seq_fts, 1)
        f_2 = _conv1d(seq_fts, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])  # [batch, N, N]

        # ---- structural phi injection ----
        # Each phi feature gets one learnable scalar weight, initialised at 0
        # so the model starts as vanilla GAT and learns to use phi if helpful.
        if phi_mat is not None:
            K = int(phi_mat.shape[-1])
            w_phi = tf.Variable(tf.zeros([K]), name='w_phi', dtype=tf.float32)
            phi_contrib = tf.reduce_sum(phi_mat * w_phi, axis=-1)  # [batch, N, N]
            logits = logits + phi_contrib

        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        coefs = _dropout(coefs, coef_drop)
        seq_fts = _dropout(seq_fts, in_drop)

        vals = tf.matmul(coefs, seq_fts)
        bias = tf.Variable(tf.zeros([int(vals.shape[-1])]), name='bias')
        ret = vals + bias

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + _conv1d(seq, int(ret.shape[-1]), use_bias=True)
            else:
                ret = ret + seq

        return activation(ret)


def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes,
                 phi_indices=None, phi_values=None,
                 in_drop=0.0, coef_drop=0.0, residual=False):
    """
    Sparse GAT attention head with optional phi.

    For sparse mode, phi must be provided in sparse form (only values on edges).

    Args:
        adj_mat: SparseTensor binary adjacency (shape [N, N])
        phi_indices: [E, 2] indices of edges (must match adj_mat.indices), or None
        phi_values: [E, K] phi feature values for each edge, or None
    """
    with tf.name_scope('sp_attn'):
        seq = _dropout(seq, in_drop)

        seq_fts = _conv1d(seq, out_sz, use_bias=False)

        f_1 = _conv1d(seq_fts, 1)
        f_2 = _conv1d(seq_fts, 1)

        f_1 = tf.reshape(f_1, (nb_nodes, 1))
        f_2 = tf.reshape(f_2, (nb_nodes, 1))

        f_1 = adj_mat * f_1
        f_2 = adj_mat * tf.transpose(f_2, [1, 0])

        logits = tf.sparse_add(f_1, f_2)
        logit_values = logits.values

        # ---- sparse phi injection ----
        if phi_values is not None:
            K = int(phi_values.shape[-1])
            w_phi = tf.Variable(tf.zeros([K]), name='w_phi', dtype=tf.float32)
            phi_contrib = tf.reduce_sum(phi_values * w_phi, axis=-1)  # [E]
            logit_values = logit_values + phi_contrib

        lrelu = tf.SparseTensor(indices=logits.indices,
                                values=tf.nn.leaky_relu(logit_values),
                                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                                    values=_dropout(coefs.values, coef_drop),
                                    dense_shape=coefs.dense_shape)
        seq_fts = _dropout(seq_fts, in_drop)

        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        bias = tf.Variable(tf.zeros([int(vals.shape[-1])]), name='bias')
        ret = vals + bias

        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + _conv1d(seq, int(ret.shape[-1]), use_bias=True)
            else:
                ret = ret + seq

        return activation(ret)
