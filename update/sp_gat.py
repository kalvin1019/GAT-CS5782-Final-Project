import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN


class SpGAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat, hid_units, n_heads,
                  activation=tf.nn.elu, residual=False,
                  phi_indices=None, phi_values=None):
        """
        Sparse forward pass with optional structural phi features.

        Args:
            phi_indices: [E, 2] edge indices (must match bias_mat sparsity)
            phi_values:  [E, K] phi feature values per edge
        """
        attns = []
        for _ in range(n_heads[0]):
            attns.append(layers.sp_attn_head(
                inputs, adj_mat=bias_mat,
                phi_indices=phi_indices, phi_values=phi_values,
                out_sz=hid_units[0], activation=activation, nb_nodes=nb_nodes,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1)

        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.sp_attn_head(
                    h_1, adj_mat=bias_mat,
                    phi_indices=phi_indices, phi_values=phi_values,
                    out_sz=hid_units[i], activation=activation, nb_nodes=nb_nodes,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)

        out = []
        for i in range(n_heads[-1]):
            out.append(layers.sp_attn_head(
                h_1, adj_mat=bias_mat,
                phi_indices=phi_indices, phi_values=phi_values,
                out_sz=nb_classes, activation=lambda x: x, nb_nodes=nb_nodes,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]

        return logits
