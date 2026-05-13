# Revisiting Graph Attention Networks:
## A Reproduction Study with Phi-Attention Scoring

This repository contains our CS4782/5782 final project on reproducing and extending the original Graph Attention Networks (GAT) paper proposed by Veličković et al.

Our work first reproduces the original GAT results on the Cora citation dataset and then explores several modifications to the attention mechanism by incorporating structural graph information into the attention score. In addition to reproducing the reported performance of the original paper, we propose a phi-attention scoring mechanism that improves node classification accuracy while maintaining computational efficiency.

The complete methodology, experimental comparisons, and analysis are summarized in our final poster and report.

---

# 1. Introduction

Attention mechanisms have become a fundamental component in modern deep learning architectures, especially after the success of the Transformer architecture introduced in *Attention Is All You Need*. Graph Attention Networks (GAT) extended attention mechanisms to graph neural networks by allowing nodes to assign different importance weights to neighboring nodes during message passing.

The original GAT paper demonstrated strong performance on benchmark citation datasets such as Cora, Citeseer, and Pubmed through masked self-attention over graph neighborhoods.

Later works such as GATv2 pointed out that the original GAT attention mechanism is relatively static and proposed dynamic attention mechanisms for improved expressiveness.

The goal of this project is:

- Reproduce the original GAT implementation and benchmark performance on the Cora dataset.
- Investigate whether explicit structural graph information can improve attention-based node classification.
- Compare several modified attention formulations against the original GAT baseline.

---

# 2. Chosen Result

The primary reproduction target is the node classification accuracy reported by the original GAT paper on the Cora citation network dataset.

The original paper reports:

- **Original GAT accuracy on Cora:**  
  `83.0 ± 0.7%`

Using the official implementation and adapting the codebase for modern TensorFlow environments, we successfully reproduced:

- **Reproduced baseline accuracy:**  
  `82.6%`

This result falls within the expected accuracy range reported in the original paper, validating the correctness of our reproduction.

We then introduced modified attention mechanisms incorporating structural features:

```math
e_{ij} = \text{LeakyReLU}
\left(
a^T [Wh_i \parallel Wh_j \parallel \phi_{ij}]
\right)
```

Our modified GAT+\(\phi\) model achieved consistently better performance than the reproduced baseline across multiple experimental runs.

---

# 3. GitHub Contents

```text
├── data/                  # Cora dataset and preprocessing files
├── models/                # GAT baseline and modified attention implementations
├── utils/                 # Graph preprocessing and helper functions
├── poster/                # Final project poster
├── pretrained/cora        # pretrained parameters from original paper
├── report/                # project report
├── results/               # results
├── README.md              # Project overview
└── update/     # our modification to original code
```

The repository includes implementations of:

- Original GAT baseline
- GAT with structural phi-attention
- Pairwise projection GAT variants
- Evaluation and visualization utilities

---

# 4. Re-implementation Details

## Baseline Reproduction

We reproduced the original Graph Attention Network (GAT) architecture for semi-supervised node classification on the Cora citation dataset.

The original implementation was based on TensorFlow 1.x, which is incompatible with modern Colab and TensorFlow environments. Therefore, parts of the original implementation were rewritten while preserving the original training setup and hyperparameters.

The baseline attention computation follows:

```math
e_{ij} =
\text{LeakyReLU}
\left(
a^T [Wh_i \parallel Wh_j]
\right)
```

The reproduction uses:

- Multi-head graph attention layers
- LeakyReLU activation
- Softmax attention normalization
- Dropout regularization
- Semi-supervised node classification on Cora

---

## Phi-Attention Modification

To incorporate structural graph information into attention computation, we introduced a structural feature vector \(\phi_{ij}\):

```math
\phi_{ij} =
\left[
\frac{\deg(j)}{\deg_{\max}},
\frac{|N_i \cap N_j|}{|N_i \cup N_j|},
\sum_{k \in N_i \cap N_j}
\frac{1}{\log(\deg(k))}
\right]
```

The three terms correspond to:

- Degree centrality
- Jaccard similarity
- Adamic-Adar similarity

The modified attention score becomes:

```math
e_{ij} =
\text{LeakyReLU}
\left(
a^T [Wh_i \parallel Wh_j \parallel \phi_{ij}]
\right)
```

The intuition behind this modification is that citation graphs such as Cora are highly homophilous, meaning connected nodes often share semantic labels and similar local graph structures.

By explicitly encoding neighborhood similarity into attention computation, the model can better identify meaningful neighboring nodes during aggregation.

---

## Pairwise Projection Variant

We also explored an alternative projection strategy by applying the projection matrix after feature concatenation:

```math
e_{ij} =
\text{LeakyReLU}
\left(
a^T (W[h_i \parallel h_j])
\right)
```

and further combined it with structural features:

```math
e_{ij} =
\text{LeakyReLU}
\left(
a^T (W[h_i \parallel h_j \parallel \phi_{ij}])
\right)
```

---

## Compared Variants

We evaluated four different architectures:

1. Original GAT baseline
2. GAT with \(\phi_{ij}\)
3. Pairwise projection GAT
4. Pairwise projection GAT with \(\phi_{ij}\)

All models were evaluated using node classification accuracy on the Cora test set.

---

# 5. Reproduction Steps

## Environment Setup

Create a Python environment and install dependencies:

```bash
```

---

## Train Original GAT

```bash

```

---

## Train Phi-Attention GAT

```bash

```


## Recommended Hardware

- NVIDIA GPU recommended
- Google Colab compatible
- TensorFlow environment required
- CPU execution is possible but slower

---

# 6. Results / Insights

Our experiments successfully reproduced the original GAT performance on the Cora benchmark.

Main findings include:

- The reproduced baseline closely matches the reported accuracy of the original paper.
- Structural graph priors improve node classification performance.
- Phi-attention provides complementary information beyond node features alone.
- The added computational complexity remains relatively small compared to the original attention computation.
- Pairwise projection variants further improve flexibility in attention modeling.

The improvements suggest that local structural similarity plays an important role in homophilous citation graphs such as Cora.

Additionally, the experiments demonstrate that lightweight modifications to graph attention mechanisms can improve representation quality without significantly increasing computational cost.

---

# 7. Conclusion

This project successfully reproduced the original Graph Attention Network results on the Cora dataset and explored several extensions to the attention mechanism.

Our experiments demonstrate that:

- the original GAT implementation is reproducible in modern environments,
- structural graph information improves attention-based aggregation,
- neighborhood similarity provides meaningful supervision for node classification,
- modified attention mechanisms can outperform the original baseline while remaining computationally efficient.

Future work may include:

- experiments on Citeseer and Pubmed,
- evaluation on heterophilous graph datasets,
- integrating dynamic attention mechanisms from GATv2,
- combining structural priors with graph transformers.

---

# 8. References

1. Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio.  
   *Graph Attention Networks.*  
   International Conference on Learning Representations (ICLR), 2018.

2. Brody, Alon, Uri Alon, and Eran Yahav.  
   *How Attentive are Graph Attention Networks?*  
   International Conference on Learning Representations (ICLR), 2022.

3. CS4782/5782 Final Project Poster:  
   *Revisiting Graph Attention Networks: A Reproduction Study with Phi-Attention Scoring.*

---

# 9. Acknowledgements

We would like to thank the instructors and teaching assistants of CS4782/5782 for their guidance, feedback, and support throughout this project.
