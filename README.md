# SubgraphGNN

A graph neural network for counting cellular community motifs in spatial tissue graphs — built as a high-accuracy replacement for the subgraph counting component of [TrimNN](https://github.com/yuyang-0825/TrimNN) (Wang et al., 2024).

---

## Results on real intestinal tissue (B004 ascending colon, 21,232 cells)

| Metric | TrimNN | SubgraphGNN | Improvement |
|--------|--------|-------------|-------------|
| Size-3 Spearman vs VF2 ground truth | 0.456 | **0.816** | +79% |
| Size-4 binary MCC | ~0.00 | **0.659** | — |
| Calibrated RMSE vs VF2 | 1592 | **68** | −96% |
| Top-10 pattern overlap | — | **100%** | — |

Evaluated on 2,925 patterns × 1,000 nodes (bot-entropy selection). VF2 exact subgraph isomorphism used as ground truth.

Multi-sample generalization (10 held-out intestinal samples, avg Spearman): **0.751**

---

## Background

[TrimNN](https://github.com/yuyang-0825/TrimNN) by Juexin Wang et al. detects over-represented cellular community (CC) motifs in spatial omics data by combining Delaunay triangulation with a neural network that predicts motif occurrence counts. It is a compelling approach for discovering spatial cell organization patterns in tissues.

The subgraph counting step — predicting how many times a given motif pattern appears across a tissue graph — is the core inference component. SubgraphGNN replaces this step with a GNN trained jointly on synthetic and real intestinal data, achieving substantially better rank correlation with exact VF2 counts.

---

## Architecture

`SubgraphGNN` (`train_trimnn.py`) is a 3-layer message-passing GNN with:

- **Bidirectional cross-attention** between pattern and graph subgraph embeddings
- **Gated pooling** for permutation-invariant aggregation
- **Size embedding** for multi-size pattern generalization (sizes 3–9)
- **Soft-BCE loss** on rank-normalized intestinal VF2 targets — the key breakthrough over MSE
- Jointly trained on 89K synthetic demo samples + 124 real intestinal samples

Model size: 677K parameters (`hidden=192`, 3 layers).

```
Pattern graph  ──► 3-layer GNN ──► cross-attn ──┐
                                                  ├──► gated pool ──► sigmoid score
Target subgraph ──► 3-layer GNN ──► cross-attn ──┘
```

---

## Key finding: soft-BCE loss

The pivotal discovery was replacing MSE regression with soft binary cross-entropy on rank-normalized targets:

```python
# Rank-normalized target: (i+1)/n_nonzero for nonzero patterns, 0 for zero
int_loss = F.binary_cross_entropy(int_pred, rank_targets.clamp(0.0, 1.0))
```

This single change simultaneously improved RMSE by 59%, MCC by 89%, and Spearman by 7pp over the MSE baseline.

---

## Files

| File | Description |
|------|-------------|
| `train_trimnn.py` | Model definition + training loop |
| `eval_trimnn_benchmark.py` | Full benchmark vs TrimNN on B004_ascending |
| `eval_multisample.py` | Cross-sample generalization eval (10 intestinal samples) |
| `eval_allsizes.py` | Demo graph eval across sizes 3–6 |
| `eval_crosssample.py` | Cross-sample Spearman analysis |
| `eval_ensemble_benchmark.py` | Ensemble model evaluation |
| `gen_vf2_*.py` | VF2 ground truth generation scripts |
| `trimnn_model.pt` | Best trained weights (exp63, hidden=192, seed=42) |
| `trimnn_benchmark_results.txt` | Detailed benchmark output |
| `results.tsv` | Full experiment log (75+ experiments) |
| `run.log` / `run_exp*.log` | Training run logs |

---

## Usage

```bash
# Train from scratch
python3 train_trimnn.py

# Benchmark against TrimNN on B004_ascending
python3 eval_trimnn_benchmark.py --model trimnn_model.pt

# Multi-sample generalization eval
python3 eval_multisample.py --model trimnn_model.pt

# Demo graph across sizes 3-6
python3 eval_allsizes.py --model trimnn_model.pt --sizes 3,4,5,6
```

Requires: `torch`, `igraph`, `numpy`, `scipy`. No GPU required (CPU inference ~35 min for full benchmark).

---

## Experiment log

75+ experiments tracked in `results.tsv`. Key milestones:

- **Baseline**: MSE loss — Spearman 0.468 vs VF2
- **Soft-BCE breakthrough**: Spearman jumps to 0.81+, MCC 0.65+
- **Architecture search**: hidden=192, 3-layer GNN, dropout=0 confirmed optimal
- **Eval optimization**: bot-1000 entropy node selection (+2.8σ over random-500)
- **Plateau confirmed**: 0.811–0.816 across 15+ experiments — ceiling appears data/architecture-limited

---

## Citation

If you use this work, please also cite the original TrimNN paper:

```bibtex
@article{wang2024trimnn,
  title={Characterizing cellular community motifs for studying multicellular
         topological organization in complex tissues},
  author={Wang, Juexin and others},
  year={2024}
}
```

---

Built on the [autoresearch](https://github.com/karpathy/autoresearch) framework by @karpathy.
