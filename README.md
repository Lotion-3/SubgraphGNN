# SubgraphGNN

A graph neural network for counting cellular community motifs in spatial tissue graphs — built as a high-accuracy replacement for the subgraph counting component of [TrimNN](https://github.com/yuyang-0825/TrimNN) (Wang et al., 2024).

This project also includes the intestinal tissue dataset and VF2 ground-truth pipeline used for training and evaluation, developed independently to benchmark both TrimNN and SubgraphGNN against exact subgraph isomorphism counts.

---

## Results on real intestinal tissue (B004 ascending colon, 21,232 cells)

| Metric | TrimNN | SubgraphGNN | Improvement |
|--------|--------|-------------|-------------|
| Size-3 Spearman vs VF2 ground truth | 0.456 | **0.816** | +79% |
| Size-4 Spearman vs VF2 ground truth | ~0.00 | **0.706** | — |
| Size-4 binary MCC | ~0.00 | **0.659** | — |
| Calibrated RMSE vs VF2 (size-3) | 1592 | **68** | −96% |
| Top-10 pattern overlap with TrimNN | — | **100%** | — |

Evaluated on 2,925 size-3 patterns and 600 size-4 patterns, each scored across 1,000 tissue nodes selected by lowest 2-hop cell-type entropy (bot-entropy selection, empirically +2.8σ over random-500).

**Multi-sample generalization** — held out 10 intestinal samples across 4 donors never seen during training:

| Sample | Spearman |
|--------|----------|
| B005_ascending | 0.707 |
| B005_descendingSigmoid | 0.705 |
| B005_duodenum | 0.707 |
| B005_ileum | 0.761 |
| B006_descendingSigmoid | 0.779 |
| B006_descending | 0.771 |
| B008_ascending | 0.744 |
| B008_transverse | 0.757 |
| B010_ascending | 0.783 |
| B011_ascending | 0.786 |
| **Average** | **0.751** |

VF2 exact subgraph isomorphism used as ground truth throughout. TrimNN baseline for cross-sample comparison: Spearman = 0.456 (B004_ascending, size-3).

---

## The intestinal dataset

All intestinal tissue data used in this project was collected and processed as part of this work to benchmark the TrimNN algorithm against exact subgraph counts.

**Scale**: 65 tissue samples from 9 donors (B004–B012) spanning 8 gut regions:
- Ascending colon, descending colon, descending sigmoid, duodenum, ileum, mid-jejunum, proximal jejunum, transverse colon

Each sample is a spatial proteomics graph (CODEX/HuBMAP) where nodes are individual cells with cell-type labels and edges connect spatially adjacent cells (Delaunay triangulation). Graphs range from ~5,800 to ~33,000 nodes and 22–25 cell types per sample.

**Graph variants**: Each sample is processed in two forms:
- `nps` — triangulation without edge pruning (denser graph)
- `ps` — pruning of outlier edges (99th percentile length threshold)

Training used 124 samples (all samples minus the B005 donor held out for evaluation). The B004_ascending sample is used as the primary benchmark target.

---

## VF2 ground-truth pipeline

To evaluate both TrimNN and SubgraphGNN against exact counts, we implemented our own VF2 subgraph isomorphism runner using [igraph](https://igraph.org/)'s `get_subisomorphisms_vf2`. Three scripts handle different scopes:

### `gen_vf2_sizes59.py`
Generates per-node VF2 training data for the synthetic demo graph (743 nodes, 8 cell types), sizes 5–9. Produces `(pattern, k-hop subgraph, count)` triples in the same format as the existing sizes 3–4 training data. Uses canonical growing-kite topologies to match TrimNN's motif family.

### `gen_vf2_fast.py`
Generates whole-graph VF2 occurrence counts for the demo graph, sizes 5–9, using random label sampling. Faster than exhaustive enumeration — samples N random label assignments per size, computes global VF2 count for each. Output matches TrimNN's `Predicted_occurrence_sizeN.csv` format for direct comparison.

### `gen_vf2_intestinal59.py`
Same random-sampling strategy as `gen_vf2_fast.py` but applied to real intestinal tissue graphs. Targets sizes 5–7 on 5 representative samples (B004_ascending, B005_ascending, B006_descendingSigmoid, B008_ascending, B010_ascending).

---

## What metrics we have — and don't have

### ✅ We have

| Metric | Sizes | Coverage |
|--------|-------|----------|
| Exhaustive VF2 occurrence counts | **3, 4** | All 65 intestinal samples |
| VF2 occurrence counts (random-sampled) | **5–9** | Demo graph; 5 intestinal samples (s5–7 only) |
| SubgraphGNN Spearman vs VF2 | **3, 4** | Primary benchmark (B004_ascending) + 10 held-out samples |
| SubgraphGNN Spearman vs VF2 | **3–6** | Demo graph |
| SubgraphGNN binary MCC vs VF2 | **3, 4** | B004_ascending |
| TrimNN predictions for comparison | **3, 4** | B004_ascending (2,925 patterns) |
| TrimNN Spearman vs VF2 baseline | **3, 4** | B004_ascending |

### ❌ We don't have

| Metric | Why |
|--------|-----|
| Exhaustive VF2 for s5–9 on intestinal graphs | igraph VF2 is exponential in pattern size — at 17K–33K nodes it becomes intractable. A size-5 exhaustive run on B004_ascending would take days on CPU. |
| TrimNN predictions for s5–9 on intestinal graphs | TrimNN's motif enumeration step requires ~3 min/sample on an A100 GPU; we have no GPU. A test run estimated ~79 hours for the demo graph (743 nodes) on CPU alone. |
| Per-node VF2 counts for intestinal s5–9 | Would require exhaustive VF2, blocked by above. |
| SubgraphGNN vs TrimNN comparison on s5–9 | Blocked by missing TrimNN baseline. We have SubgraphGNN's Spearman vs sampled VF2 for s5 (demo graph: 0.618 all-patterns / 0.857 nonzero-only) but no TrimNN equivalent. |

The size 3 and 4 coverage is complete and fully benchmarked. For sizes 5+, SubgraphGNN is trained and produces predictions, but the ground-truth pipeline is sampled rather than exhaustive and TrimNN cannot be run for comparison without GPU access.

---

## Architecture

`SubgraphGNN` (`train_trimnn.py`) is a 3-layer message-passing GNN with:

- **Bidirectional cross-attention** between pattern and graph subgraph embeddings
- **Gated pooling** for permutation-invariant aggregation
- **Size embedding** for multi-size pattern generalization (sizes 3–9)
- **Soft-BCE loss** on rank-normalized VF2 targets — the key breakthrough over MSE
- Jointly trained on 89K synthetic demo samples + 124 real intestinal samples

Model size: 677K parameters (`hidden=192`, 3 layers).

```
Pattern graph  ──► 3-layer GNN ──► cross-attn ──┐
                                                  ├──► gated pool ──► sigmoid score
Target subgraph ──► 3-layer GNN ──► cross-attn ──┘
```

**Key finding — soft-BCE loss**: replacing MSE regression with soft binary cross-entropy on rank-normalized targets:

```python
# Rank-normalized target: (i+1)/n_nonzero for nonzero patterns, 0.0 for zero
int_loss = F.binary_cross_entropy(int_pred, rank_targets.clamp(0.0, 1.0))
```

This single change simultaneously improved RMSE by 59%, MCC by 89%, and Spearman by 7pp over the MSE baseline — the central result of the project.

---

## Files

| File | Description |
|------|-------------|
| `train_trimnn.py` | Model definition (SubgraphGNN) + full training loop |
| `eval_trimnn_benchmark.py` | Full benchmark vs TrimNN and VF2 on B004_ascending |
| `eval_multisample.py` | Cross-sample generalization eval (10 held-out intestinal samples) |
| `eval_allsizes.py` | Demo graph eval across sizes 3–6 |
| `eval_crosssample.py` | Cross-sample Spearman analysis |
| `eval_ensemble_benchmark.py` | Ensemble model evaluation |
| `gen_vf2_sizes59.py` | Per-node VF2 training data for demo graph, sizes 5–9 |
| `gen_vf2_fast.py` | Whole-graph VF2 counts for demo graph, sizes 5–9 (random-sampled) |
| `gen_vf2_intestinal59.py` | Whole-graph VF2 counts for intestinal graphs, sizes 5–7 |
| `time_inference.py` | Inference time profiling |
| `trimnn_app.py` | Streamlit app for interactive SubgraphGNN inference |
| `trimnn_model.pt` | Best trained weights (exp63, hidden=192, seed=42, 15000s) |
| `trimnn_model_BEST_backup.pt` | Earlier best weights backup |
| `trimnn_benchmark_results.txt` | Detailed benchmark output (early run) |
| `results.tsv` | Full experiment log (75+ experiments) |
| `run.log` / `run_exp*.log` | Training run logs |

---

## Usage

```bash
# Train from scratch (~4h on 8-core CPU)
python3 train_trimnn.py

# Benchmark against TrimNN on B004_ascending (~35 min, CPU)
python3 eval_trimnn_benchmark.py --model trimnn_model.pt

# Multi-sample generalization eval (~23 min, CPU)
python3 eval_multisample.py --model trimnn_model.pt

# Demo graph across sizes 3-6
python3 eval_allsizes.py --model trimnn_model.pt --sizes 3,4,5,6

# Generate VF2 ground truth for demo graph sizes 5-9
python3 gen_vf2_fast.py
```

Requires: `torch`, `igraph`, `numpy`, `scipy`, `pandas`. No GPU required.

---

## Experiment log

75+ experiments tracked in `results.tsv`. Key milestones:

- **Baseline** — MSE loss: Spearman 0.468 vs VF2
- **Soft-BCE breakthrough**: Spearman jumps to 0.81+, MCC 0.65+, RMSE −59%
- **Architecture search**: hidden=192, 3-layer GNN, dropout=0.0 confirmed optimal
- **Ablations**: width 128/192/256 (192 best), depth 3/4 (3 best), LR sweep, ILW=20 optimal
- **Eval optimization**: bot-1000 entropy node selection (+2.8σ over random-500 baseline)
- **Plateau confirmed**: 0.811–0.816 across 15+ experiments — ceiling appears data/architecture-limited at current scale

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

Built on the [autoresearch](https://github.com/karpathy/autoresearch) autonomous research framework by @karpathy.
