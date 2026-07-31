# autoresearch — SubgraphGNN

Autonomous research loop for improving neural subgraph matching on real intestinal tissue graphs.

## Task

Predict how many times a small graph pattern (size 3 or 4) appears as a subgraph inside a large
spatial cell-type graph. Ground truth is exact VF2 subgraph isomorphism counting. The metric is
**vf2_spearman** — the average of size-3 and size-4 Spearman rank correlations vs VF2.

**The only file you modify is `train_trimnn.py`.** Everything else is fixed.

---

## Current best result (your baseline to beat)

```
vf2_spearman:  0.7614   (avg of s3=0.8164, s4=0.7063)
s3_spearman:   0.8164   (size-3 vs VF2 ground truth, B004_ascending)
s4_spearman:   0.7063
s4_MCC:        0.6594
calib_RMSE:    68.4
multi_avg:     0.7514   (avg Spearman across 10 held-out intestinal samples)
```

These are your targets. Any experiment that doesn't improve `vf2_spearman` above 0.7614 is reverted.

---

## Current architecture (`SubgraphGNN` in `train_trimnn.py`)

```
Pattern graph  ──► 3-layer GNN ──► cross-attn ──┐
                                                  ├──► gated pool ──► sigmoid score
Target subgraph ──► 3-layer GNN ──► cross-attn ──┘
```

Key components:
- **3-layer message-passing GNN** (hidden=192) with ReLU, layer norm
- **Bidirectional cross-attention**: pattern attends to subgraph, subgraph attends to pattern
- **Gated pooling**: learned gate `σ(Wh)` weights the mean pool
- **Size embedding**: lookup by motif size (3–9), added to pattern embeddings
- **Sigmoid output**: score in [0, 1]

Current hyperparameters (optimal from 75+ experiments):
- `HIDDEN_DIM = 192`, `NUM_LAYERS = 3`, `DROPOUT = 0.0`
- `LR = 1e-3`, `WEIGHT_DECAY = 1e-5`, `BATCH_SIZE = 256`
- `MAX_GRAD_NORM = 5.0`, `WARMUP_STEPS = 50`
- `INT_LOSS_WEIGHT = 20` (intestinal loss weight vs demo)
- `INT_EVERY = 1` (intestinal batch every step)

Loss: **soft-BCE on rank-normalized targets** — this was the key breakthrough (+59% RMSE, +89% MCC).
```python
# Rank-normalized target: (i+1)/n_nonzero for nonzero, 0.0 for zero
int_loss = F.binary_cross_entropy(int_pred, rank_targets.clamp(0, 1))
```

---

## What has already been tried and confirmed as suboptimal (do NOT repeat these)

| Change | Result |
|--------|--------|
| hidden=128 | worse |
| hidden=256 | worse |
| num_layers=4 | worse |
| dropout=0.1 | worse (hurts cross-attn) |
| LR=5e-4 | worse |
| LR=2e-3 | worse |
| weight_decay=1e-3 | worse (over-regularized) |
| ILW=30, ILW=40 | worse (ILW=20 is optimal) |
| INT_EVERY=2 | worse |
| MSE loss | much worse (-59% RMSE vs soft-BCE) |
| Pairwise RankNet loss | completely flat (no gradient signal) |
| Ensemble of two models | worse than single |
| seed=7 (vs seed=42) | slightly worse |
| More training time (20000s) | same ceiling — time is not the bottleneck |

**The performance ceiling (~0.816 s3 Spearman) is architecture/data-limited, not compute-limited.**
More training steps or longer budgets won't help. New ideas are needed.

---

## Promising directions not yet tried (try these first)

1. **K_HOP=3** — currently using 2-hop subgraphs. Larger neighborhoods may capture more context.
   Risk: much slower data loading; subgraphs become large. Try on a small subset first.

2. **Structural node features** — add degree, clustering coefficient, or betweenness as extra
   node features alongside cell-type label embeddings. Currently using only cell-type labels.

3. **Direct ListNet / approx-Spearman loss** — RankNet failed (pairwise), but ListNet (listwise)
   or a differentiable Spearman approximation may work better than soft-BCE.

4. **Larger architecture with GPU** — now running on T4. hidden=256 with num_layers=4 was tried
   on CPU (slow, underfits). With GPU it may now converge properly. Try hidden=256, layers=4.

5. **Pattern edge features** — currently edge labels are all 0. The topology carries structure
   (kite vs triangle) but encoding edge positions explicitly might help size generalization.

6. **Contrastive training** — pair similar patterns (same label permutation, different size) and
   push their scores to be similar. Could help multi-size generalization.

7. **Two-stage model** — stage 1: binary classifier (zero vs nonzero); stage 2: regressor for
   nonzero counts. The zero/nonzero imbalance (~80% zeros) may be hurting the regressor.

8. **Graph-level features** — add global stats of the target subgraph (mean degree, label
   entropy) as extra context to the cross-attention.

---

## Data

- **Demo graph**: 743 nodes, 8 cell types, 89K (pattern, subgraph, VF2-count) training triples.
  Used as regularization. Loaded from `../TrimNN/demo_data/`.
- **Intestinal graphs**: 124 training samples (B004–B012 donors, 8 gut regions, ~5K–33K nodes,
  22–25 cell types). Held out: B005 donor (10 samples). Loaded from `../TrimNN/intestinalOutputs/`.
- Evaluation: B004_ascending (21,232 nodes, 2,925 size-3 patterns, 600 size-4 patterns),
  scored on 1,000 lowest-entropy nodes.

---

## Output format (what the research loop reads)

Training must print this block to stdout at the end:
```
vf2_spearman:          0.XXXXXX
s3_spearman:           0.XXXXXX
s4_spearman:           0.XXXXXX
s4_mcc:                0.XXXXXX
calib_rmse:            XX.XXXX
training_seconds:      300.1
num_steps:             6000
num_params_M:          0.68
hidden_dim:            192
num_layers:            3
```

The loop extracts `vf2_spearman` with:
```
grep "^vf2_spearman:" run.log
```

**Do not change this output format.** The research loop depends on it.

---

## Rules

- **Modify only `train_trimnn.py`**. Do not touch eval scripts, data loaders, or VF2 files.
- Each experiment runs for `_TIME_BUDGET = 300` seconds (5 minutes). Do not increase this.
- The cosine LR schedule `total` is currently calibrated for ~6000 steps on T4. Adjust if
  you change the architecture significantly (different step time → different step count).
- If a run crashes with no `vf2_spearman` in the log: revert, diagnose from `run.log`, fix.
- **NEVER STOP.** Run until the human interrupts. You are an autonomous researcher.
