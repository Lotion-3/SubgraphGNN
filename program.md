# autoresearch — TrimNN edition

Autonomous research loop for improving neural subgraph matching on spatial cell-type graphs.

## Background

TrimNN predicts how many times a small graph pattern appears as a subgraph inside a large
spatial transcriptomics graph. The ground truth is exact VF2 subgraph isomorphism counting.
The goal is to train a GNN that approximates this counting function as accurately as possible.

Two pattern sizes are supported:
- **Size-3** (triangle): 3 nodes, 3 edges — 120 non-isomorphic labeled patterns (8 cell types).
- **Size-4** (kite): 4 nodes, 5 edges (0--1, 0--2, 1--2, 1--3, 2--3) — the only motif type
  in the size-4 VF2 CSVs.

**Data**: intestinal samples (up to ~40K nodes, 25 cell types, 124 samples for size-3 / 94
for size-4) plus `demo_data.gml` (743 nodes, 8 cell types) used as regularization.
Each training sample is a (pattern, k-hop subgraph, rank-normalized VF2 count) triple.

## Setup

1. **Agree on a run tag** — propose a tag based on today's date (e.g. `apr11`). The branch
   `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `prepare.py` — fixed: data generation, VF2 ground truth, dataset, evaluation. Do not modify.
   - `train.py` — the file you modify: model architecture, optimizer, hyperparameters.
4. **Verify data exists**: run `python prepare.py` once if the cache is missing
   (`~/.cache/autoresearch_trimnn/data_k2.pkl`).
5. **Initialize results.tsv** with just the header row.
6. **Confirm and go.**

## Experimentation

Each experiment runs for a **fixed time budget of 5 minutes** (wall-clock training time,
excluding startup). Launch with:

```
python train.py > run.log 2>&1
```

**What you CAN do** — modify `train.py` only:
- Model architecture: GNN layers, hidden dim, aggregation, attention, skip connections
- Optimizer: Adam, SGD, Muon, learning rate, weight decay, scheduling
- Loss function: MSE, MAE, Huber, log-space, count-weighted, etc.
- Hyperparameters: batch size, dropout, gradient clipping
- Training loop: learning rate warmup/warmdown, gradient accumulation

**What you CANNOT do**:
- Modify `prepare.py` — it defines the task and the fixed evaluation
- Change the data, VF2 counts, or train/val split
- Install new packages

**The goal: maximize `vf2_spearman`** (average of size-3 and size-4 VF2 Spearman rank
correlations). Higher is better. The time budget is fixed, so improvements come entirely
from better architecture or optimization.

**Hints**:
- Most patterns have count = 0 (sparse). Consider loss weighting or a two-stage model.
- The model must handle both size-3 (triangle) and size-4 (kite) patterns simultaneously.
  Pattern identity and structure both matter — the GNN must distinguish cell-type label
  combinations AND edge structure.
- Cross-graph attention (pattern queries graph) is a natural inductive bias here.
- Rank-normalized targets (rank/n_nonzero) directly optimize Spearman correlation.
- The demo graph (743 nodes, 8 cell types) provides regularization; intestinal graphs
  (up to 40K nodes, 25 cell types) provide the main signal.

**Simplicity criterion**: a small improvement from simpler code is better than
a large improvement from fragile complexity.

## Output format

```
---
vf2_spearman:     0.XXXXXX   (average VF2 Spearman, size-3 + size-4)
val_bin_mse:      0.123456
val_bce:          0.123456
trimnn_mse:       123.4567
training_seconds: 300.1
total_seconds:    305.2
peak_ram_mb:      0.0
num_steps:        1234
num_params_M:     0.52
hidden_dim:       128
num_layers:       3
```

Extract the key metric:
```
grep "^vf2_spearman:" run.log
```

## Logging results

Track in `results.tsv` (tab-separated, untracked by git):

```
commit	vf2_spearman	status	description
```

## Experiment loop

LOOP FOREVER:

1. Check git state (branch, last commit)
2. Tune `train.py` with an experimental idea
3. `git commit`
4. `python train.py > run.log 2>&1`
5. `grep "^vf2_spearman:" run.log`
6. If empty → crash. Run `tail -50 run.log` to diagnose. Fix if trivial, else skip.
7. Log to `results.tsv`
8. If `vf2_spearman` improved (higher) → keep the commit
9. If `vf2_spearman` is equal or worse → `git reset --hard HEAD~1`

**NEVER STOP.** Once started, run until the human interrupts you. Do not ask for permission
to continue. You are an autonomous researcher.
