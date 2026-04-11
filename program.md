# autoresearch — TrimNN edition

Autonomous research loop for improving neural subgraph matching on spatial cell-type graphs.

## Background

TrimNN predicts how many times a small graph pattern (e.g. a triangle of cell types A-B-C)
appears as a subgraph inside a large spatial transcriptomics graph. The ground truth is
exact VF2 subgraph isomorphism counting. The goal is to train a GNN that approximates
this counting function as accurately as possible.

**Data**: `demo_data.gml` — 743 nodes (cells), 2211 edges, 8 cell types.
Each training sample is a (pattern, k-hop subgraph, VF2 count) triple.
There are 120 non-isomorphic triangle patterns × 743 nodes ≈ 89,160 samples total.

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

**The goal: get the lowest `val_mse`** (mean squared error between predicted and
VF2-exact counts). Lower is better. The time budget is fixed, so improvements come
entirely from better architecture or optimization.

**Hints**:
- Most samples have count = 0 (sparse). Consider loss weighting or a two-stage model.
- The input distribution is (120 patterns) × (743 k-hop subgraphs). Pattern identity
  matters — the model must distinguish cell-type label combinations.
- The GNN must compare pattern structure against graph structure to predict matches.
- Cross-graph attention (pattern queries graph) is a natural inductive bias here.
- Log-transforming counts (log1p / expm1) often helps with skewed count distributions.

**Simplicity criterion**: a small improvement from simpler code is better than
a large improvement from fragile complexity.

## Output format

```
---
val_mse:          0.123456
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
grep "^val_mse:" run.log
```

## Logging results

Track in `results.tsv` (tab-separated, untracked by git):

```
commit	val_mse	status	description
```

## Experiment loop

LOOP FOREVER:

1. Check git state (branch, last commit)
2. Tune `train.py` with an experimental idea
3. `git commit`
4. `python train.py > run.log 2>&1`
5. `grep "^val_mse:" run.log`
6. If empty → crash. Run `tail -50 run.log` to diagnose. Fix if trivial, else skip.
7. Log to `results.tsv`
8. If `val_mse` improved (lower) → keep the commit
9. If `val_mse` is equal or worse → `git reset --hard HEAD~1`

**NEVER STOP.** Once started, run until the human interrupts you. Do not ask for permission
to continue. You are an autonomous researcher.
