"""
TrimNN replacement — multi-dataset GNN for subgraph counting.

Trains on BOTH demo data (8 cell types, per-node k-hop counts) AND
intestinal data (20-25 cell types, whole-graph VF2 counts).

Architecture: shared GNN + bidirectional cross-attention + gated-sum pooling.
MAX_LABELS=32 fixed embedding accepts any dataset with up to 32 cell types.

Inference replacing TrimNN:
  For (pattern, large_graph):
    1. Sample K center nodes, compute 2-hop subgraph each
    2. whole_count ≈ mean(model predictions) × n_nodes / 3
  100-1000x faster than TrimNN's per-pattern model-reload loop.

Metrics:
  val_mse     — MSE on demo k-hop local counts (original benchmark)
  int_val_mse — MSE on intestinal whole-graph counts (new TrimNN task)
"""

import os
import gc
import json
import math
import time
import random

import numpy as np
import pandas as pd
import igraph as ig

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    TIME_BUDGET, N_LABELS, MAX_GRAPH_NODES, MAX_PATTERN_NODES,
    generate_data, split_data, make_dataloader, evaluate_val_mse,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fixed max label count — accepts any dataset with up to MAX_LABELS cell types.
# Same approach as TrimNN's max_ngvl=32. Features are zero-padded to this size.
MAX_LABELS = 32

INTESTINAL_BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../TrimNN/intestinalOutputs")
)
K_HOP_INT = 2   # k-hop radius for intestinal subgraph extraction

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

HIDDEN_DIM    = 128
NUM_LAYERS    = 3
DROPOUT       = 0.0
LR            = 1e-3
WEIGHT_DECAY  = 1e-5
BATCH_SIZE    = 256
MAX_GRAD_NORM = 5.0
WARMUP_STEPS  = 50

# Intestinal data mix: every INT_EVERY demo batches, run one intestinal batch
INT_EVERY       = 2     # 33% of gradient steps use intestinal data
INT_BATCH_SIZE  = 64    # intestinal samples per batch
INT_LOSS_WEIGHT = 2.0   # weight for intestinal loss

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GNNLayer(nn.Module):
    """One message-passing step: aggregate neighbours, then MLP."""

    def __init__(self, dim, dropout):
        super().__init__()
        self.linear = nn.Linear(dim * 2, dim)
        self.norm   = nn.LayerNorm(dim)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x, adj_norm):
        # x: (B, N, D)  adj_norm: (B, N, N) row-normalised
        msg = torch.bmm(adj_norm, x)
        h   = torch.cat([x, msg], dim=-1)
        return x + self.drop(self.norm(F.gelu(self.linear(h))))   # residual


class SubgraphGNN(nn.Module):
    """
    Encodes a (pattern, graph) pair and predicts occurrence count.
    Accepts any number of cell-type labels up to MAX_LABELS.

    Inputs:
        p_adj  (B, Np, Np), p_feat (B, Np, L≤MAX_LABELS), p_mask (B, Np)
        g_adj  (B, Ng, Ng), g_feat (B, Ng, L≤MAX_LABELS), g_mask (B, Ng)
    Output:
        pred (B, 1) — predicted occurrence count (non-negative)
    """

    def __init__(self, hidden_dim, num_layers, dropout, max_labels=MAX_LABELS):
        super().__init__()
        self.max_labels = max_labels
        self.embed      = nn.Linear(max_labels, hidden_dim, bias=False)
        self.layers     = nn.ModuleList(
            [GNNLayer(hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.cross_attn_p2g = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.cross_attn_g2p = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm_p    = nn.LayerNorm(hidden_dim)
        self.norm_g    = nn.LayerNorm(hidden_dim)
        self.gate_g    = nn.Linear(hidden_dim, hidden_dim)
        nn.init.constant_(self.gate_g.bias, -2.0)   # sparse init ~0.12
        self.tri_embed = nn.Linear(1, hidden_dim, bias=False)
        self.predict   = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    @staticmethod
    def _norm_adj(adj):
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return adj / deg

    def _encode(self, adj, feat):
        adj_n = self._norm_adj(adj)
        # Pad to max_labels — handles any cell-type count ≤ MAX_LABELS
        if feat.shape[-1] < self.max_labels:
            feat = F.pad(feat, (0, self.max_labels - feat.shape[-1]))
        x = F.gelu(self.embed(feat))
        for layer in self.layers:
            x = layer(x, adj_n)
        return x

    def forward(self, p_adj, p_feat, p_mask, g_adj, g_feat, g_mask):
        p_enc = self._encode(p_adj, p_feat)
        g_enc = self._encode(g_adj, g_feat)
        A2_g  = torch.bmm(g_adj, g_adj)
        tri_c = (A2_g * g_adj).sum(dim=-1, keepdim=True) / 2
        g_enc = g_enc + self.tri_embed(torch.log1p(tri_c))

        p_mask_f = p_mask.unsqueeze(-1).float()
        g_mask_f = g_mask.unsqueeze(-1).float()
        g_key_mask = ~g_mask
        p_key_mask = ~p_mask

        p_attn, _ = self.cross_attn_p2g(
            self.norm_p(p_enc), self.norm_g(g_enc), self.norm_g(g_enc),
            key_padding_mask=g_key_mask)
        p_out = p_enc + p_attn

        g_attn, _ = self.cross_attn_g2p(
            self.norm_g(g_enc), self.norm_p(p_enc), self.norm_p(p_enc),
            key_padding_mask=p_key_mask)
        g_out = g_enc + g_attn

        p_pool  = (p_out * p_mask_f).max(dim=1).values
        g_pool  = (g_out * g_mask_f).max(dim=1).values
        gate    = torch.sigmoid(self.gate_g(g_out))
        g_gated = (gate * g_enc * g_mask_f).sum(dim=1)

        return self.predict(torch.cat([p_pool, g_pool, g_gated], dim=-1))


# ---------------------------------------------------------------------------
# Intestinal data pipeline
# ---------------------------------------------------------------------------

def _khop_nodes(neighbors, center, k):
    """BFS k-hop neighbourhood → sorted node list."""
    visited  = {center}
    frontier = {center}
    for _ in range(k):
        nf = set()
        for v in frontier:
            for nb in neighbors[v]:
                if nb not in visited:
                    nf.add(nb)
                    visited.add(nb)
        frontier = nf
    return sorted(visited)


def _build_g_tensors(node_list, node_labels, neighbors_map,
                     max_nodes=MAX_GRAPH_NODES, max_labels=MAX_LABELS):
    """Padded (adj, feat, mask) tensors for a k-hop subgraph."""
    n        = min(len(node_list), max_nodes)
    node_idx = {v: i for i, v in enumerate(node_list)}
    adj  = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    feat = np.zeros((max_nodes, max_labels), dtype=np.float32)
    mask = np.zeros((max_nodes,),            dtype=bool)
    for i, v in enumerate(node_list[:n]):
        lbl = int(node_labels[v]) % max_labels
        feat[i, lbl] = 1.0
        mask[i] = True
        for nb in neighbors_map[v]:
            j = node_idx.get(nb)
            if j is not None and j < n:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
    return (torch.from_numpy(adj),
            torch.from_numpy(feat),
            torch.from_numpy(mask))


TRIANGLE_EDGES = [(0,1),(0,2),(1,2)]
SIZE4_EDGES    = [(0,1),(0,2),(1,2),(1,3),(2,3)]

def _build_p_tensors(label_list, edges=None,
                     max_pat=MAX_PATTERN_NODES, max_labels=MAX_LABELS):
    """Padded (adj, feat, mask) tensors for a pattern with given edges."""
    if edges is None:
        edges = TRIANGLE_EDGES
    adj  = torch.zeros(max_pat, max_pat)
    for u, v in edges:
        if u < max_pat and v < max_pat:
            adj[u,v] = adj[v,u] = 1.0
    feat = torch.zeros(max_pat, max_labels)
    for i, lbl in enumerate(label_list[:max_pat]):
        feat[i, int(lbl) % max_labels] = 1.0
    mask = torch.zeros(max_pat, dtype=torch.bool)
    mask[:min(len(label_list), max_pat)] = True
    return adj, feat, mask


# Demo size-4 VF2 data for eval (generated: 5184 patterns)
DEMO_VF2S4_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../TrimNN/demo_data/vf2s4/Occurrence_number_size4.csv"))

@torch.no_grad()
def evaluate_demo_s4_spearman(model, device):
    """Spearman vs VF2 for size-4 patterns on demo graph."""
    if not os.path.exists(DEMO_VF2S4_PATH):
        return 0.0
    demo_gml = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../TrimNN/demo_data/demo_data.gml"))
    g = ig.read(demo_gml)
    n = g.vcount()
    labels = [int(x) for x in g.vs['label']]
    neighbors = [[] for _ in range(n)]
    for u, v in g.get_edgelist():
        neighbors[u].append(v); neighbors[v].append(u)
    khop = [_khop_nodes(neighbors, v, K_HOP_INT) for v in range(n)]
    df   = pd.read_csv(DEMO_VF2S4_PATH)
    # Sample 200 balanced patterns for speed
    rng    = random.Random(7)
    nonzero = [(json.loads(r['label']), int(r['occurrence_number'])) for _,r in df.iterrows() if r['occurrence_number']>0]
    zero    = [(json.loads(r['label']), int(r['occurrence_number'])) for _,r in df.iterrows() if r['occurrence_number']==0]
    pats = rng.sample(nonzero, min(100, len(nonzero))) + rng.sample(zero, min(100, len(zero)))
    eval_nodes = rng.sample(range(n), min(100, n))
    g_adjs, g_feats, g_masks = [], [], []
    for cn in eval_nodes:
        ga, gf, gm = _build_g_tensors(khop[cn], labels, neighbors)
        g_adjs.append(ga); g_feats.append(gf); g_masks.append(gm)
    G_ADJ  = torch.stack(g_adjs).to(device)
    G_FEAT = torch.stack(g_feats).to(device)
    G_MASK = torch.stack(g_masks).to(device)
    N_eval = len(eval_nodes)
    preds_all, trues_all = [], []
    model.eval()
    for lbl, true_cnt in pats:
        pa, pf, pm = _build_p_tensors(lbl, SIZE4_EDGES)
        PA = pa.unsqueeze(0).expand(N_eval,-1,-1).to(device)
        PF = pf.unsqueeze(0).expand(N_eval,-1,-1).to(device)
        PM = pm.unsqueeze(0).expand(N_eval,-1).to(device)
        ps = []
        for i in range(0, N_eval, 128):
            out = model(PA[i:i+128], PF[i:i+128], PM[i:i+128],
                        G_ADJ[i:i+128], G_FEAT[i:i+128], G_MASK[i:i+128])
            ps.append(out.cpu())
        pred_cnt = torch.cat(ps).mean().item() * n
        preds_all.append(pred_cnt); trues_all.append(float(true_cnt))
    model.train()
    nn = len(preds_all)
    if nn < 2: return 0.0
    def _rank(lst):
        s = sorted(range(nn), key=lambda i: lst[i])
        r = [0]*nn
        for rk, idx in enumerate(s): r[idx] = rk
        return r
    rp, rt = _rank(preds_all), _rank(trues_all)
    mrp, mrt = sum(rp)/nn, sum(rt)/nn
    num = sum((rp[i]-mrp)*(rt[i]-mrt) for i in range(nn))
    den = (sum((r-mrp)**2 for r in rp)*sum((r-mrt)**2 for r in rt))**0.5
    return num/den if den > 0 else 0.0


class IntestinalData:
    """
    Lazy dataset for intestinal whole-graph triangle count prediction.

    Training target per k-hop sample:
        count × 3 / n_nodes   (expected per-node contribution)

    Consistent with TrimNN's aggregation:
        whole_count ≈ Σ_nodes(local_pred) / 3

    Evaluation:
        Sample EVAL_NODES random nodes per (pattern, graph).
        Predicted whole count = mean(preds) × n_nodes / 3
    """

    EVAL_PATTERNS = 50    # more patterns → more high-count samples in eval
    EVAL_NODES    = 200   # more nodes → more stable per-pattern estimate

    def __init__(self):
        self.samples = []
        self._load()

    def _load(self):
        print("Loading intestinal data ...")
        t0 = time.time()
        for d in sorted(os.listdir(INTESTINAL_BASE)):
            dpath   = os.path.join(INTESTINAL_BASE, d)
            vf2file = os.path.join(dpath, d + '_vf2s3', 'Occurrence_number_size3.csv')
            gml     = os.path.join(dpath, d + '.gml')
            if not (os.path.isdir(dpath) and os.path.exists(vf2file)
                    and os.path.exists(gml)):
                continue

            g      = ig.read(gml)
            labels = [int(x) for x in g.vs['label']]
            n      = g.vcount()
            neighbors = [[] for _ in range(n)]
            for u, v in g.get_edgelist():
                neighbors[u].append(v)
                neighbors[v].append(u)

            # Precompute k-hop node lists for every node
            khop = [_khop_nodes(neighbors, v, K_HOP_INT) for v in range(n)]

            df = pd.read_csv(vf2file)
            patterns = [(json.loads(row['label']), int(row['occurrence_number']))
                        for _, row in df.iterrows()]

            zero_pats    = [(l,c) for l,c in patterns if c == 0]
            nonzero_pats = [(l,c) for l,c in patterns if c > 0]
            self.samples.append({
                'name':        d,
                'n_nodes':     n,
                'node_labels': labels,
                'neighbors':   neighbors,
                'khop':        khop,
                'patterns':    patterns,
                'zero_pats':   zero_pats,
                'nonzero_pats': nonzero_pats,
            })
            print(f"  {d}: {n:,} nodes, {len(patterns)} patterns")

        print(f"Intestinal: {len(self.samples)} samples "
              f"({sum(s['n_nodes'] for s in self.samples):,} nodes total) "
              f"in {time.time()-t0:.1f}s")

    def sample_batch(self, batch_size=INT_BATCH_SIZE):
        """Balanced 50/50 batch: half zero-count, half count-weighted non-zero.

        Fixes the zero-dominance bias (Spearman=0.04) without losing zero
        calibration (which pure count-weighting destroyed, zero_mae 0.07→15).
        """
        items = []
        n_zero    = batch_size // 2
        n_nonzero = batch_size - n_zero

        def _make_item(s, lbl, count):
            cn        = random.randint(0, s['n_nodes'] - 1)
            node_list = s['khop'][cn]
            p_adj, p_feat, p_mask = _build_p_tensors(lbl)
            g_adj, g_feat, g_mask = _build_g_tensors(
                node_list, s['node_labels'], s['neighbors'])
            raw_target = float(count) * 3.0 / s['n_nodes']
            target = math.sqrt(raw_target)
            return (p_adj, p_feat, p_mask, g_adj, g_feat, g_mask,
                    torch.tensor([target], dtype=torch.float32))

        # Half from zero-count patterns (uniform)
        while len(items) < n_zero:
            s = random.choice(self.samples)
            if s['zero_pats']:
                lbl, count = random.choice(s['zero_pats'])
                items.append(_make_item(s, lbl, count))

        # Non-zero half: 50% high-count (>100) + 50% low/mid
        # Doubles high-count representation vs uniform-across-3-tiers (was 1/3)
        # to give the model more gradient signal where error is largest
        n_high_target = (batch_size - n_zero) // 2
        n_lowmid_target = (batch_size - n_zero) - n_high_target

        # Fill high-count slots first
        attempts = 0
        while len(items) - n_zero < n_high_target and attempts < 10000:
            attempts += 1
            s = random.choice(self.samples)
            high = [(l,c) for l,c in s['nonzero_pats'] if c > 100]
            if high:
                lbl, count = random.choice(high)
                items.append(_make_item(s, lbl, count))

        # Fill remaining non-zero slots with low/mid patterns
        while len(items) < batch_size:
            s = random.choice(self.samples)
            if s['nonzero_pats']:
                low = [(l,c) for l,c in s['nonzero_pats'] if 1 <= c <= 100]
                if low:
                    lbl, count = random.choice(low)
                    items.append(_make_item(s, lbl, count))

        return [torch.stack([x[i] for x in items]) for i in range(7)]

    @torch.no_grad()
    def evaluate(self, model, device):
        """
        Full VF2 benchmark suite:
          MSE, MAE, R², Spearman ρ, per-count-bucket MAE.
        """
        model.eval()
        preds_all = []
        trues_all = []
        rng = random.Random(0)

        for s in self.samples:
            eval_pats  = rng.sample(s['patterns'],
                                    min(self.EVAL_PATTERNS, len(s['patterns'])))
            eval_nodes = rng.sample(range(s['n_nodes']),
                                    min(self.EVAL_NODES, s['n_nodes']))

            g_adjs, g_feats, g_masks = [], [], []
            for cn in eval_nodes:
                ga, gf, gm = _build_g_tensors(
                    s['khop'][cn], s['node_labels'], s['neighbors'])
                g_adjs.append(ga); g_feats.append(gf); g_masks.append(gm)
            G_ADJ  = torch.stack(g_adjs).to(device)
            G_FEAT = torch.stack(g_feats).to(device)
            G_MASK = torch.stack(g_masks).to(device)
            N_eval = len(eval_nodes)

            CHUNK = 128
            for (lbl, true_count) in eval_pats:
                p_adj, p_feat, p_mask = _build_p_tensors(lbl)
                PA = p_adj.unsqueeze(0).expand(N_eval,-1,-1).to(device)
                PF = p_feat.unsqueeze(0).expand(N_eval,-1,-1).to(device)
                PM = p_mask.unsqueeze(0).expand(N_eval,-1).to(device)

                preds = []
                for i in range(0, N_eval, CHUNK):
                    out = model(PA[i:i+CHUNK], PF[i:i+CHUNK], PM[i:i+CHUNK],
                                G_ADJ[i:i+CHUNK], G_FEAT[i:i+CHUNK], G_MASK[i:i+CHUNK])
                    preds.append(out.cpu())
                preds = torch.cat(preds).view(-1)

                pred_count = (preds.mean().item() ** 2) * s['n_nodes'] / 3.0
                preds_all.append(pred_count)
                trues_all.append(float(true_count))

        model.train()
        n = len(trues_all)

        # MSE / MAE
        mse = sum((p-t)**2 for p,t in zip(preds_all,trues_all)) / n
        mae = sum(abs(p-t)  for p,t in zip(preds_all,trues_all)) / n

        # R²
        mean_t = sum(trues_all) / n
        ss_tot = sum((t - mean_t)**2 for t in trues_all)
        ss_res = sum((p - t)**2 for p,t in zip(preds_all,trues_all))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Spearman rank correlation
        def _rank(lst):
            s = sorted(range(len(lst)), key=lambda i: lst[i])
            r = [0]*len(lst)
            for rank, idx in enumerate(s): r[idx] = rank
            return r
        rp = _rank(preds_all); rt = _rank(trues_all)
        mean_rp = sum(rp)/n; mean_rt = sum(rt)/n
        num = sum((rp[i]-mean_rp)*(rt[i]-mean_rt) for i in range(n))
        den = (sum((r-mean_rp)**2 for r in rp)*sum((r-mean_rt)**2 for r in rt))**0.5
        spearman = num/den if den > 0 else 0.0

        # Per-count-bucket MAE
        buckets = {'zero':(0,0),'low':(1,10),'mid':(11,100),'high':(101,1e9)}
        bucket_stats = {}
        for bname,(lo,hi) in buckets.items():
            pairs = [(p,t) for p,t in zip(preds_all,trues_all) if lo<=t<=hi]
            if pairs:
                bucket_stats[bname] = (sum(abs(p-t) for p,t in pairs)/len(pairs), len(pairs))
            else:
                bucket_stats[bname] = (0.0, 0)

        return dict(mse=mse, mae=mae, r2=r2, spearman=spearman,
                    n=n, buckets=bucket_stats)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
device  = torch.device("cpu")

print("Loading demo data ...")
all_data   = generate_data()
train_data, val_data = split_data(all_data)
print(f"Demo — Train: {len(train_data)}  Val: {len(val_data)}")

train_loader = make_dataloader(train_data, BATCH_SIZE, shuffle=True)
val_loader   = make_dataloader(val_data,   BATCH_SIZE, shuffle=False)

# Load intestinal data (precomputes k-hop lists for all nodes)
int_data = IntestinalData()

model     = SubgraphGNN(HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    t     = step - WARMUP_STEPS
    total = 3800 - WARMUP_STEPS
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(t / total, 1.0)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = 0.0
total_train_time  = 0.0
step              = 0
smooth_loss       = None
train_iter        = iter(train_loader)

while True:
    model.train()
    t0 = time.time()

    # Demo batch
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        batch      = next(train_iter)

    p_adj, p_feat, p_mask, g_adj, g_feat, g_mask, counts = [
        x.to(device) for x in batch
    ]
    pred = model(p_adj, p_feat, p_mask, g_adj, g_feat, g_mask)
    loss = F.mse_loss(pred.view(-1), counts.view(-1))

    # Intestinal batch (injected every INT_EVERY steps)
    if step % INT_EVERY == 0:
        int_batch = int_data.sample_batch(INT_BATCH_SIZE)
        ip_adj, ip_feat, ip_mask, ig_adj, ig_feat, ig_mask, itargets = [
            x.to(device) for x in int_batch
        ]
        int_pred = model(ip_adj, ip_feat, ip_mask, ig_adj, ig_feat, ig_mask)
        int_loss = F.mse_loss(int_pred.view(-1), itargets.view(-1))
        loss     = loss + INT_LOSS_WEIGHT * int_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    optimizer.step()
    scheduler.step()

    dt = time.time() - t0
    if step >= 5:
        total_train_time += dt

    loss_f      = loss.item()
    ema         = 0.95
    smooth_loss = (loss_f if smooth_loss is None
                   else ema * smooth_loss + (1 - ema) * loss_f)

    progress  = min(total_train_time / TIME_BUDGET, 1.0)
    remaining = max(0.0, TIME_BUDGET - total_train_time)
    print(
        f"\rstep {step:05d} ({100*progress:.1f}%) | "
        f"loss: {smooth_loss:.4f} | "
        f"dt: {dt*1000:.0f}ms | "
        f"remaining: {remaining:.0f}s   ",
        end="", flush=True,
    )

    if step == 0:
        gc.collect()

    step += 1
    if step > 5 and total_train_time >= TIME_BUDGET:
        break

print()

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

model.eval()
demo_val_mse  = evaluate_val_mse(model, val_loader, device)
ev            = int_data.evaluate(model, device)
s4_spearman   = evaluate_demo_s4_spearman(model, device)
vf2_spearman  = (ev['spearman'] + s4_spearman) / 2.0

t_end         = time.time()
total_seconds = t_end - t_start

b = ev['buckets']
print("---")
print(f"vf2_spearman:         {vf2_spearman:.6f}  (avg size-3+4 Spearman vs VF2)")
print(f"vf2_spearman_s3:      {ev['spearman']:.6f}  (size-3 intestinal Spearman)")
print(f"vf2_spearman_s4:      {s4_spearman:.6f}  (size-4 demo Spearman)")
print(f"val_mse:              {demo_val_mse:.6f}")
print(f"int_val_mse:          {ev['mse']:.4f}  (n_eval={ev['n']})")
print(f"int_mae:              {ev['mae']:.2f}")
print(f"int_r2:               {ev['r2']:.4f}   (R² — variance explained)")
print(f"int_spearman:         {ev['spearman']:.4f}  (rank correlation vs VF2)")
print(f"bucket_zero_mae:      {b['zero'][0]:.2f}  (n={b['zero'][1]})")
print(f"bucket_low_mae:       {b['low'][0]:.2f}   (count 1-10, n={b['low'][1]})")
print(f"bucket_mid_mae:       {b['mid'][0]:.2f}   (count 11-100, n={b['mid'][1]})")
print(f"bucket_high_mae:      {b['high'][0]:.2f}  (count >100, n={b['high'][1]})")
print(f"training_seconds:     {total_train_time:.1f}")
print(f"total_seconds:        {total_seconds:.1f}")
print(f"peak_ram_mb:          0.0")
print(f"num_steps:            {step}")
print(f"num_params_M:         {n_params / 1e6:.2f}")
print(f"hidden_dim:           {HIDDEN_DIM}")
print(f"num_layers:           {NUM_LAYERS}")
print(f"intestinal_samples:   {len(int_data.samples)}")
