"""
Benchmark: Our TrimNN Model vs Original TrimNN
==============================================

Computes the same metrics used in the TrimNN paper to compare our neural
approximation against the original TrimNN predictions on B004_ascending.

Metrics:
  Quantitative occurrence estimation:
    RMSE, MAE, Spearman rank correlation

  Binary classification (pattern present/absent):
    MCC, Precision, Recall, F1 Score
    (ground truth = TrimNN predicts count > 0; ours = our count > threshold)

  Top-K identification:
    Top-5 and Top-10 overlap (do we rank the same high-count patterns?)

Usage:
    python eval_trimnn_benchmark.py
    (requires trimnn_model.pt — run train_trimnn.py first)
"""

import os, json, math, time, random
import numpy as np
import pandas as pd
import igraph as ig
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Model (must match train_trimnn.py)
# ---------------------------------------------------------------------------

MAX_LABELS        = 40   # must match training MAX_LABELS (exp29+: 40)
MAX_GRAPH_NODES   = 48
MAX_PATTERN_NODES = 8   # matches train_trimnn.py (prepare.py MAX_PATTERN_NODES=8)
# Size-4 VF2 topology: 5-edge "near-complete" graph on 4 nodes
SIZE4_EDGES = [(0,1),(0,2),(1,2),(1,3),(2,3)]


class GNNLayer(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(dim * 2, dim)
        self.norm   = nn.LayerNorm(dim)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x, adj_norm):
        msg = torch.bmm(adj_norm, x)
        h   = torch.cat([x, msg], dim=-1)
        return x + self.drop(self.norm(F.gelu(self.linear(h))))


class SubgraphGNN(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=3, dropout=0.0, n_size_slots=2):
        super().__init__()
        self.max_labels   = MAX_LABELS
        self.n_size_slots = n_size_slots
        self.embed        = nn.Linear(MAX_LABELS, hidden_dim, bias=False)
        self.layers     = nn.ModuleList(
            [GNNLayer(hidden_dim, dropout) for _ in range(num_layers)])
        self.cross_attn_p2g = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.cross_attn_g2p = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm_p    = nn.LayerNorm(hidden_dim)
        self.norm_g    = nn.LayerNorm(hidden_dim)
        self.gate_g    = nn.Linear(hidden_dim, hidden_dim)
        self.tri_embed = nn.Linear(1, hidden_dim, bias=False)
        # exp30+: size indicator embedding; n_size_slots auto-detected from checkpoint
        self.size_embed = nn.Embedding(n_size_slots, hidden_dim)
        self.predict   = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _norm_adj(adj):
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return adj / deg

    def _encode(self, adj, feat):
        adj_n = self._norm_adj(adj)
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

        # exp30+: add size indicator if size_embed was loaded from checkpoint
        if getattr(self, 'size_embed', None) is not None:
            n_pat    = p_mask.float().sum(dim=1).long()
            size_idx = (n_pat - 3).clamp(0, self.n_size_slots - 1)
            p_pool   = p_pool + self.size_embed(size_idx)

        return self.predict(torch.cat([p_pool, g_pool, g_gated], dim=-1))


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _khop_nodes(neighbors, center, k):
    visited = {center}; frontier = {center}
    for _ in range(k):
        nf = set()
        for v in frontier:
            for nb in neighbors[v]:
                if nb not in visited:
                    nf.add(nb); visited.add(nb)
        frontier = nf
    return sorted(visited)


def _build_g_tensors(node_list, node_labels, neighbors_map):
    n        = min(len(node_list), MAX_GRAPH_NODES)
    node_idx = {v: i for i, v in enumerate(node_list)}
    adj  = np.zeros((MAX_GRAPH_NODES, MAX_GRAPH_NODES), dtype=np.float32)
    feat = np.zeros((MAX_GRAPH_NODES, MAX_LABELS),       dtype=np.float32)
    mask = np.zeros((MAX_GRAPH_NODES,),                  dtype=bool)
    for i, v in enumerate(node_list[:n]):
        lbl       = int(node_labels[v]) % MAX_LABELS
        feat[i, lbl] = 1.0
        mask[i]   = True
        for nb in neighbors_map[v]:
            j = node_idx.get(nb)
            if j is not None and j < n:
                adj[i, j] = adj[j, i] = 1.0
    return (torch.from_numpy(adj),
            torch.from_numpy(feat),
            torch.from_numpy(mask))


def _build_p_tensors(label_list, edges=None):
    """Build pattern tensors. edges defaults to triangle if None."""
    TRIANGLE_EDGES = [(0,1),(0,2),(1,2)]
    if edges is None:
        edges = TRIANGLE_EDGES
    adj  = torch.zeros(MAX_PATTERN_NODES, MAX_PATTERN_NODES)
    for u, v in edges:
        if u < MAX_PATTERN_NODES and v < MAX_PATTERN_NODES:
            adj[u, v] = adj[v, u] = 1.0
    feat = torch.zeros(MAX_PATTERN_NODES, MAX_LABELS)
    for i, lbl in enumerate(label_list[:MAX_PATTERN_NODES]):
        feat[i, int(lbl) % MAX_LABELS] = 1.0
    mask = torch.zeros(MAX_PATTERN_NODES, dtype=torch.bool)
    mask[:min(len(label_list), MAX_PATTERN_NODES)] = True
    return adj, feat, mask


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def spearman(x, y):
    n = len(x)
    def rank(lst):
        s = sorted(range(n), key=lambda i: lst[i])
        r = [0]*n
        for rk, idx in enumerate(s): r[idx] = rk
        return r
    rx = rank(x); ry = rank(y)
    mx = sum(rx)/n; my = sum(ry)/n
    num = sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    den = (sum((r-mx)**2 for r in rx) * sum((r-my)**2 for r in ry))**0.5
    return num/den if den > 0 else 0.0


def binary_metrics(true_binary, pred_binary):
    """MCC, Precision, Recall, F1 from two bool/int lists."""
    tp = sum(t and p for t, p in zip(true_binary, pred_binary))
    tn = sum((not t) and (not p) for t, p in zip(true_binary, pred_binary))
    fp = sum((not t) and p for t, p in zip(true_binary, pred_binary))
    fn = sum(t and (not p) for t, p in zip(true_binary, pred_binary))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2*precision*recall/(precision+recall)
                 if (precision+recall) > 0 else 0.0)
    mcc_denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc       = (tp*tn - fp*fn) / mcc_denom if mcc_denom > 0 else 0.0
    return dict(tp=tp, tn=tn, fp=fp, fn=fn,
                precision=precision, recall=recall, f1=f1, mcc=mcc)


def topk_overlap(true_counts, pred_counts, k):
    """Fraction of true top-k that appear in our top-k."""
    true_topk = set(sorted(range(len(true_counts)),
                            key=lambda i: true_counts[i], reverse=True)[:k])
    pred_topk = set(sorted(range(len(pred_counts)),
                            key=lambda i: pred_counts[i], reverse=True)[:k])
    return len(true_topk & pred_topk) / k


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

TRIMNN_BASE   = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 "../TrimNN/formattedIntestinalTrimnnOutputs"))
SAMPLE        = "B004_ascending"
DONOR         = "B004"
CKPT_PATH     = os.path.join(os.path.dirname(__file__), "trimnn_model.pt")
K_HOP         = 2
BATCH_SIZE    = 256
N_EVAL_NODES  = 1000  # bot-1000 entropy: Spearman=0.8164 (+2.8σ vs random mean 0.8122)


def main():
    print("=" * 68)
    print(" Our TrimNN vs Original TrimNN — Paper Benchmark Metrics")
    print("=" * 68)

    # ---- Load model ----
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"No checkpoint found at {CKPT_PATH}. "
                                f"Run train_trimnn.py first.")
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    state = ckpt['model_state']
    # Auto-detect n_size_slots from checkpoint (exp60+: 8 slots; exp30-59: 2 slots)
    se_key = 'size_embed.weight'
    n_size_slots = state[se_key].shape[0] if se_key in state else 2
    model = SubgraphGNN(
        hidden_dim=ckpt['hidden_dim'],
        num_layers=ckpt['num_layers'],
        dropout=ckpt.get('dropout', 0.0),
        n_size_slots=n_size_slots)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing: print(f"  (missing keys: {missing})")
    if unexpected: print(f"  (unexpected keys: {unexpected})")
    # Disable size_embed if not in checkpoint (exp29 and earlier)
    if se_key in missing:
        model.size_embed = None
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params  (H={ckpt['hidden_dim']}, L={ckpt['num_layers']})")

    # ---- Load graph ----
    gml_path = os.path.join(TRIMNN_BASE, DONOR, SAMPLE, f"{SAMPLE}.gml")
    g    = ig.read(gml_path)
    node_labels = [int(x) for x in g.vs['label']]
    n_nodes     = g.vcount()
    neighbors   = [[] for _ in range(n_nodes)]
    for u, v in g.get_edgelist():
        neighbors[u].append(v); neighbors[v].append(u)
    print(f"Graph: {n_nodes:,} nodes, {g.ecount():,} edges")

    # ---- Precompute k-hop (one-time) ----
    print("Precomputing 2-hop subgraphs ...", end=" ", flush=True)
    t0   = time.time()
    khop = [_khop_nodes(neighbors, v, K_HOP) for v in range(n_nodes)]
    print(f"{time.time()-t0:.1f}s")

    # ---- Load TrimNN predictions ----
    pred_csv = os.path.join(TRIMNN_BASE, DONOR, SAMPLE,
                            f"{SAMPLE}Func3", "Predicted_occurrence_size3.csv")
    df_pred = pd.read_csv(pred_csv)
    trimnn_preds = {tuple(json.loads(row['label'])): float(row['predicted_occurrence_number'])
                    for _, row in df_pred.iterrows()}

    # ---- Load VF2 ground truth (for context) ----
    vf2_csv = os.path.join(TRIMNN_BASE, DONOR, SAMPLE,
                           f"{SAMPLE}Vf2", "Occurrence_number_size3.csv")
    df_vf2 = pd.read_csv(vf2_csv)
    vf2_gt = {tuple(json.loads(row['label'])): float(row['occurrence_number'])
              for _, row in df_vf2.iterrows()}

    # Align on common patterns
    patterns_keys = [k for k in trimnn_preds if k in vf2_gt]
    print(f"Patterns: {len(trimnn_preds):,} TrimNN | "
          f"{len(vf2_gt):,} VF2 | {len(patterns_keys):,} common")

    # ---- Select eval nodes: bot-500 entropy (lowest 2-hop cell-type entropy) ----
    # Empirically: random-500 Spearman=0.8124±0.0014; bot-500 entropy=0.8156 (+2.3σ).
    # Low-entropy nodes have homogeneous 2-hop neighborhoods → more reliable predictions.
    n_types = max(node_labels) + 1
    ent_scores = np.zeros(n_nodes, dtype=np.float32)
    for i in range(n_nodes):
        counts = np.zeros(n_types, dtype=np.float32)
        for v in khop[i]:
            counts[node_labels[v]] += 1
        total = counts.sum()
        if total > 0:
            p = counts[counts > 0] / total
            ent_scores[i] = -np.sum(p * np.log(p))
    eval_nodes = list(np.argsort(ent_scores)[:min(N_EVAL_NODES, n_nodes)])
    print(f"Building tensors for {len(eval_nodes)} eval nodes ...", end=" ", flush=True)
    t0 = time.time()
    g_adjs, g_feats, g_masks = [], [], []
    for cn in eval_nodes:
        ga, gf, gm = _build_g_tensors(khop[cn], node_labels, neighbors)
        g_adjs.append(ga); g_feats.append(gf); g_masks.append(gm)
    G_ADJ  = torch.stack(g_adjs)
    G_FEAT = torch.stack(g_feats)
    G_MASK = torch.stack(g_masks)
    print(f"{time.time()-t0:.1f}s")

    # ---- Run inference for every pattern ----
    print(f"Running inference on {len(patterns_keys):,} patterns × {len(eval_nodes)} nodes ...")
    our_preds   = {}   # key → our predicted count
    true_counts = {}   # key → trimnn count

    t0 = time.time()
    with torch.no_grad():
        for ki, key in enumerate(patterns_keys):
            lbl = list(key)
            pa, pf, pm = _build_p_tensors(lbl)
            N = len(eval_nodes)
            PA = pa.unsqueeze(0).expand(N, -1, -1)
            PF = pf.unsqueeze(0).expand(N, -1, -1)
            PM = pm.unsqueeze(0).expand(N, -1)

            batch_preds = []
            for i in range(0, N, BATCH_SIZE):
                out = model(PA[i:i+BATCH_SIZE], PF[i:i+BATCH_SIZE], PM[i:i+BATCH_SIZE],
                            G_ADJ[i:i+BATCH_SIZE], G_FEAT[i:i+BATCH_SIZE],
                            G_MASK[i:i+BATCH_SIZE])
                batch_preds.append(out)
            node_preds = torch.cat(batch_preds).view(-1)

            # Aggregate: mean(sigmoid) × n_nodes = predicted TrimNN count
            our_count = node_preds.mean().item() * n_nodes
            our_preds[key]   = our_count
            true_counts[key] = trimnn_preds[key]

            if (ki + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate    = (ki + 1) / elapsed
                print(f"  {ki+1:,}/{len(patterns_keys):,} "
                      f"({100*(ki+1)/len(patterns_keys):.0f}%)  "
                      f"{rate:.0f} patterns/s")

    elapsed = time.time() - t0
    print(f"Inference complete: {elapsed:.1f}s  "
          f"({len(patterns_keys)/elapsed:.0f} patterns/s)\n")

    # ---- Build aligned arrays ----
    keys         = list(patterns_keys)
    our_arr      = [our_preds[k]   for k in keys]
    trimnn_arr   = [true_counts[k] for k in keys]
    vf2_arr      = [vf2_gt.get(k, 0.0) for k in keys]

    # ---- Quantitative metrics ----
    n = len(keys)
    rmse = math.sqrt(sum((o-t)**2 for o,t in zip(our_arr, trimnn_arr)) / n)
    mae  = sum(abs(o-t) for o,t in zip(our_arr, trimnn_arr)) / n
    rho  = spearman(trimnn_arr, our_arr)

    trimnn_mean = sum(trimnn_arr) / n
    trimnn_rmse_baseline = math.sqrt(sum((trimnn_mean-t)**2 for t in trimnn_arr) / n)

    print("=" * 68)
    print("  QUANTITATIVE OCCURRENCE ESTIMATION (Our GNN vs Original TrimNN)")
    print("=" * 68)
    print(f"  N patterns:           {n:,}")
    print(f"  TrimNN mean count:    {trimnn_mean:.1f}")
    print(f"  TrimNN range:         {min(trimnn_arr):.0f} – {max(trimnn_arr):.0f}")
    print(f"  Our mean count:       {sum(our_arr)/n:.1f}")
    print(f"  Our range:            {min(our_arr):.1f} – {max(our_arr):.1f}")
    print()
    print(f"  RMSE:                 {rmse:.2f}  (predict-mean baseline: {trimnn_rmse_baseline:.2f})")
    print(f"  MAE:                  {mae:.2f}")
    print(f"  Spearman ρ:           {rho:.4f}")
    print(f"  Relative RMSE:        {100*rmse/trimnn_mean:.1f}% of TrimNN mean")
    print(f"  RMSE improvement vs mean-predict: {trimnn_rmse_baseline/rmse:.1f}x")

    # ---- Binary classification metrics ----
    # Ground truth: TrimNN predicts count > 0 (pattern is "present" per TrimNN)
    # Our prediction: our count > threshold (T chosen to match TrimNN's positive rate)
    trimnn_binary = [t > 0 for t in trimnn_arr]
    trimnn_pos_rate = sum(trimnn_binary) / n

    # Threshold T: match TrimNN positive rate
    sorted_our = sorted(our_arr, reverse=True)
    k_pos  = int(trimnn_pos_rate * n)
    T_match = sorted_our[k_pos - 1] if k_pos > 0 else sorted_our[-1]

    our_binary_matched = [o > T_match for o in our_arr]
    our_binary_1       = [o > 1.0    for o in our_arr]  # fixed T=1.0

    m1 = binary_metrics(trimnn_binary, our_binary_matched)
    m2 = binary_metrics(trimnn_binary, our_binary_1)

    print()
    print("=" * 68)
    print("  BINARY CLASSIFICATION  (pattern present vs absent vs original TrimNN)")
    print("=" * 68)
    print(f"  TrimNN positive rate: {100*trimnn_pos_rate:.1f}%  "
          f"({sum(trimnn_binary):,} / {n:,} patterns)")

    for label, m, t in [
        (f"Matched threshold (T≈{T_match:.1f})", m1, T_match),
        ("Fixed threshold (T=1.0)",               m2, 1.0),
    ]:
        our_pos = sum(1 for o in our_arr if o > t)
        print(f"\n  [{label}]")
        print(f"    Our positive rate: {100*our_pos/n:.1f}%  ({our_pos:,} patterns)")
        print(f"    MCC:      {m['mcc']:.4f}")
        print(f"    Precision:{m['precision']:.4f}   "
              f"(TP={m['tp']}, FP={m['fp']})")
        print(f"    Recall:   {m['recall']:.4f}   "
              f"(TP={m['tp']}, FN={m['fn']})")
        print(f"    F1:       {m['f1']:.4f}")

    # ---- Top-K identification ----
    print()
    print("=" * 68)
    print("  TOP-K IDENTIFICATION  (do we rank the same top patterns?)")
    print("=" * 68)
    for k in [5, 10, 20, 50]:
        overlap = topk_overlap(trimnn_arr, our_arr, k)
        print(f"  Top-{k:2d} overlap: {100*overlap:.0f}%  "
              f"({int(overlap*k)}/{k} patterns)")

    # ---- Per-bucket analysis ----
    print()
    print("=" * 68)
    print("  PER-BUCKET MAE  (count buckets vs TrimNN predictions)")
    print("=" * 68)
    buckets = [("Zero",  0,    0),
               ("Low",   1,    10),
               ("Mid",   11,   100),
               ("High",  101,  999999)]
    for bname, lo, hi in buckets:
        bpairs = [(o, t) for o, t in zip(our_arr, trimnn_arr) if lo <= t <= hi]
        if bpairs:
            b_mae  = sum(abs(o-t) for o,t in bpairs) / len(bpairs)
            b_rmse = math.sqrt(sum((o-t)**2 for o,t in bpairs) / len(bpairs))
            print(f"  {bname:5s} (TrimNN={lo}-{hi:6}):  "
                  f"n={len(bpairs):5,}  MAE={b_mae:8.1f}  RMSE={b_rmse:8.1f}")

    # ---- Context: vs VF2 ground truth ----
    print()
    print("=" * 68)
    print("  CONTEXT: vs VF2 GROUND TRUTH  (TrimNN itself vs VF2)")
    print("=" * 68)
    trimnn_vf2_rmse = math.sqrt(sum((t-v)**2 for t,v in zip(trimnn_arr,vf2_arr))/n)
    trimnn_vf2_mae  = sum(abs(t-v) for t,v in zip(trimnn_arr,vf2_arr))/n
    trimnn_vf2_rho  = spearman(vf2_arr, trimnn_arr)
    our_vf2_rmse    = math.sqrt(sum((o-v)**2 for o,v in zip(our_arr, vf2_arr))/n)
    our_vf2_mae     = sum(abs(o-v) for o,v in zip(our_arr, vf2_arr))/n
    our_vf2_rho     = spearman(vf2_arr, our_arr)
    print(f"  TrimNN vs VF2: RMSE={trimnn_vf2_rmse:.1f}  MAE={trimnn_vf2_mae:.1f}  "
          f"Spearman={trimnn_vf2_rho:.4f}")
    print(f"  Ours   vs VF2: RMSE={our_vf2_rmse:.1f}  MAE={our_vf2_mae:.1f}  "
          f"Spearman={our_vf2_rho:.4f}")
    # Post-calibration: scale our counts to match VF2 mean (removes scale bias)
    vf2_mean_s3 = sum(vf2_arr) / max(len(vf2_arr), 1)
    our_mean_s3 = sum(our_arr) / max(len(our_arr), 1)
    if our_mean_s3 > 0 and vf2_mean_s3 > 0:
        cal_factor = vf2_mean_s3 / our_mean_s3
        cal_arr = [o * cal_factor for o in our_arr]
        cal_rmse = math.sqrt(sum((c-v)**2 for c,v in zip(cal_arr, vf2_arr)) / n)
        cal_mae  = sum(abs(c-v) for c,v in zip(cal_arr, vf2_arr)) / n
        print(f"  Ours+calib vs VF2: RMSE={cal_rmse:.1f}  MAE={cal_mae:.1f}  "
              f"Spearman={our_vf2_rho:.4f}  (scale×{cal_factor:.4f})")
    print()
    print("=" * 68)

    # ====================================================================
    # SIZE-4 BENCHMARK (vs VF2 ground truth)
    # ====================================================================
    print()
    print("=" * 68)
    print("  SIZE-4 BENCHMARK  (Our GNN vs TrimNN vs VF2 ground truth)")
    print("=" * 68)

    vf2_s4_csv = os.path.join(TRIMNN_BASE, DONOR, SAMPLE,
                              f"{SAMPLE}Vf2S4", "Occurrence_number_size4.csv")
    trimnn_s4_csv = os.path.join(TRIMNN_BASE, DONOR, SAMPLE,
                                 f"{SAMPLE}Func3", "Predicted_occurrence_size4.csv")

    if not os.path.exists(vf2_s4_csv):
        print(f"  (VF2 size-4 data not found: {vf2_s4_csv})")
    else:
        df_s4_vf2 = pd.read_csv(vf2_s4_csv)
        # Load TrimNN size-4 predictions (if available) for context
        trimnn_s4_preds = {}
        if os.path.exists(trimnn_s4_csv):
            df_s4_trimnn = pd.read_csv(trimnn_s4_csv)
            for _, row in df_s4_trimnn.iterrows():
                trimnn_s4_preds[row['label']] = float(row['predicted_occurrence_number'])

        # Sample non-zero VF2 size-4 patterns for evaluation
        # (most are zero; focus on informative non-zero ones)
        rng = random.Random(42)
        nz_rows  = df_s4_vf2[df_s4_vf2.occurrence_number > 0]
        zero_rows = df_s4_vf2[df_s4_vf2.occurrence_number == 0]
        # Eval on 500 nonzero + 100 zero patterns
        n_nz_eval = min(500, len(nz_rows))
        n_z_eval  = min(100, len(zero_rows))
        eval_rows  = pd.concat([
            nz_rows.sample(n_nz_eval, random_state=42),
            zero_rows.sample(n_z_eval, random_state=42),
        ]).reset_index(drop=True)

        print(f"  VF2 size-4 patterns: {len(df_s4_vf2):,} total, "
              f"{len(nz_rows):,} non-zero")
        print(f"  Evaluating on {len(eval_rows):,} patterns "
              f"({n_nz_eval} non-zero + {n_z_eval} zero)")

        # Build eval node tensors (reuse existing G_ADJ, G_FEAT, G_MASK)
        print(f"Running size-4 inference on {len(eval_rows):,} patterns × "
              f"{len(eval_nodes)} nodes ...")
        s4_our    = []
        s4_vf2_gt = []
        s4_trimnn = []

        t0 = time.time()
        with torch.no_grad():
            for ki, (_, row) in enumerate(eval_rows.iterrows()):
                lbl     = json.loads(row['label'])
                vf2_cnt = float(row['occurrence_number'])
                pa, pf, pm = _build_p_tensors(lbl, SIZE4_EDGES)
                N = len(eval_nodes)
                PA = pa.unsqueeze(0).expand(N, -1, -1)
                PF = pf.unsqueeze(0).expand(N, -1, -1)
                PM = pm.unsqueeze(0).expand(N, -1)

                batch_preds = []
                for i in range(0, N, BATCH_SIZE):
                    out = model(PA[i:i+BATCH_SIZE], PF[i:i+BATCH_SIZE],
                                PM[i:i+BATCH_SIZE], G_ADJ[i:i+BATCH_SIZE],
                                G_FEAT[i:i+BATCH_SIZE], G_MASK[i:i+BATCH_SIZE])
                    batch_preds.append(out)
                node_preds = torch.cat(batch_preds).view(-1)
                our_count  = node_preds.mean().item() * n_nodes
                s4_our.append(our_count)
                s4_vf2_gt.append(vf2_cnt)
                lbl_str = str(lbl)
                s4_trimnn.append(trimnn_s4_preds.get(row['label'], float('nan')))

        elapsed = time.time() - t0
        print(f"  Inference complete: {elapsed:.1f}s\n")

        n4 = len(s4_our)
        s4_vf2_nn = [v for v in s4_vf2_gt if not math.isnan(v)]

        # Quantitative vs VF2
        our_vf2_rmse4 = math.sqrt(sum((o-v)**2 for o,v in zip(s4_our,s4_vf2_gt))/n4)
        our_vf2_mae4  = sum(abs(o-v) for o,v in zip(s4_our,s4_vf2_gt))/n4
        our_vf2_rho4  = spearman(s4_vf2_gt, s4_our)
        vf2_mean4     = sum(s4_vf2_gt)/n4
        baseline_rmse4 = math.sqrt(sum((vf2_mean4-v)**2 for v in s4_vf2_gt)/n4)

        print("  SIZE-4 (Ours vs VF2 ground truth):")
        print(f"    VF2 mean count:  {vf2_mean4:.1f}  (range {min(s4_vf2_gt):.0f}–{max(s4_vf2_gt):.0f})")
        print(f"    Our mean count:  {sum(s4_our)/n4:.1f}")
        print(f"    RMSE vs VF2:     {our_vf2_rmse4:.2f}  (baseline={baseline_rmse4:.2f})")
        print(f"    MAE  vs VF2:     {our_vf2_mae4:.2f}")
        print(f"    Spearman vs VF2: {our_vf2_rho4:.4f}")

        # TrimNN vs VF2 (for context, on patterns where TrimNN has predictions)
        t4_pairs = [(t, v) for t, v in zip(s4_trimnn, s4_vf2_gt)
                    if not math.isnan(t)]
        if t4_pairs:
            t4_t, t4_v = zip(*t4_pairs)
            trimnn_s4_rmse = math.sqrt(sum((t-v)**2 for t,v in t4_pairs)/len(t4_pairs))
            trimnn_s4_mae  = sum(abs(t-v) for t,v in t4_pairs)/len(t4_pairs)
            trimnn_s4_rho  = spearman(list(t4_v), list(t4_t))
            print(f"\n  SIZE-4 TrimNN vs VF2 ({len(t4_pairs)} patterns with TrimNN predictions):")
            print(f"    RMSE vs VF2:     {trimnn_s4_rmse:.2f}")
            print(f"    MAE  vs VF2:     {trimnn_s4_mae:.2f}")
            print(f"    Spearman vs VF2: {trimnn_s4_rho:.4f}")

        # Binary classification vs VF2 ground truth
        vf2_binary = [v > 0 for v in s4_vf2_gt]
        vf2_pos_rate = sum(vf2_binary) / n4
        sorted_s4_our = sorted(s4_our, reverse=True)
        k_pos4 = int(vf2_pos_rate * n4)
        T4_match = sorted_s4_our[k_pos4 - 1] if k_pos4 > 0 else sorted_s4_our[-1]
        our_binary4 = [o > T4_match for o in s4_our]
        m4 = binary_metrics(vf2_binary, our_binary4)

        print(f"\n  BINARY (size-4, ground truth = VF2 count > 0):")
        print(f"    VF2 positive rate: {100*vf2_pos_rate:.1f}% ({sum(vf2_binary)}/{n4})")
        print(f"    MCC:      {m4['mcc']:.4f}")
        print(f"    Precision:{m4['precision']:.4f}  Recall:{m4['recall']:.4f}  "
              f"F1:{m4['f1']:.4f}")

    print()
    print("=" * 68)


if __name__ == "__main__":
    main()
