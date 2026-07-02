"""
Inference timing benchmark for the TrimNN replacement GNN model.

Measures forward-pass throughput on B004_ascending (21,232 nodes, 2925 patterns).
Reports: seconds per pattern (sampled nodes), projected total for all patterns.
"""

import os, math, time, random
import numpy as np
import pandas as pd
import igraph as ig

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Minimal model copy (same SubgraphGNN as train.py)
# ---------------------------------------------------------------------------

MAX_LABELS      = 32
MAX_GRAPH_NODES = 48
MAX_PATTERN_NODES = 4

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
    def __init__(self, hidden_dim=128, num_layers=3, dropout=0.0):
        super().__init__()
        self.max_labels = MAX_LABELS
        self.embed      = nn.Linear(MAX_LABELS, hidden_dim, bias=False)
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
# Data helpers (same as train.py)
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


def _build_g_tensors(node_list, node_labels, neighbors_map,
                     max_nodes=MAX_GRAPH_NODES):
    n        = min(len(node_list), max_nodes)
    node_idx = {v: i for i, v in enumerate(node_list)}
    adj  = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    feat = np.zeros((max_nodes, MAX_LABELS),  dtype=np.float32)
    mask = np.zeros((max_nodes,),             dtype=bool)
    for i, v in enumerate(node_list[:n]):
        lbl = int(node_labels[v]) % MAX_LABELS
        feat[i, lbl] = 1.0
        mask[i] = True
        for nb in neighbors_map[v]:
            j = node_idx.get(nb)
            if j is not None and j < n:
                adj[i, j] = adj[j, i] = 1.0
    return (torch.from_numpy(adj),
            torch.from_numpy(feat),
            torch.from_numpy(mask))


def _build_p_tensors(label_list, max_pat=MAX_PATTERN_NODES):
    adj  = torch.zeros(max_pat, max_pat)
    adj[0,1]=adj[1,0]=adj[0,2]=adj[2,0]=adj[1,2]=adj[2,1]=1.0
    feat = torch.zeros(max_pat, MAX_LABELS)
    for i, lbl in enumerate(label_list[:max_pat]):
        feat[i, int(lbl) % MAX_LABELS] = 1.0
    mask = torch.zeros(max_pat, dtype=torch.bool)
    mask[:min(len(label_list), max_pat)] = True
    return adj, feat, mask


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

TRIMNN_BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 "../TrimNN/formattedIntestinalTrimnnOutputs"))
SAMPLE      = "B004_ascending"
K_HOP       = 2
BATCH_SIZE  = 128
N_NODES_SAMPLE = 150   # nodes sampled per pattern (eval mode)

def main():
    print("=" * 60)
    print("GNN Inference Timing Benchmark")
    print("=" * 60)

    # ---- Load graph ----
    gml_path = os.path.join(TRIMNN_BASE, "B004", SAMPLE, f"{SAMPLE}.gml")
    t0 = time.time()
    g  = ig.read(gml_path)
    node_labels = [int(x) for x in g.vs['label']]
    n_nodes     = g.vcount()
    neighbors   = [[] for _ in range(n_nodes)]
    for u, v in g.get_edgelist():
        neighbors[u].append(v); neighbors[v].append(u)
    print(f"Graph loaded: {n_nodes:,} nodes, {g.ecount():,} edges  [{time.time()-t0:.2f}s]")

    # ---- Precompute k-hop subgraphs ----
    t0   = time.time()
    khop = [_khop_nodes(neighbors, v, K_HOP) for v in range(n_nodes)]
    t_khop = time.time() - t0
    khop_sizes = [len(k) for k in khop]
    print(f"K-hop precompute: {t_khop:.2f}s  "
          f"(avg subgraph size {sum(khop_sizes)/len(khop_sizes):.1f} nodes)")

    # ---- Load patterns ----
    pred_csv = os.path.join(TRIMNN_BASE, "B004", SAMPLE,
                            f"{SAMPLE}Func3", "Predicted_occurrence_size3.csv")
    df = pd.read_csv(pred_csv)
    import json
    patterns = [(json.loads(row['label']), float(row['predicted_occurrence_number']))
                for _, row in df.iterrows()]
    print(f"Patterns: {len(patterns):,}")

    # ---- Build model ----
    model = SubgraphGNN(hidden_dim=128, num_layers=3, dropout=0.0)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,} ({n_params/1e6:.2f}M)")

    # ---- Precompute graph tensors for sampled nodes ----
    rng          = random.Random(0)
    sampled_nodes = rng.sample(range(n_nodes), min(N_NODES_SAMPLE, n_nodes))
    t0 = time.time()
    g_adjs, g_feats, g_masks = [], [], []
    for cn in sampled_nodes:
        ga, gf, gm = _build_g_tensors(khop[cn], node_labels, neighbors)
        g_adjs.append(ga); g_feats.append(gf); g_masks.append(gm)
    G_ADJ  = torch.stack(g_adjs)
    G_FEAT = torch.stack(g_feats)
    G_MASK = torch.stack(g_masks)
    t_build = time.time() - t0
    print(f"Node tensor build ({N_NODES_SAMPLE} nodes): {t_build*1000:.1f}ms")

    # ---- Warm-up (1 forward pass) ----
    with torch.no_grad():
        lbl0, _ = patterns[0]
        pa, pf, pm = _build_p_tensors(lbl0)
        PA = pa.unsqueeze(0).expand(BATCH_SIZE, -1, -1)[:BATCH_SIZE]
        PF = pf.unsqueeze(0).expand(BATCH_SIZE, -1, -1)[:BATCH_SIZE]
        PM = pm.unsqueeze(0).expand(BATCH_SIZE, -1)[:BATCH_SIZE]
        _ = model(PA, PF, PM, G_ADJ[:BATCH_SIZE], G_FEAT[:BATCH_SIZE], G_MASK[:BATCH_SIZE])
    print("Warm-up done.")

    # ---- Benchmark: N_BENCH_PATTERNS patterns × N_NODES_SAMPLE nodes ----
    N_BENCH = min(100, len(patterns))
    bench_pats = patterns[:N_BENCH]

    t0 = time.time()
    with torch.no_grad():
        for lbl, _ in bench_pats:
            pa, pf, pm = _build_p_tensors(lbl)
            PA = pa.unsqueeze(0).expand(N_NODES_SAMPLE, -1, -1)
            PF = pf.unsqueeze(0).expand(N_NODES_SAMPLE, -1, -1)
            PM = pm.unsqueeze(0).expand(N_NODES_SAMPLE, -1)
            preds = []
            for i in range(0, N_NODES_SAMPLE, BATCH_SIZE):
                out = model(PA[i:i+BATCH_SIZE], PF[i:i+BATCH_SIZE], PM[i:i+BATCH_SIZE],
                            G_ADJ[i:i+BATCH_SIZE], G_FEAT[i:i+BATCH_SIZE], G_MASK[i:i+BATCH_SIZE])
                preds.append(out)
            preds = torch.cat(preds).view(-1)
    t_bench = time.time() - t0

    secs_per_pattern_sampled = t_bench / N_BENCH
    patterns_per_sec_sampled  = 1.0 / secs_per_pattern_sampled

    print()
    print("=" * 60)
    print(f"  Mode: {N_NODES_SAMPLE} sampled nodes per pattern")
    print(f"  {N_BENCH} patterns in {t_bench:.2f}s")
    print(f"  Per pattern:     {secs_per_pattern_sampled*1000:.1f}ms")
    print(f"  Patterns/sec:    {patterns_per_sec_sampled:.1f}")
    proj_all_sampled = secs_per_pattern_sampled * len(patterns)
    print(f"  All {len(patterns):,} patterns: {proj_all_sampled:.1f}s  ({proj_all_sampled/60:.1f} min)")

    # ---- Benchmark: 1 pattern × ALL nodes ----
    print()
    print(f"  Mode: all {n_nodes:,} nodes, 1 pattern")
    g_adjs_all, g_feats_all, g_masks_all = [], [], []
    BUILD_BATCH = 1000
    t0 = time.time()
    for start in range(0, n_nodes, BUILD_BATCH):
        end = min(start + BUILD_BATCH, n_nodes)
        for cn in range(start, end):
            ga, gf, gm = _build_g_tensors(khop[cn], node_labels, neighbors)
            g_adjs_all.append(ga); g_feats_all.append(gf); g_masks_all.append(gm)
    G_ADJ_ALL  = torch.stack(g_adjs_all)
    G_FEAT_ALL = torch.stack(g_feats_all)
    G_MASK_ALL = torch.stack(g_masks_all)
    t_build_all = time.time() - t0
    print(f"  All-node tensor build: {t_build_all:.2f}s")

    lbl0, _ = patterns[0]
    pa, pf, pm = _build_p_tensors(lbl0)
    t0 = time.time()
    with torch.no_grad():
        PA = pa.unsqueeze(0).expand(n_nodes, -1, -1)
        PF = pf.unsqueeze(0).expand(n_nodes, -1, -1)
        PM = pm.unsqueeze(0).expand(n_nodes, -1)
        preds_all = []
        for i in range(0, n_nodes, BATCH_SIZE):
            out = model(PA[i:i+BATCH_SIZE], PF[i:i+BATCH_SIZE], PM[i:i+BATCH_SIZE],
                        G_ADJ_ALL[i:i+BATCH_SIZE], G_FEAT_ALL[i:i+BATCH_SIZE],
                        G_MASK_ALL[i:i+BATCH_SIZE])
            preds_all.append(out)
    t_one_full = time.time() - t0
    print(f"  1 pattern, all nodes: {t_one_full:.2f}s")
    proj_all_full = t_one_full * len(patterns)
    print(f"  All {len(patterns):,} patterns: {proj_all_full:.0f}s  ({proj_all_full/3600:.1f}h)")

    print()
    print("=" * 60)
    print("Summary (CPU, no model reload per pattern):")
    print(f"  K-hop precompute (one-time): {t_khop:.2f}s")
    print(f"  Sampled ({N_NODES_SAMPLE} nodes/pattern): "
          f"{proj_all_sampled:.0f}s ({proj_all_sampled/60:.1f}min) for all patterns")
    print(f"  Full ({n_nodes:,} nodes/pattern): "
          f"{proj_all_full:.0f}s ({proj_all_full/3600:.1f}h) for all patterns")
    print("=" * 60)


if __name__ == "__main__":
    main()
