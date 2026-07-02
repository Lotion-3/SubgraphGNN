"""
Ensemble benchmark eval — averages predictions from multiple checkpoints.
Usage: python eval_ensemble_benchmark.py --models m1.pt m2.pt m3.pt ...
"""
import os, sys, json, math, time, random, argparse
import numpy as np
import pandas as pd
import igraph as ig
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_LABELS        = 40
MAX_GRAPH_NODES   = 48
MAX_PATTERN_NODES = 8
SIZE4_EDGES = [(0,1),(0,2),(1,2),(1,3),(2,3)]

# ---- Model (copy of eval_trimnn_benchmark.py) ----

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
        self.layers       = nn.ModuleList([GNNLayer(hidden_dim, dropout) for _ in range(num_layers)])
        self.cross_attn_p2g = nn.MultiheadAttention(hidden_dim, 4, dropout=dropout, batch_first=True)
        self.cross_attn_g2p = nn.MultiheadAttention(hidden_dim, 4, dropout=dropout, batch_first=True)
        self.norm_p    = nn.LayerNorm(hidden_dim)
        self.norm_g    = nn.LayerNorm(hidden_dim)
        self.gate_g    = nn.Linear(hidden_dim, hidden_dim)
        self.tri_embed = nn.Linear(1, hidden_dim, bias=False)
        self.size_embed = nn.Embedding(n_size_slots, hidden_dim)
        self.predict   = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid())

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
        p_attn, _ = self.cross_attn_p2g(self.norm_p(p_enc), self.norm_g(g_enc), self.norm_g(g_enc), key_padding_mask=~g_mask)
        p_out = p_enc + p_attn
        g_attn, _ = self.cross_attn_g2p(self.norm_g(g_enc), self.norm_p(p_enc), self.norm_p(p_enc), key_padding_mask=~p_mask)
        g_out = g_enc + g_attn
        p_pool  = (p_out * p_mask_f).max(dim=1).values
        g_pool  = (g_out * g_mask_f).max(dim=1).values
        gate    = torch.sigmoid(self.gate_g(g_out))
        g_gated = (gate * g_enc * g_mask_f).sum(dim=1)
        if getattr(self, 'size_embed', None) is not None:
            n_pat    = p_mask.float().sum(dim=1).long()
            size_idx = (n_pat - 3).clamp(0, self.n_size_slots - 1)
            p_pool   = p_pool + self.size_embed(size_idx)
        return self.predict(torch.cat([p_pool, g_pool, g_gated], dim=-1))


def load_model(path):
    ckpt  = torch.load(path, map_location='cpu')
    state = ckpt['model_state']
    se_key = 'size_embed.weight'
    n_size_slots = state[se_key].shape[0] if se_key in state else 2
    m = SubgraphGNN(ckpt['hidden_dim'], ckpt['num_layers'],
                    ckpt.get('dropout', 0.0), n_size_slots)
    missing, _ = m.load_state_dict(state, strict=False)
    if se_key in missing:
        m.size_embed = None
    m.eval()
    return m, ckpt['hidden_dim'], ckpt['num_layers']


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
        lbl = int(node_labels[v]) % MAX_LABELS
        feat[i, lbl] = 1.0; mask[i] = True
        for nb in neighbors_map[v]:
            j = node_idx.get(nb)
            if j is not None and j < n:
                adj[i, j] = adj[j, i] = 1.0
    return torch.tensor(adj), torch.tensor(feat), torch.tensor(mask)


def _build_p_tensors(label_list):
    n    = min(len(label_list), MAX_PATTERN_NODES)
    adj  = np.zeros((MAX_PATTERN_NODES, MAX_PATTERN_NODES), dtype=np.float32)
    feat = np.zeros((MAX_PATTERN_NODES, MAX_LABELS),         dtype=np.float32)
    mask = np.zeros((MAX_PATTERN_NODES,),                    dtype=bool)
    for i in range(n):
        lbl = int(label_list[i]) % MAX_LABELS
        feat[i, lbl] = 1.0; mask[i] = True
        for j in range(i):
            adj[i, j] = adj[j, i] = 1.0
    return torch.tensor(adj), torch.tensor(feat), torch.tensor(mask)


def spearman(x, y):
    n = len(x)
    if n < 2: return 0.0
    rx = sorted(range(n), key=lambda i: x[i]); rankx = [0]*n
    for r, i in enumerate(rx): rankx[i] = r
    ry = sorted(range(n), key=lambda i: y[i]); ranky = [0]*n
    for r, i in enumerate(ry): ranky[i] = r
    mx = sum(rankx)/n; my = sum(ranky)/n
    num = sum((rankx[i]-mx)*(ranky[i]-my) for i in range(n))
    dx  = math.sqrt(sum((rankx[i]-mx)**2 for i in range(n)))
    dy  = math.sqrt(sum((ranky[i]-my)**2 for i in range(n)))
    return num/(dx*dy) if dx*dy > 0 else 0.0


def binary_metrics(true_binary, pred_binary):
    tp = sum(t and p for t,p in zip(true_binary, pred_binary))
    tn = sum((not t) and (not p) for t,p in zip(true_binary, pred_binary))
    fp = sum((not t) and p for t,p in zip(true_binary, pred_binary))
    fn = sum(t and (not p) for t,p in zip(true_binary, pred_binary))
    mcc_denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = (tp*tn - fp*fn) / mcc_denom if mcc_denom > 0 else 0.0
    return mcc


TRIMNN_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            "../TrimNN/formattedIntestinalTrimnnOutputs"))
SAMPLE = "B004_ascending"; DONOR = "B004"
K_HOP  = 2; BATCH_SIZE = 256; N_EVAL_NODES = 500


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True)
    args = parser.parse_args()

    print(f"Loading {len(args.models)} models ...")
    models = []
    for p in args.models:
        m, h, l = load_model(p)
        models.append(m)
        print(f"  {os.path.basename(p)}: H={h} L={l} params={sum(x.numel() for x in m.parameters()):,}")

    gml_path = os.path.join(TRIMNN_BASE, DONOR, SAMPLE, f"{SAMPLE}.gml")
    g = ig.read(gml_path)
    node_labels = [int(x) for x in g.vs['label']]
    n_nodes = g.vcount()
    neighbors = [[] for _ in range(n_nodes)]
    for u, v in g.get_edgelist():
        neighbors[u].append(v); neighbors[v].append(u)
    print(f"Graph: {n_nodes:,} nodes")

    print("Precomputing 2-hop subgraphs ...", end=" ", flush=True)
    khop = [_khop_nodes(neighbors, v, K_HOP) for v in range(n_nodes)]
    print("done")

    pred_csv = os.path.join(TRIMNN_BASE, DONOR, SAMPLE, f"{SAMPLE}Func3", "Predicted_occurrence_size3.csv")
    vf2_csv  = os.path.join(TRIMNN_BASE, DONOR, SAMPLE, f"{SAMPLE}Vf2",   "Occurrence_number_size3.csv")
    trimnn_preds = {tuple(json.loads(r['label'])): float(r['predicted_occurrence_number'])
                    for _, r in pd.read_csv(pred_csv).iterrows()}
    vf2_gt       = {tuple(json.loads(r['label'])): float(r['occurrence_number'])
                    for _, r in pd.read_csv(vf2_csv).iterrows()}
    patterns_keys = [k for k in trimnn_preds if k in vf2_gt]
    print(f"Patterns: {len(patterns_keys):,} common")

    rng = random.Random(42)
    eval_nodes = rng.sample(range(n_nodes), min(N_EVAL_NODES, n_nodes))
    print(f"Building tensors for {len(eval_nodes)} nodes ...", end=" ", flush=True)
    g_adjs, g_feats, g_masks = [], [], []
    for cn in eval_nodes:
        ga, gf, gm = _build_g_tensors(khop[cn], node_labels, neighbors)
        g_adjs.append(ga); g_feats.append(gf); g_masks.append(gm)
    G_ADJ = torch.stack(g_adjs); G_FEAT = torch.stack(g_feats); G_MASK = torch.stack(g_masks)
    print("done")

    print(f"Running ensemble inference ({len(models)} models × {len(patterns_keys):,} patterns × {len(eval_nodes)} nodes) ...")
    our_preds = {}; true_counts = {}
    t0 = time.time()

    with torch.no_grad():
        for ki, key in enumerate(patterns_keys):
            lbl = list(key)
            pa, pf, pm = _build_p_tensors(lbl)
            N  = len(eval_nodes)
            PA = pa.unsqueeze(0).expand(N, -1, -1)
            PF = pf.unsqueeze(0).expand(N, -1, -1)
            PM = pm.unsqueeze(0).expand(N, -1)

            # Average predictions across all models
            avg_preds = None
            for model in models:
                batch_preds = []
                for i in range(0, N, BATCH_SIZE):
                    out = model(PA[i:i+BATCH_SIZE], PF[i:i+BATCH_SIZE], PM[i:i+BATCH_SIZE],
                                G_ADJ[i:i+BATCH_SIZE], G_FEAT[i:i+BATCH_SIZE], G_MASK[i:i+BATCH_SIZE])
                    batch_preds.append(out)
                node_preds = torch.cat(batch_preds).view(-1)
                avg_preds = node_preds if avg_preds is None else avg_preds + node_preds

            avg_preds = avg_preds / len(models)
            our_preds[key]   = avg_preds.mean().item() * n_nodes
            true_counts[key] = trimnn_preds[key]

            if (ki + 1) % 500 == 0:
                print(f"  {ki+1:,}/{len(patterns_keys):,} ({100*(ki+1)/len(patterns_keys):.0f}%)")

    print(f"Inference: {time.time()-t0:.1f}s\n")

    keys       = list(patterns_keys)
    our_arr    = [our_preds[k] for k in keys]
    trimnn_arr = [true_counts[k] for k in keys]
    vf2_arr    = [vf2_gt.get(k, 0.0) for k in keys]
    n = len(keys)

    our_vf2_rho = spearman(vf2_arr, our_arr)
    vf2_mean    = sum(vf2_arr) / max(n, 1)
    our_mean    = sum(our_arr) / max(n, 1)
    cal_factor  = vf2_mean / our_mean if our_mean > 0 else 1.0
    cal_arr     = [o * cal_factor for o in our_arr]
    cal_rmse    = math.sqrt(sum((c-v)**2 for c,v in zip(cal_arr, vf2_arr)) / n)

    trimnn_vf2_rho = spearman(vf2_arr, trimnn_arr)

    trimnn_binary = [t > 0 for t in trimnn_arr]
    trimnn_pos_rate = sum(trimnn_binary) / n
    sorted_our = sorted(our_arr, reverse=True)
    k_pos = int(trimnn_pos_rate * n)
    T_match = sorted_our[k_pos - 1] if k_pos > 0 else sorted_our[-1]
    our_binary = [o > T_match for o in our_arr]
    mcc = binary_metrics(trimnn_binary, our_binary)

    print("=" * 60)
    print(f"  ENSEMBLE RESULTS ({len(models)} models)")
    print("=" * 60)
    print(f"  Ours   vs VF2: Spearman={our_vf2_rho:.4f}  (TrimNN={trimnn_vf2_rho:.4f})")
    print(f"  Ours+calib vs VF2: RMSE={cal_rmse:.1f}  Spearman={our_vf2_rho:.4f}  (scale×{cal_factor:.4f})")
    print(f"  s3 MCC (matched threshold): {mcc:.4f}")
    print(f"  TrimNN vs VF2 baseline: Spearman={trimnn_vf2_rho:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
