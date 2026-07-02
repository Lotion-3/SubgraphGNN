"""
Cross-sample generalization test: model trained on B004_ascending, evaluated on another sample.
Computes VF2 Spearman on out-of-distribution data.

Standalone — does NOT import train_trimnn.py to avoid triggering its module-level training code.
"""

import os, json, math, time, random
import numpy as np
import pandas as pd
import igraph as ig
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Config ----
CKPT_PATH    = os.path.join(os.path.dirname(__file__), "trimnn_model.pt")
INT_BASE     = "/media/volume/LLM_largeData/TrimNNCMD/TrimNN/intestinalOutputs"
EVAL_SAMPLE  = "B005_ascending_ct24"
GML_PATH     = os.path.join(INT_BASE, EVAL_SAMPLE, f"{EVAL_SAMPLE}.gml")
VF2_PATH     = os.path.join(INT_BASE, EVAL_SAMPLE, f"{EVAL_SAMPLE}_vf2s3",
                             "Occurrence_number_size3.csv")
K_HOP        = 2
N_EVAL_NODES = 500
BATCH_SIZE   = 64
MAX_LABELS   = 32
MAX_GRAPH    = 48
MAX_PAT      = 8

TRIANGLE_EDGES = [(0,1),(0,2),(1,2)]

# ---- Model (copied from train_trimnn.py) ----

class GNNLayer(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.linear = nn.Linear(dim * 2, dim)
        self.norm   = nn.LayerNorm(dim)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x, adj_norm):
        msg = torch.bmm(adj_norm, x)
        h   = torch.cat([x, msg], dim=-1)
        return x + self.drop(self.norm(F.gelu(self.linear(h))))


class SubgraphGNN(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout, max_labels=MAX_LABELS):
        super().__init__()
        self.max_labels = max_labels
        self.embed      = nn.Linear(max_labels, hidden_dim, bias=False)
        self.layers     = nn.ModuleList([GNNLayer(hidden_dim, dropout) for _ in range(num_layers)])
        self.cross_attn_p2g = nn.MultiheadAttention(hidden_dim, 4, dropout=dropout, batch_first=True)
        self.cross_attn_g2p = nn.MultiheadAttention(hidden_dim, 4, dropout=dropout, batch_first=True)
        self.norm_p    = nn.LayerNorm(hidden_dim)
        self.norm_g    = nn.LayerNorm(hidden_dim)
        self.gate_g    = nn.Linear(hidden_dim, hidden_dim)
        nn.init.constant_(self.gate_g.bias, -2.0)
        self.tri_embed = nn.Linear(1, hidden_dim, bias=False)
        self.predict   = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid())

    @staticmethod
    def _norm_adj(adj):
        return adj / adj.sum(dim=-1, keepdim=True).clamp(min=1.0)

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
        g_key_mask = ~g_mask
        p_key_mask = ~p_mask
        p_mask_f = p_mask.unsqueeze(-1).float()
        g_mask_f = g_mask.unsqueeze(-1).float()
        p_attn, _ = self.cross_attn_p2g(self.norm_p(p_enc), self.norm_g(g_enc), self.norm_g(g_enc), key_padding_mask=g_key_mask)
        p_out = p_enc + p_attn
        g_attn, _ = self.cross_attn_g2p(self.norm_g(g_enc), self.norm_p(p_enc), self.norm_p(p_enc), key_padding_mask=p_key_mask)
        g_out = g_enc + g_attn
        p_pool  = (p_out * p_mask_f).max(dim=1).values
        g_pool  = (g_out * g_mask_f).max(dim=1).values
        gate    = torch.sigmoid(self.gate_g(g_out))
        g_gated = (gate * g_enc * g_mask_f).sum(dim=1)
        return self.predict(torch.cat([p_pool, g_pool, g_gated], dim=-1))


# ---- Helpers ----

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
    n = min(len(node_list), MAX_GRAPH)
    node_idx = {v: i for i, v in enumerate(node_list)}
    adj  = np.zeros((MAX_GRAPH, MAX_GRAPH), dtype=np.float32)
    feat = np.zeros((MAX_GRAPH, MAX_LABELS), dtype=np.float32)
    mask = np.zeros((MAX_GRAPH,), dtype=bool)
    for i, v in enumerate(node_list[:n]):
        lbl = int(node_labels[v]) % MAX_LABELS
        feat[i, lbl] = 1.0
        mask[i] = True
        for nb in neighbors_map[v]:
            j = node_idx.get(nb)
            if j is not None and j < n:
                adj[i, j] = adj[j, i] = 1.0
    return torch.from_numpy(adj), torch.from_numpy(feat), torch.from_numpy(mask)


def _build_p_tensors(label_list, edges=None):
    if edges is None:
        edges = TRIANGLE_EDGES
    adj  = torch.zeros(MAX_PAT, MAX_PAT)
    for u, v in edges:
        if u < MAX_PAT and v < MAX_PAT:
            adj[u, v] = adj[v, u] = 1.0
    feat = torch.zeros(MAX_PAT, MAX_LABELS)
    for i, lbl in enumerate(label_list[:MAX_PAT]):
        feat[i, int(lbl) % MAX_LABELS] = 1.0
    mask = torch.zeros(MAX_PAT, dtype=torch.bool)
    mask[:min(len(label_list), MAX_PAT)] = True
    return adj, feat, mask


def spearman(x, y):
    n = len(x)
    if n < 2: return 0.0
    def ranks(lst):
        s = sorted(range(n), key=lambda i: lst[i])
        r = [0]*n
        for rk, idx in enumerate(s): r[idx] = rk
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx)/n, sum(ry)/n
    num = sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    den = (sum((r-mx)**2 for r in rx)*sum((r-my)**2 for r in ry))**0.5
    return num/den if den > 0 else 0.0


# ---- Main ----

print(f"Loading model from {CKPT_PATH} ...")
ckpt  = torch.load(CKPT_PATH, map_location='cpu')
model = SubgraphGNN(ckpt['hidden_dim'], ckpt['num_layers'], ckpt['dropout'])
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params  "
      f"(H={ckpt['hidden_dim']}, L={ckpt['num_layers']})")

# Load label harmonization for the eval sample
GLOBAL_CELL_TYPES = [
    'B','CD4+Tcell','CD57+Enterocyte','CD66+Enterocyte','CD7+Immune','CD8+T',
    'CyclingTA','DC','Endothelial','Enterocyte','Goblet','ICC','Lymphatic',
    'M1Macrophage','M2Macrophage','MUC1+Enterocyte','NK','Nerve','Neuroendocrine',
    'Neutrophil','Paneth','Plasma','Smoothmuscle','Stroma','TA'
]
GLOBAL_TYPE_ID = {ct: i for i, ct in enumerate(GLOBAL_CELL_TYPES)}

cell_csv = os.path.join(INT_BASE, EVAL_SAMPLE, 'cell_type_to_id.csv')
remap = {}
if os.path.exists(cell_csv):
    ct_df = pd.read_csv(cell_csv)
    remap = {int(row['cell_type_id']): GLOBAL_TYPE_ID.get(row['cell_type'], int(row['cell_type_id']))
             for _, row in ct_df.iterrows()}
    n_changes = sum(1 for k,v in remap.items() if k != v)
    print(f"Label harmonization: {len(remap)} types, {n_changes} remapped")

print(f"\nLoading graph: {GML_PATH} ...")
g           = ig.read(GML_PATH)
n_nodes     = g.vcount()
# Apply harmonized labels to graph nodes
node_labels = [remap.get(int(v['label']), int(v['label'])) for v in g.vs]
neighbors   = [[] for _ in range(n_nodes)]
for u, v in g.get_edgelist():
    neighbors[u].append(v); neighbors[v].append(u)
print(f"Graph: {n_nodes:,} nodes, {g.ecount():,} edges")

print(f"Precomputing {K_HOP}-hop subgraphs ...")
t0   = time.time()
khop = [_khop_nodes(neighbors, v, K_HOP) for v in range(n_nodes)]
print(f"Done in {time.time()-t0:.1f}s")

print(f"\nLoading VF2 data: {VF2_PATH} ...")
vf2_df   = pd.read_csv(VF2_PATH)
# Apply harmonized labels to pattern labels
patterns = [([remap.get(l,l) for l in json.loads(row['label'])], int(row['occurrence_number']))
            for _, row in vf2_df.iterrows()]
n_zero    = sum(1 for _, c in patterns if c == 0)
n_nonzero = sum(1 for _, c in patterns if c > 0)
vf2_mean  = sum(c for _, c in patterns) / len(patterns)
print(f"Patterns: {len(patterns):,}  (zero={n_zero}, nonzero={n_nonzero}, mean={vf2_mean:.1f})")

rng        = random.Random(42)
eval_nodes = rng.sample(range(n_nodes), min(N_EVAL_NODES, n_nodes))
print(f"\nBuilding tensors for {len(eval_nodes)} eval nodes ...")
t0 = time.time()
g_adjs, g_feats, g_masks = [], [], []
for cn in eval_nodes:
    ga, gf, gm = _build_g_tensors(khop[cn], node_labels, neighbors)
    g_adjs.append(ga); g_feats.append(gf); g_masks.append(gm)
G_ADJ  = torch.stack(g_adjs)
G_FEAT = torch.stack(g_feats)
G_MASK = torch.stack(g_masks)
N_eval = len(eval_nodes)
print(f"Done in {time.time()-t0:.1f}s")

print(f"\nRunning inference on {len(patterns):,} patterns × {N_eval} nodes ...")
t0  = time.time()
our = []
vf2 = []
with torch.no_grad():
    for ki, (lbl, vf2_cnt) in enumerate(patterns):
        pa, pf, pm = _build_p_tensors(lbl)
        PA = pa.unsqueeze(0).expand(N_eval, -1, -1)
        PF = pf.unsqueeze(0).expand(N_eval, -1, -1)
        PM = pm.unsqueeze(0).expand(N_eval, -1)
        preds = []
        for i in range(0, N_eval, BATCH_SIZE):
            out = model(PA[i:i+BATCH_SIZE], PF[i:i+BATCH_SIZE], PM[i:i+BATCH_SIZE],
                        G_ADJ[i:i+BATCH_SIZE], G_FEAT[i:i+BATCH_SIZE], G_MASK[i:i+BATCH_SIZE])
            preds.append(out)
        our_cnt = torch.cat(preds).mean().item() * n_nodes
        our.append(our_cnt)
        vf2.append(float(vf2_cnt))
        if (ki+1) % 500 == 0:
            elapsed = time.time()-t0
            print(f"  {ki+1:,}/{len(patterns):,}  {(ki+1)/elapsed:.0f} pat/s")

elapsed = time.time()-t0
print(f"Inference: {elapsed:.1f}s  ({len(patterns)/elapsed:.0f} pat/s)")

# Metrics
n    = len(our)
rho  = spearman(our, vf2)
rmse = math.sqrt(sum((o-v)**2 for o,v in zip(our,vf2)) / n)
mae  = sum(abs(o-v) for o,v in zip(our,vf2)) / n

print(f"\n{'='*60}")
print(f"  CROSS-SAMPLE: B004_ascending (train) → {EVAL_SAMPLE} (eval)")
print(f"{'='*60}")
print(f"  N patterns:      {n:,}")
print(f"  VF2 mean count:  {sum(vf2)/n:.1f}")
print(f"  Our mean count:  {sum(our)/n:.1f}")
print(f"  Spearman vs VF2: {rho:.4f}")
print(f"  RMSE vs VF2:     {rmse:.2f}")
print(f"  MAE  vs VF2:     {mae:.2f}")
print(f"{'='*60}")
