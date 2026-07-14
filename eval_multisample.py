"""
Multi-sample cross-generalization test: evaluate model on multiple held-out samples.
Computes VF2 Spearman for each sample and reports average.

Standalone — does NOT import train_trimnn.py.
"""

import os, json, math, time, random
import numpy as np
import pandas as pd
import igraph as ig
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Config ----
CKPT_PATH  = os.path.join(os.path.dirname(__file__), "trimnn_model.pt")
INT_BASE   = os.path.join(os.path.dirname(__file__), "../TrimNN/intestinalOutputs")
K_HOP        = 2
N_EVAL_NODES = 150   # fewer nodes for faster multi-sample eval
N_EVAL_PATS  = 600   # sample 600 patterns (300 nonzero + 300 zero) per sample
BATCH_SIZE   = 64
MAX_LABELS   = 40   # must match training MAX_LABELS (exp29+: 40)
MAX_GRAPH    = 48
MAX_PAT      = 8

# Eval samples: (sample_dir, gml_stem) — standard variants only
EVAL_SAMPLES = [
    ("B005_ascending_ct24",        "B005_ascending_ct24"),
    ("B005_descendingSigmoid_ct20","B005_descendingSigmoid_ct20"),
    ("B005_duodenum_ct25",         "B005_duodenum_ct25"),
    ("B005_ileum_ct25",            "B005_ileum_ct25"),
    ("B006_descendingSigmoid_ct23","B006_descendingSigmoid_ct23"),
    ("B006_descending_ct24",       "B006_descending_ct24"),
    ("B008_ascending_ct22",        "B008_ascending_ct22"),
    ("B008_transverse_ct23",       "B008_transverse_ct23"),
    ("B010_ascending_ct25",        "B010_ascending_ct25"),
    ("B011_ascending_ct24",        "B011_ascending_ct24"),
]

GLOBAL_CELL_TYPES = [
    'B','CD4+Tcell','CD57+Enterocyte','CD66+Enterocyte','CD7+Immune','CD8+T',
    'CyclingTA','DC','Endothelial','Enterocyte','Goblet','ICC','Lymphatic',
    'M1Macrophage','M2Macrophage','MUC1+Enterocyte','NK','Nerve','Neuroendocrine',
    'Neutrophil','Paneth','Plasma','Smoothmuscle','Stroma','TA'
]
GLOBAL_TYPE_ID = {ct: i for i, ct in enumerate(GLOBAL_CELL_TYPES)}

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
    def __init__(self, hidden_dim, num_layers, dropout, max_labels=MAX_LABELS, n_size_slots=2):
        super().__init__()
        self.max_labels   = max_labels
        self.n_size_slots = n_size_slots
        self.embed        = nn.Linear(max_labels, hidden_dim, bias=False)
        self.layers     = nn.ModuleList([GNNLayer(hidden_dim, dropout) for _ in range(num_layers)])
        self.cross_attn_p2g = nn.MultiheadAttention(hidden_dim, 4, dropout=dropout, batch_first=True)
        self.cross_attn_g2p = nn.MultiheadAttention(hidden_dim, 4, dropout=dropout, batch_first=True)
        self.norm_p    = nn.LayerNorm(hidden_dim)
        self.norm_g    = nn.LayerNorm(hidden_dim)
        self.gate_g    = nn.Linear(hidden_dim, hidden_dim)
        nn.init.constant_(self.gate_g.bias, -2.0)
        self.tri_embed = nn.Linear(1, hidden_dim, bias=False)
        # exp30+: size indicator; n_size_slots auto-detected from checkpoint
        self.size_embed = nn.Embedding(n_size_slots, hidden_dim)
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
        # exp30+: add size indicator if size_embed was loaded from checkpoint
        if getattr(self, 'size_embed', None) is not None:
            n_pat    = p_mask.float().sum(dim=1).long()
            size_idx = (n_pat - 3).clamp(0, self.n_size_slots - 1)
            p_pool   = p_pool + self.size_embed(size_idx)
        return self.predict(torch.cat([p_pool, g_pool, g_gated], dim=-1))


# ---- Helpers ----

def _load_remap(sample_dir, gml_stem):
    candidates = [
        os.path.join(INT_BASE, sample_dir, f"cell_type_to_id_{gml_stem}.csv"),
        os.path.join(INT_BASE, sample_dir, "cell_type_to_id.csv"),
    ]
    csv_path = next((c for c in candidates if os.path.exists(c)), None)
    if csv_path is None:
        return {}
    df = pd.read_csv(csv_path)
    return {int(row['cell_type_id']): GLOBAL_TYPE_ID.get(row['cell_type'], int(row['cell_type_id']))
            for _, row in df.iterrows()}


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


def eval_one_sample(model, sample_dir, gml_stem):
    remap   = _load_remap(sample_dir, gml_stem)
    gml_path = os.path.join(INT_BASE, sample_dir, f"{gml_stem}.gml")
    vf2_path = os.path.join(INT_BASE, sample_dir, f"{gml_stem}_vf2s3",
                            "Occurrence_number_size3.csv")
    if not os.path.exists(gml_path) or not os.path.exists(vf2_path):
        return None, "missing GML or VF2"

    g = ig.read(gml_path)
    n_nodes = g.vcount()
    node_labels = [remap.get(int(v['label']), int(v['label'])) for v in g.vs]
    neighbors = [[] for _ in range(n_nodes)]
    for u, v in g.get_edgelist():
        neighbors[u].append(v); neighbors[v].append(u)

    rng = random.Random(42)
    eval_nodes = rng.sample(range(n_nodes), min(N_EVAL_NODES, n_nodes))
    khop = [_khop_nodes(neighbors, v, K_HOP) for v in eval_nodes]

    g_adjs, g_feats, g_masks = [], [], []
    for node_list in khop:
        ga, gf, gm = _build_g_tensors(node_list, node_labels, neighbors)
        g_adjs.append(ga); g_feats.append(gf); g_masks.append(gm)
    G_ADJ  = torch.stack(g_adjs)
    G_FEAT = torch.stack(g_feats)
    G_MASK = torch.stack(g_masks)
    N_eval = len(eval_nodes)

    vf2_df   = pd.read_csv(vf2_path)
    all_patterns = [([remap.get(l, l) for l in json.loads(row['label'])],
                     int(row['occurrence_number']))
                    for _, row in vf2_df.iterrows()]
    # Sample balanced zero/nonzero patterns
    zero_pats    = [(lbl, cnt) for lbl, cnt in all_patterns if cnt == 0]
    nonzero_pats = [(lbl, cnt) for lbl, cnt in all_patterns if cnt > 0]
    rng2 = random.Random(99)
    n_nz  = min(N_EVAL_PATS // 2, len(nonzero_pats))
    n_z   = min(N_EVAL_PATS - n_nz, len(zero_pats))
    patterns = (rng2.sample(nonzero_pats, n_nz) + rng2.sample(zero_pats, n_z))

    our = []; vf2 = []
    with torch.no_grad():
        for lbl, vf2_cnt in patterns:
            pa, pf, pm = _build_p_tensors(lbl)
            PA = pa.unsqueeze(0).expand(N_eval, -1, -1)
            PF = pf.unsqueeze(0).expand(N_eval, -1, -1)
            PM = pm.unsqueeze(0).expand(N_eval, -1)
            preds = []
            for i in range(0, N_eval, BATCH_SIZE):
                out = model(PA[i:i+BATCH_SIZE], PF[i:i+BATCH_SIZE], PM[i:i+BATCH_SIZE],
                            G_ADJ[i:i+BATCH_SIZE], G_FEAT[i:i+BATCH_SIZE], G_MASK[i:i+BATCH_SIZE])
                preds.append(out)
            our.append(torch.cat(preds).mean().item() * n_nodes)
            vf2.append(float(vf2_cnt))

    rho = spearman(our, vf2)
    return rho, f"n={len(patterns)}, nodes={n_nodes}"


# ---- Main ----

print(f"Loading model from {CKPT_PATH} ...")
ckpt  = torch.load(CKPT_PATH, map_location='cpu')
state = ckpt['model_state']
se_key = 'size_embed.weight'
n_size_slots = state[se_key].shape[0] if se_key in state else 2
model = SubgraphGNN(ckpt['hidden_dim'], ckpt['num_layers'], ckpt['dropout'], n_size_slots=n_size_slots)
missing, unexpected = model.load_state_dict(state, strict=False)
if missing: print(f"  (missing keys: {missing})")
if unexpected: print(f"  (unexpected keys: {unexpected})")
# Disable size_embed if not in checkpoint (exp29 and earlier)
if se_key in missing:
    model.size_embed = None
model.eval()
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params  "
      f"(H={ckpt['hidden_dim']}, L={ckpt['num_layers']})")
print()

results = []
for sample_dir, gml_stem in EVAL_SAMPLES:
    t0 = time.time()
    print(f"Evaluating {gml_stem} ...", flush=True)
    rho, info = eval_one_sample(model, sample_dir, gml_stem)
    elapsed = time.time() - t0
    if rho is not None:
        print(f"  → Spearman={rho:.4f}  ({info})  {elapsed:.0f}s", flush=True)
        results.append((gml_stem, rho))
    else:
        print(f"  → SKIP: {info}", flush=True)

print()
print("=" * 60)
print("  MULTI-SAMPLE CROSS-GENERALIZATION SUMMARY")
print("=" * 60)
for name, rho in results:
    print(f"  {name:<40} Spearman={rho:.4f}")
if results:
    avg = sum(r for _, r in results) / len(results)
    print(f"  {'AVERAGE':<40} Spearman={avg:.4f}")
print("=" * 60)
print(f"\nTrimNN vs VF2 baseline: Spearman=0.4561 (B004_ascending benchmark)")
