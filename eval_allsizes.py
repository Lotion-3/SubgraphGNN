"""
Benchmark GNN model at ALL sizes (3-9) against VF2 ground truth on demo graph.

For each size:
  - Loads VF2 whole-graph counts from demo_data/vf2s3/, vf2s4/, vf2_sizes59/
  - Runs our GNN model on all patterns (using DEMO_LABEL_OFFSET for consistent features)
  - Reports Spearman(GNN, VF2) — comparison vs TrimNN for sizes 3-4

Important: demo graph labels (0-7) are mapped to slots (25-32) via DEMO_LABEL_OFFSET=25,
matching the training convention in train_trimnn.py (DemoVF2Data.DEMO_LABEL_OFFSET).

Usage:
  python eval_allsizes.py --model /path/to/model.pt [--sizes 3,4,5,6]
"""

import os, sys, argparse, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import igraph as ig
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
TRIMNN_DIR  = os.path.abspath(os.path.join(SCRIPT_DIR, "../TrimNN"))
DEMO_DIR    = os.path.join(TRIMNN_DIR, "demo_data")
GRAPH_PATH  = os.path.join(DEMO_DIR, "demo_data.gml")

# VF2 ground truth paths per size
VF2_PATHS = {
    3: os.path.join(DEMO_DIR, "vf2s3", "Occurrence_number_size3.csv"),
    4: os.path.join(DEMO_DIR, "vf2s4", "Occurrence_number_size4.csv"),
    **{s: os.path.join(DEMO_DIR, "vf2_sizes59", f"Occurrence_number_size{s}.csv")
       for s in range(5, 10)}
}

# TrimNN prediction paths for sizes 3-4 (demo graph only — sizes 5-9 not available for demo)
TRIMNN_PRED_PATHS = {
    3: os.path.join(TRIMNN_DIR, "sparse_function3", "Predicted_occurrence_size3.csv"),
    4: os.path.join(TRIMNN_DIR, "sparse_function3", "Predicted_occurrence_size4.csv"),
}

# Topology edge definitions (matching gen_vf2_fast.py and train_trimnn.py)
SIZE_EDGES = {
    3: [(0,1),(0,2),(1,2)],
    4: [(0,1),(0,2),(1,2),(1,3),(2,3)],
    5: [(0,1),(0,2),(1,2),(1,3),(2,3),(1,4),(3,4)],
    6: [(0,1),(0,2),(1,2),(1,3),(2,3),(1,4),(3,4),(2,5),(3,5)],
    7: [(0,1),(0,2),(1,2),(1,3),(2,3),(1,4),(3,4),(2,5),(3,5),(4,6),(3,6)],
    8: [(0,1),(0,2),(1,2),(1,3),(2,3),(1,4),(3,4),(2,5),(3,5),(4,6),(3,6),(4,7),(6,7)],
    9: [(0,1),(0,2),(1,2),(1,3),(2,3),(1,4),(3,4),(2,5),(3,5),(4,6),(3,6),(4,7),(6,7),(5,8),(3,8)],
}

MAX_GRAPH_NODES = 64
MAX_PAT_NODES   = 9       # accommodate up to size-9 patterns
MAX_LABELS      = 40      # must match train_trimnn.py MAX_LABELS
DEMO_LABEL_OFFSET = 25    # must match train_trimnn.py DemoVF2Data.DEMO_LABEL_OFFSET
K_HOP           = 2

# ---------------------------------------------------------------------------
# Exact SubgraphGNN from train_trimnn.py (keep in sync!)
# ---------------------------------------------------------------------------
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
    def __init__(self, hidden_dim, num_layers, dropout, max_labels=MAX_LABELS, n_size_slots=8):
        super().__init__()
        self.max_labels  = max_labels
        self.n_size_slots = n_size_slots
        self.embed      = nn.Linear(max_labels, hidden_dim, bias=False)
        self.layers     = nn.ModuleList(
            [GNNLayer(hidden_dim, dropout) for _ in range(num_layers)])
        self.cross_attn_p2g = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.cross_attn_g2p = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm_p    = nn.LayerNorm(hidden_dim)
        self.norm_g    = nn.LayerNorm(hidden_dim)
        self.gate_g    = nn.Linear(hidden_dim, hidden_dim)
        nn.init.constant_(self.gate_g.bias, -2.0)
        self.tri_embed = nn.Linear(1, hidden_dim, bias=False)
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

    def encode_g(self, g_adj, g_feat, g_mask):
        """Graph-only encoding — reusable across patterns."""
        g_enc = self._encode(g_adj, g_feat)
        A2_g  = torch.bmm(g_adj, g_adj)
        tri_c = (A2_g * g_adj).sum(dim=-1, keepdim=True) / 2
        g_enc = g_enc + self.tri_embed(torch.log1p(tri_c))
        return g_enc  # (B, MAX_GRAPH, hidden)

    def forward_with_g_enc(self, p_adj, p_feat, p_mask, g_enc, g_mask):
        """Fast forward reusing precomputed g_enc (skips g GNN pass)."""
        p_enc    = self._encode(p_adj, p_feat)
        p_mask_f = p_mask.unsqueeze(-1).float()
        g_mask_f = g_mask.unsqueeze(-1).float()

        p_attn, _ = self.cross_attn_p2g(
            self.norm_p(p_enc), self.norm_g(g_enc), self.norm_g(g_enc),
            key_padding_mask=~g_mask)
        p_out = p_enc + p_attn

        g_attn, _ = self.cross_attn_g2p(
            self.norm_g(g_enc), self.norm_p(p_enc), self.norm_p(p_enc),
            key_padding_mask=~p_mask)
        g_out = g_enc + g_attn

        p_pool  = (p_out * p_mask_f).max(dim=1).values
        g_pool  = (g_out * g_mask_f).max(dim=1).values
        gate    = torch.sigmoid(self.gate_g(g_out))
        g_gated = (gate * g_enc * g_mask_f).sum(dim=1)

        # size_embed: distinguish pattern sizes (matches train_trimnn.py logic)
        n_pat_nodes = p_mask.float().sum(dim=1).long()
        size_idx    = (n_pat_nodes - 3).clamp(0, self.n_size_slots - 1)
        p_pool      = p_pool + self.size_embed(size_idx)

        return self.predict(torch.cat([p_pool, g_pool, g_gated], dim=-1))

    def forward(self, p_adj, p_feat, p_mask, g_adj, g_feat, g_mask):
        g_enc = self.encode_g(g_adj, g_feat, g_mask)
        return self.forward_with_g_enc(p_adj, p_feat, p_mask, g_enc, g_mask)


# ---------------------------------------------------------------------------
# Tensor builders (matching train_trimnn.py conventions)
# ---------------------------------------------------------------------------
def build_p_tensors(label_list, edges, max_pat=MAX_PAT_NODES, max_labels=MAX_LABELS):
    """Build pattern adjacency, feature, mask tensors with DEMO_LABEL_OFFSET applied."""
    adj  = torch.zeros(max_pat, max_pat)
    for u, v in edges:
        if u < max_pat and v < max_pat:
            adj[u, v] = adj[v, u] = 1.0
    feat = torch.zeros(max_pat, max_labels)
    for i, lbl in enumerate(label_list[:max_pat]):
        slot = int(lbl) % max_labels
        feat[i, slot] = 1.0
    mask = torch.zeros(max_pat, dtype=torch.bool)
    mask[:min(len(label_list), max_pat)] = True
    return adj, feat, mask


def build_g_tensors(node_list, node_labels, neighbors_map,
                    max_nodes=MAX_GRAPH_NODES, max_labels=MAX_LABELS):
    """Build graph subgraph tensors with DEMO_LABEL_OFFSET applied."""
    n        = min(len(node_list), max_nodes)
    node_idx = {v: i for i, v in enumerate(node_list)}
    adj  = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    feat = np.zeros((max_nodes, max_labels), dtype=np.float32)
    mask = np.zeros((max_nodes,),            dtype=bool)
    for i, v in enumerate(node_list[:n]):
        lbl  = int(node_labels[v]) % max_labels
        feat[i, lbl] = 1.0
        mask[i] = True
        for nb in neighbors_map[v]:
            j = node_idx.get(nb)
            if j is not None and j < n:
                adj[i, j] = adj[j, i] = 1.0
    return (torch.from_numpy(adj),
            torch.from_numpy(feat),
            torch.from_numpy(mask))


def khop_nodes(neighbors, center, k):
    visited = {center}; frontier = {center}
    for _ in range(k):
        nf = set()
        for v in frontier:
            for nb in neighbors[v]:
                if nb not in visited:
                    nf.add(nb); visited.add(nb)
        frontier = nf
    return sorted(visited)


def _precompute_g_tensors(n_nodes, node_labels, neighbors, khops, min_size):
    """Precompute graph tensors for all nodes once.

    Returns (all_ga, all_gf, all_gm, valid_mask) as stacked tensors.
    valid_mask[i] = True iff khop[i] has >= min_size nodes.
    """
    adjs, feats, masks, valid = [], [], [], []
    for center in range(n_nodes):
        nl = khops[center]
        if len(nl) < min_size:
            adjs.append(torch.zeros(MAX_GRAPH_NODES, MAX_GRAPH_NODES))
            feats.append(torch.zeros(MAX_GRAPH_NODES, MAX_LABELS))
            masks.append(torch.zeros(MAX_GRAPH_NODES, dtype=torch.bool))
            valid.append(False)
        else:
            ga, gf, gm = build_g_tensors(nl, node_labels, neighbors)
            adjs.append(ga); feats.append(gf); masks.append(gm)
            valid.append(True)
    return (torch.stack(adjs),
            torch.stack(feats),
            torch.stack(masks),
            np.array(valid, dtype=bool))


@torch.no_grad()
def precompute_g_enc(model, precomp_g, device, chunk=256):
    """Precompute g_enc for ALL nodes using model.encode_g().
    Returns g_encs: (n_nodes, MAX_GRAPH, hidden) and g_masks: (n_nodes, MAX_GRAPH).
    """
    all_ga, all_gf, all_gm, valid_mask = precomp_g
    n = all_ga.shape[0]
    hidden = model.embed.out_features
    g_encs = torch.zeros(n, MAX_GRAPH_NODES, hidden)
    for ci in range(0, n, chunk):
        sl = slice(ci, ci + chunk)
        ga_b = all_ga[sl].to(device)
        gf_b = all_gf[sl].to(device)
        gm_b = all_gm[sl].to(device)
        enc = model.encode_g(ga_b, gf_b, gm_b)
        g_encs[sl] = enc.cpu()
    return g_encs, all_gm  # g_encs: (n, MAX_GRAPH, H), masks: (n, MAX_GRAPH)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_path, device):
    """Load checkpoint saved by train_trimnn.py."""
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict):
        state = ckpt.get('model_state', ckpt.get('model_state_dict', ckpt))
        hidden_dim = ckpt.get('hidden_dim', 128)
        num_layers = ckpt.get('num_layers', 3)
        dropout    = ckpt.get('dropout', 0.0)
    else:
        state = ckpt
        hidden_dim, num_layers, dropout = 128, 3, 0.0

    # Detect size_embed slots from checkpoint to handle old (Embedding(2)) vs new (Embedding(8))
    se_key = 'size_embed.weight'
    n_size_slots = state[se_key].shape[0] if se_key in state else 8
    model = SubgraphGNN(hidden_dim, num_layers, dropout, n_size_slots=n_size_slots).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] Missing keys: {missing[:5]}...")
    if unexpected:
        print(f"  [warn] Unexpected keys: {unexpected[:5]}...")
    model.eval()
    return model, hidden_dim


# ---------------------------------------------------------------------------
# Demo graph loading
# ---------------------------------------------------------------------------
def load_demo_graph():
    g = ig.read(GRAPH_PATH)
    g.vs["label"] = [int(x) for x in g.vs["label"]]
    n = g.vcount()
    # Build adjacency as Python lists for fast k-hop
    neighbors = [[] for _ in range(n)]
    for u, v in g.get_edgelist():
        neighbors[u].append(v); neighbors[v].append(u)
    # Apply DEMO_LABEL_OFFSET to node labels (matches DemoVF2Data convention)
    node_labels = [int(v['label']) + DEMO_LABEL_OFFSET for v in g.vs]
    # Precompute all k-hop neighborhoods
    print(f"  Precomputing k-hop (k={K_HOP}) for {n} nodes...")
    khops = [khop_nodes(neighbors, v, K_HOP) for v in range(n)]
    return n, node_labels, neighbors, khops


# ---------------------------------------------------------------------------
# Evaluate one size
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_size(size, model, n_nodes, node_labels, neighbors, khops, device,
                  chunk=128, max_pats=None, precomp_g=None, g_encs=None):
    """Evaluate GNN vs VF2 Spearman for one pattern size.

    precomp_g: (all_ga, all_gf, all_gm, valid_mask) — raw graph tensors (fallback)
    g_encs: (enc_tensor, mask_tensor) — precomputed g encodings, skips g GNN pass
    """
    vf2_path = VF2_PATHS.get(size)
    if not vf2_path or not os.path.exists(vf2_path):
        return None

    df = pd.read_csv(vf2_path)
    count_col = ("occurrence_number" if "occurrence_number" in df.columns
                 else "predicted_occurrence_number")
    vf2_counts = df[count_col].values.astype(float)

    # Build pattern list from CSV labels + topology edges (with DEMO_LABEL_OFFSET)
    edges = SIZE_EDGES[size]
    pat_labels  = []
    raw_labels  = []  # original label strings for TrimNN alignment
    for _, row in df.iterrows():
        lbl_raw = json.loads(row["label"])
        pat_labels.append([l + DEMO_LABEL_OFFSET for l in lbl_raw])
        raw_labels.append(str(row["label"]))

    # Optionally subsample patterns (preserving nonzero fraction)
    if max_pats is not None and len(pat_labels) > max_pats:
        nz_idx  = [i for i, c in enumerate(vf2_counts) if c > 0]
        z_idx   = [i for i, c in enumerate(vf2_counts) if c == 0]
        rng = np.random.default_rng(42)
        n_nz_keep = min(len(nz_idx), max_pats // 2)
        n_z_keep  = max_pats - n_nz_keep
        keep = (list(rng.choice(nz_idx, n_nz_keep, replace=False)) +
                list(rng.choice(z_idx,  min(n_z_keep, len(z_idx)), replace=False)))
        keep.sort()
        pat_labels  = [pat_labels[i]  for i in keep]
        raw_labels  = [raw_labels[i]  for i in keep]
        vf2_counts  = vf2_counts[keep]

    n_pats = len(pat_labels)

    # Determine valid nodes
    if precomp_g is None:
        precomp_g = _precompute_g_tensors(n_nodes, node_labels, neighbors, khops, size)
    all_ga, all_gf, all_gm, valid_mask = precomp_g
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)

    # Use precomputed g_enc if available (skips g GNN pass — major speedup)
    if g_encs is not None:
        all_enc, all_mask = g_encs
        ve  = all_enc[valid_indices]   # (n_valid, MAX_GRAPH, hidden)
        vm  = all_mask[valid_indices]  # (n_valid, MAX_GRAPH)
        use_enc = True
    else:
        va = all_ga[valid_indices]
        vf = all_gf[valid_indices]
        vm = all_gm[valid_indices]
        use_enc = False

    gnn_preds = []
    t0 = time.time()

    for pi, lbl in enumerate(pat_labels):
        p_adj, p_feat, p_mask = build_p_tensors(lbl, edges)
        pa = p_adj.to(device)
        pf = p_feat.to(device)
        pm = p_mask.to(device)

        node_scores = np.zeros(n_nodes, dtype=np.float32)
        for ci in range(0, n_valid, chunk):
            sl = slice(ci, ci + chunk)
            B  = vm[sl].shape[0]
            pa_b = pa.unsqueeze(0).expand(B, -1, -1)
            pf_b = pf.unsqueeze(0).expand(B, -1, -1)
            pm_b = pm.unsqueeze(0).expand(B, -1)
            if use_enc:
                out = model.forward_with_g_enc(
                    pa_b, pf_b, pm_b,
                    ve[sl].to(device), vm[sl].to(device))
            else:
                out = model(pa_b, pf_b, pm_b,
                            va[sl].to(device), vf[sl].to(device), vm[sl].to(device))
            node_scores[valid_indices[sl]] = out.cpu().view(-1).numpy()

        gnn_preds.append(float(node_scores.sum()))

        if (pi + 1) % 50 == 0 or (pi + 1) == n_pats:
            elapsed = time.time() - t0
            print(f"    pattern {pi+1}/{n_pats} | {elapsed:.0f}s", end="\r", flush=True)

    print()
    gnn_preds = np.array(gnn_preds)

    # Spearman correlation (all patterns)
    spear_all = spearmanr(gnn_preds, vf2_counts).statistic

    # Spearman on nonzero-VF2 subset
    nz_mask = vf2_counts > 0
    n_nz    = int(nz_mask.sum())
    spear_nz = (spearmanr(gnn_preds[nz_mask], vf2_counts[nz_mask]).statistic
                if n_nz > 1 else float('nan'))

    # TrimNN baseline (sizes 3-4 only)
    trimnn_spear = None
    tp = TRIMNN_PRED_PATHS.get(size)
    if tp and os.path.exists(tp):
        df_t = pd.read_csv(tp)
        # Align by label string (use raw_labels — already subsampled to match vf2_counts)
        trim_dict   = dict(zip(df_t["label"].astype(str),
                               df_t["predicted_occurrence_number"].astype(float)))
        trim_counts = np.array([trim_dict.get(l, 0.0) for l in raw_labels])
        trimnn_spear = spearmanr(trim_counts, vf2_counts).statistic

    return {
        "size": size,
        "n_patterns": n_pats,
        "nonzero_vf2": n_nz,
        "gnn_spearman_all": float(spear_all),
        "gnn_spearman_nz":  float(spear_nz),
        "trimnn_spearman":  float(trimnn_spear) if trimnn_spear is not None else None,
        "beat_trimnn": (trimnn_spear is not None and spear_all > trimnn_spear),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True, help="Path to model .pt checkpoint")
    parser.add_argument("--sizes",    default="3,4,5,6", help="Comma-separated sizes")
    parser.add_argument("--device",   default="cpu")
    parser.add_argument("--max_pats", type=int, default=300,
                        help="Max patterns per size (subsample if more; 0=all)")
    args = parser.parse_args()

    sizes    = [int(s) for s in args.sizes.split(",")]
    device   = torch.device(args.device)
    max_pats = args.max_pats if args.max_pats > 0 else None

    print(f"Loading model from {args.model}...")
    model, hidden_dim = load_model(args.model, device)
    print(f"  hidden_dim={hidden_dim}, device={device}")

    print(f"\nLoading demo graph (DEMO_LABEL_OFFSET={DEMO_LABEL_OFFSET})...")
    n_nodes, node_labels, neighbors, khops = load_demo_graph()
    print(f"  {n_nodes} nodes")

    # Precompute g tensors once — reused across all sizes
    min_size = min(sizes)
    print(f"  Precomputing graph tensors for all {n_nodes} nodes...")
    precomp_g = _precompute_g_tensors(n_nodes, node_labels, neighbors, khops, min_size)
    print(f"  {int(precomp_g[3].sum())} valid nodes (>= {min_size} in k-hop)")

    # Precompute g_enc (GNN pass over graph) — shared across ALL patterns/sizes
    print(f"  Precomputing g_enc (GNN graph encoding)...")
    t_enc = time.time()
    g_encs = precompute_g_enc(model, precomp_g, device)
    print(f"  Done in {time.time()-t_enc:.1f}s")

    print("\n" + "="*72)
    print("  ALL-SIZES BENCHMARK: GNN vs VF2 ground truth (demo graph)")
    print("="*72)

    results = []
    for size in sizes:
        print(f"\n--- Size {size} ---")
        r = evaluate_size(size, model, n_nodes, node_labels, neighbors, khops, device,
                          max_pats=max_pats, precomp_g=precomp_g, g_encs=g_encs)
        if r is None:
            print(f"  Skipped (no VF2 data)")
            continue
        results.append(r)
        gnn_s  = f"{r['gnn_spearman_all']:.4f}"
        nz_s   = f"{r['gnn_spearman_nz']:.4f}" if not np.isnan(r['gnn_spearman_nz']) else "  N/A"
        trim_s = f"{r['trimnn_spearman']:.4f}" if r['trimnn_spearman'] is not None else "  N/A"
        beat   = "✓ BEAT" if r['beat_trimnn'] else ("  N/A" if r['trimnn_spearman'] is None else "✗")
        print(f"  Spearman(all) = {gnn_s}  nonzero = {nz_s}  TrimNN = {trim_s}  [{beat}]")
        print(f"  Nonzero patterns: {r['nonzero_vf2']}/{r['n_patterns']}")

    print("\n" + "="*72)
    print(f"  {'Size':>5}  {'GNN(all)':>9}  {'GNN(nz)':>9}  {'TrimNN':>9}  {'Beat?':>7}")
    print(f"  {'-'*5}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*7}")
    for r in results:
        gnn_s  = f"{r['gnn_spearman_all']:+.4f}"
        nz_s   = (f"{r['gnn_spearman_nz']:+.4f}" if not np.isnan(r['gnn_spearman_nz'])
                  else "      N/A")
        trim_s = (f"{r['trimnn_spearman']:+.4f}" if r['trimnn_spearman'] is not None
                  else "      N/A")
        beat   = "✓" if r['beat_trimnn'] else ("-" if r['trimnn_spearman'] is None else "✗")
        print(f"  {r['size']:>5}  {gnn_s:>9}  {nz_s:>9}  {trim_s:>9}  {beat:>7}")
    beats = sum(1 for r in results if r['beat_trimnn'])
    with_trimnn = sum(1 for r in results if r['trimnn_spearman'] is not None)
    print(f"\n  Beat TrimNN on demo graph: {beats}/{with_trimnn} sizes (sizes 3-4 only)")
    print("="*72)


if __name__ == "__main__":
    main()
