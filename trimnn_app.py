#!/usr/bin/env python3
"""
TrimNN Replacement Application — Functions 1, 2, 3
Uses our trained GNN model to replicate TrimNN's three core functions.

Function 1: subgraph_matching   — predict occurrence of a specific CC motif
Function 2: specific_size        — find top overrepresented motifs of a fixed size
Function 3: all_size             — greedy growth from size-3 to target size

Input: GML files OR CSV files (with X, Y, cell_type columns → auto Delaunay triangulation)

Usage examples:
  python trimnn_app.py --function subgraph_matching \\
      --motif pattern.gml --target graph.gml --outpath results/

  python trimnn_app.py --function specific_size \\
      --size 3 --target graph.gml --celltype 8 --outpath results/

  python trimnn_app.py --function all_size \\
      --size 4 --target graph.gml --celltype 8 --outpath results/ --search greedy
"""

import os, sys, json, math, time, random, argparse, itertools
from collections import defaultdict
import numpy as np
import pandas as pd
import igraph as ig
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import Delaunay
try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

# ── Constants ────────────────────────────────────────────────────────────────
MAX_LABELS      = 32
MAX_PAT_NODES   = 9     # support sizes 3–9 (TrimNN's full range)
MAX_GRAPH_NODES = 48
K_HOP           = 2
PRED_BATCH      = 128
EVAL_NODES      = 500   # nodes sampled for count estimation
DEFAULT_MODEL   = os.path.join(os.path.dirname(__file__), "trimnn_model.pt")

# TrimNN's canonical base topology per size (used by Function 2)
BASE_EDGES = {
    3: [(0,1),(0,2),(1,2)],
    4: [(0,1),(0,2),(1,2),(1,3),(2,3)],
    5: [(0,1),(0,2),(1,2),(0,3),(0,4),(3,4)],
    6: [(0,1),(0,2),(1,2),(2,3),(2,4),(3,4),(4,5),(3,5)],
    7: [(0,1),(0,2),(1,2),(2,3),(2,4),(3,4),(3,5),(3,6),(5,6)],
    8: [(0,1),(0,2),(1,2),(2,3),(2,4),(3,4),(3,5),(3,6),(5,6),(5,7),(6,7)],
    9: [(0,1),(0,2),(1,2),(2,3),(2,4),(3,4),(3,5),(3,6),(6,7),(6,8),(7,8)],
}

# ── Model (SubgraphGNN — matches train_trimnn.py) ────────────────────────────

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
    def __init__(self, hidden_dim=128, num_layers=3, dropout=0.0, max_labels=MAX_LABELS):
        super().__init__()
        self.max_labels = max_labels
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

        p_mask_f   = p_mask.unsqueeze(-1).float()
        g_mask_f   = g_mask.unsqueeze(-1).float()
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


# ── Data helpers ─────────────────────────────────────────────────────────────

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


def build_g_tensors(node_list, node_labels, neighbors_map,
                    max_nodes=MAX_GRAPH_NODES, max_labels=MAX_LABELS):
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
                adj[i, j] = adj[j, i] = 1.0
    return (torch.from_numpy(adj),
            torch.from_numpy(feat),
            torch.from_numpy(mask))


def build_p_tensors(labels, edges, max_pat=MAX_PAT_NODES, max_labels=MAX_LABELS):
    """
    Build pattern tensors from a label list and edge list.
    labels: list of int (node labels)
    edges:  list of (i, j) tuples
    """
    n = len(labels)
    p_adj  = torch.zeros(max_pat, max_pat)
    p_feat = torch.zeros(max_pat, max_labels)
    p_mask = torch.zeros(max_pat, dtype=torch.bool)

    for i in range(min(n, max_pat)):
        lbl = int(labels[i]) % max_labels
        p_feat[i, lbl] = 1.0
        p_mask[i] = True

    for u, v in edges:
        if u < max_pat and v < max_pat:
            p_adj[u, v] = p_adj[v, u] = 1.0

    return p_adj, p_feat, p_mask


def ig_to_tensors(pattern_ig):
    """Convert igraph pattern to (labels, edges) for build_p_tensors."""
    labels = [int(x) for x in pattern_ig.vs['label']]
    edges  = pattern_ig.get_edgelist()
    return labels, edges


# ── Graph loading ─────────────────────────────────────────────────────────────

def load_graph(path, prune=True):
    """
    Load a cellular community graph from GML or CSV.
    Returns: (node_labels, neighbors, khop, n_nodes, n_celltypes, type_to_id)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        return _load_from_csv(path, prune)
    else:
        return _load_from_gml(path)


def _load_from_csv(path, prune=True):
    """CSV with X, Y, cell_type columns → Delaunay triangulation → graph."""
    df = pd.read_csv(path)
    required = {'X', 'Y', 'cell_type'}
    if not required.issubset(df.columns):
        # Try lowercase
        df.columns = [c.strip() for c in df.columns]
        if not required.issubset(df.columns):
            raise ValueError(f"CSV must have X, Y, cell_type columns. Got: {list(df.columns)}")

    unique_types = sorted(df['cell_type'].unique())
    type_to_id   = {t: i for i, t in enumerate(unique_types)}
    labels       = [type_to_id[ct] for ct in df['cell_type']]
    coords       = df[['X', 'Y']].values.astype(float)
    n_celltypes  = len(unique_types)

    tri   = Delaunay(coords)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                u, v = int(simplex[i]), int(simplex[j])
                edges.add((min(u, v), max(u, v)))

    if prune and edges:
        lengths   = [np.linalg.norm(coords[u] - coords[v]) for u, v in edges]
        threshold = np.percentile(lengths, 99)
        edges     = {e for e, l in zip(edges, lengths) if l <= threshold}

    g = ig.Graph()
    g.add_vertices(len(labels))
    g.vs['label'] = labels
    g.add_edges(list(edges))

    return _process_ig(g), n_celltypes, type_to_id


def _load_from_gml(path):
    """Load a GML file produced by TrimNN's csv2gml.py or our own pipeline."""
    g           = ig.read(path)
    labels      = [int(float(x)) for x in g.vs['label']]
    g.vs['label'] = labels
    n_celltypes = max(labels) + 1
    return _process_ig(g), n_celltypes, None


def _process_ig(g):
    """Extract node_labels, neighbors list, k-hop subgraphs, n_nodes."""
    n           = g.vcount()
    node_labels = [int(float(x)) for x in g.vs['label']]
    neighbors   = [[] for _ in range(n)]
    for u, v in g.get_edgelist():
        neighbors[u].append(v); neighbors[v].append(u)
    print(f"  Building {K_HOP}-hop subgraphs for {n:,} nodes …", flush=True)
    khop = [khop_nodes(neighbors, v, K_HOP) for v in range(n)]
    return node_labels, neighbors, khop, n


# ── Pattern / motif loading ───────────────────────────────────────────────────

def load_pattern_gml(path):
    """Load a pattern GML; return igraph.Graph with 'label' vertex attribute."""
    p = ig.read(path)
    p.vs['label'] = [int(float(x)) for x in p.vs['label']]
    return p


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_path=None):
    if model_path is None:
        model_path = DEFAULT_MODEL
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    ckpt  = torch.load(model_path, map_location='cpu')
    model = SubgraphGNN(
        hidden_dim  = ckpt.get('hidden_dim', 128),
        num_layers  = ckpt.get('num_layers', 3),
        dropout     = ckpt.get('dropout',    0.0),
    )
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model


# ── Count prediction ─────────────────────────────────────────────────────────

def _precompute_graph_tensors(graph_info, n_sample=EVAL_NODES, seed=0):
    node_labels, neighbors, khop, n_nodes = graph_info
    rng          = random.Random(seed)
    sample_nodes = rng.sample(range(n_nodes), min(n_sample, n_nodes))

    g_adjs, g_feats, g_masks = [], [], []
    for cn in sample_nodes:
        ga, gf, gm = build_g_tensors(khop[cn], node_labels, neighbors)
        g_adjs.append(ga); g_feats.append(gf); g_masks.append(gm)

    return (torch.stack(g_adjs),
            torch.stack(g_feats),
            torch.stack(g_masks),
            len(sample_nodes),
            n_nodes)


def predict_count(model, pattern_ig, G_ADJ, G_FEAT, G_MASK, n_sample, n_nodes,
                  device=torch.device('cpu')):
    """
    Predict whole-graph occurrence count for pattern_ig.
    Uses pre-computed graph tensors (call _precompute_graph_tensors once).
    """
    labels, edges = ig_to_tensors(pattern_ig)
    pa, pf, pm    = build_p_tensors(labels, edges)

    preds = []
    with torch.no_grad():
        for i in range(0, n_sample, PRED_BATCH):
            b   = min(PRED_BATCH, n_sample - i)
            PA  = pa.unsqueeze(0).expand(b, -1, -1).to(device)
            PF  = pf.unsqueeze(0).expand(b, -1, -1).to(device)
            PM  = pm.unsqueeze(0).expand(b, -1).to(device)
            out = model(PA, PF, PM,
                        G_ADJ[i:i+b].to(device),
                        G_FEAT[i:i+b].to(device),
                        G_MASK[i:i+b].to(device))
            preds.append(out.cpu())

    preds = torch.cat(preds).view(-1)
    return preds.mean().item() * n_nodes


# ── Pattern enumeration ───────────────────────────────────────────────────────

def _make_ig(n, edges, labels):
    """Create a labelled igraph Graph."""
    g = ig.Graph(n=n, edges=edges)
    g.vs['label'] = list(labels)
    g.es['label'] = 0
    return g


def _is_planar(n, edges):
    if not HAS_NX:
        return True      # skip planarity check if networkx unavailable
    G = nx.Graph(); G.add_edges_from(edges)
    return nx.is_planar(G)


def _is_dup(new_ig, seen_list):
    nl = new_ig.vs['label']
    for ex in seen_list:
        if new_ig.isomorphic_vf2(ex, color1=nl, color2=ex.vs['label']):
            return True
    return False


def enumerate_size3_patterns(n_celltypes):
    """All non-isomorphic labelled triangles (K3, only topology for size-3)."""
    base_edges = BASE_EDGES[3]
    patterns   = []
    seen       = []
    for labels in itertools.product(range(n_celltypes), repeat=3):
        g = _make_ig(3, base_edges, labels)
        if not _is_dup(g, seen):
            seen.append(g)
            patterns.append(g)
    return patterns


def enumerate_patterns_for_size(size, n_celltypes):
    """
    All non-isomorphic labelled patterns for a given size using TrimNN's
    canonical base topology for that size.  Equivalent to TrimNN Function 2.
    """
    if size not in BASE_EDGES:
        raise ValueError(f"Size {size} not supported (choose 3–9)")

    base_edges = BASE_EDGES[size]
    patterns   = []
    seen       = []
    total      = n_celltypes ** size
    t0         = time.time()

    print(f"  Enumerating size-{size} patterns "
          f"({n_celltypes}^{size}={total:,} label combos, deduplicated by isomorphism)…",
          flush=True)

    for i, labels in enumerate(itertools.product(range(n_celltypes), repeat=size)):
        g = _make_ig(size, base_edges, labels)
        if not _is_dup(g, seen):
            seen.append(g)
            patterns.append(g)
        if (i + 1) % max(1, total // 20) == 0:
            print(f"    {i+1:,}/{total:,} combos → {len(patterns)} unique patterns "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print(f"  Done: {len(patterns)} unique patterns  [{time.time()-t0:.1f}s]", flush=True)
    return patterns


def generate_candidate_extensions(current_ig, n_celltypes):
    """
    TrimNN Function 3 growth step: add one new node to current_ig.
    New node can connect to any non-empty subset of existing nodes.
    Filter: planar only.  Deduplicate by labelled isomorphism.
    Returns list of igraph.Graph candidates.
    """
    n_base     = current_ig.vcount()
    base_edges = current_ig.get_edgelist()
    candidates = []
    seen       = []

    for new_label in range(n_celltypes):
        for r in range(1, n_base + 1):
            for subset in itertools.combinations(range(n_base), r):
                new_edges = base_edges + [(i, n_base) for i in subset]
                if not _is_planar(n_base + 1, new_edges):
                    continue
                base_labels = list(current_ig.vs['label'])
                new_labels  = base_labels + [new_label]
                g = _make_ig(n_base + 1, new_edges, new_labels)
                if not _is_dup(g, seen):
                    seen.append(g)
                    candidates.append(g)

    return candidates


# ── Output helpers ───────────────────────────────────────────────────────────

def _pattern_motif_str(ig_pat):
    """Reproduce TrimNN's 'motif' column format (igraph summary string with edge list)."""
    n = ig_pat.vcount()
    m = ig_pat.ecount()
    edge_str = ' '.join(f"{e.source}--{e.target}" for e in ig_pat.es)
    return (f"IGRAPH U--- {n} {m} --\n"
            f"+ attr: type (g), label (v), label (e)\n"
            f"+ edges:\n"
            f"{edge_str}")


def save_size_results(results, outpath, size):
    """
    Save results for a given size:
      - Predicted_occurrence_sizeN.csv  (motif | label | predicted_occurrence_number)
      - Overrepresented_sizeN.gml       (best pattern as GML)
    """
    os.makedirs(outpath, exist_ok=True)
    rows = []
    for ig_pat, count in results:
        rows.append({
            'motif':                      _pattern_motif_str(ig_pat),
            'label':                      str(ig_pat.vs['label']),
            'predicted_occurrence_number': count,
        })

    df = pd.DataFrame(rows, columns=['motif', 'label', 'predicted_occurrence_number'])
    df.sort_values('predicted_occurrence_number', ascending=False, inplace=True)
    csv_path = os.path.join(outpath, f"Predicted_occurrence_size{size}.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}  ({len(df)} patterns)")

    if results:
        best_ig  = max(results, key=lambda x: x[1])[0]
        gml_path = os.path.join(outpath, f"Overrepresented_size{size}.gml")
        best_ig.write(gml_path, format='gml')
        print(f"  Saved {gml_path}")

    return df


# ── Function 1: Subgraph Matching ─────────────────────────────────────────────

def function1_subgraph_matching(args):
    """
    Predict the occurrence number of a specific CC motif in the target graph.
    Mirrors: python TrimNN.py -function subgraph_matching -motif X -target Y -outpath Z
    """
    print("[Function 1] Subgraph Matching")
    model = load_model(args.model)

    print(f"  Loading target graph: {args.target}")
    graph_info, n_celltypes, _ = load_graph(args.target, prune=args.prune)
    node_labels, neighbors, khop, n_nodes = graph_info

    print(f"  Graph: {n_nodes:,} nodes, {n_celltypes} cell types")
    print(f"  Loading motif: {args.motif}")
    pattern_ig = load_pattern_gml(args.motif)
    print(f"  Motif: {pattern_ig.vcount()} nodes, {pattern_ig.ecount()} edges, "
          f"labels={pattern_ig.vs['label']}")

    print(f"  Precomputing graph tensors ({EVAL_NODES} sampled nodes)…")
    G_ADJ, G_FEAT, G_MASK, n_sample, _ = _precompute_graph_tensors(graph_info)

    count = predict_count(model, pattern_ig, G_ADJ, G_FEAT, G_MASK, n_sample, n_nodes)
    print(f"\n  Predicted occurrence number: {count:.1f}")

    os.makedirs(args.outpath, exist_ok=True)
    out_path = os.path.join(args.outpath, "predicted_result.txt")
    with open(out_path, 'w') as f:
        f.write(f"Predicted occurrence number: {count:.1f}\n")
        f.write(f"Motif: {args.motif}\n")
        f.write(f"Target: {args.target}\n")
    print(f"  Saved {out_path}")


# ── Function 2: Specific Size ─────────────────────────────────────────────────

def function2_specific_size(args):
    """
    Enumerate all labelled CC motifs of a fixed size and predict occurrences.
    Mirrors: python TrimNN.py -function specific_size -size N -celltype C -target Y -outpath Z
    """
    print(f"[Function 2] Specific Size (size={args.size})")
    model = load_model(args.model)

    print(f"  Loading target graph: {args.target}")
    graph_info, n_celltypes_graph, _ = load_graph(args.target, prune=args.prune)
    node_labels, neighbors, khop, n_nodes = graph_info
    n_celltypes = args.celltype if args.celltype else n_celltypes_graph
    print(f"  Graph: {n_nodes:,} nodes, using {n_celltypes} cell types")

    print(f"  Precomputing graph tensors ({EVAL_NODES} sampled nodes)…")
    G_ADJ, G_FEAT, G_MASK, n_sample, _ = _precompute_graph_tensors(graph_info)

    patterns = enumerate_patterns_for_size(args.size, n_celltypes)
    print(f"  Predicting occurrence for {len(patterns)} patterns…")

    results = []
    t0 = time.time()
    for i, ig_pat in enumerate(patterns):
        count = predict_count(model, ig_pat, G_ADJ, G_FEAT, G_MASK, n_sample, n_nodes)
        results.append((ig_pat, count))
        if (i + 1) % max(1, len(patterns) // 10) == 0 or (i + 1) == len(patterns):
            best_so_far = max(results, key=lambda x: x[1])[1]
            print(f"    {i+1}/{len(patterns)} ({time.time()-t0:.0f}s)  "
                  f"best so far: {best_so_far:.1f}", flush=True)

    save_size_results(results, args.outpath, args.size)

    best_ig, best_count = max(results, key=lambda x: x[1])
    print(f"\n  Top overrepresented size-{args.size} CC motif:")
    print(f"    Labels: {best_ig.vs['label']}")
    print(f"    Edges:  {best_ig.get_edgelist()}")
    print(f"    Predicted count: {best_count:.1f}")


# ── Function 3: All Size (Greedy) ─────────────────────────────────────────────

def _enumerate_best_triangle(model, graph_info, n_celltypes, G_ADJ, G_FEAT, G_MASK, n_sample, n_nodes):
    """Find the best (highest-count) size-3 triangle pattern."""
    patterns = enumerate_size3_patterns(n_celltypes)
    print(f"  [Size 3] Evaluating {len(patterns)} triangle patterns…")
    results  = []
    for ig_pat in patterns:
        count = predict_count(model, ig_pat, G_ADJ, G_FEAT, G_MASK, n_sample, n_nodes)
        results.append((ig_pat, count))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def function3_all_size(args):
    """
    Greedy motif growth from size-3 up to target size.
    Mirrors: python TrimNN.py -function all_size -size N -celltype C -target Y -outpath Z
    """
    print(f"[Function 3] All Size — greedy growth to size {args.size}")
    model = load_model(args.model)

    print(f"  Loading target graph: {args.target}")
    graph_info, n_celltypes_graph, _ = load_graph(args.target, prune=args.prune)
    node_labels, neighbors, khop, n_nodes = graph_info
    n_celltypes = args.celltype if args.celltype else n_celltypes_graph
    print(f"  Graph: {n_nodes:,} nodes, using {n_celltypes} cell types")

    os.makedirs(args.outpath, exist_ok=True)
    print(f"  Precomputing graph tensors ({EVAL_NODES} sampled nodes)…")
    G_ADJ, G_FEAT, G_MASK, n_sample, _ = _precompute_graph_tensors(graph_info)

    # ── Size 3: find best triangle ──────────────────────────────────────────
    size3_results = _enumerate_best_triangle(
        model, graph_info, n_celltypes, G_ADJ, G_FEAT, G_MASK, n_sample, n_nodes)
    save_size_results(size3_results, args.outpath, 3)

    best_ig, best_count = size3_results[0]
    print(f"  Best size-3: labels={best_ig.vs['label']}  count={best_count:.1f}")
    best_ig.write(os.path.join(args.outpath, "Overrepresented_size3.gml"), format='gml')

    # ── Greedy growth ───────────────────────────────────────────────────────
    current_size = 3
    while current_size < args.size:
        current_size += 1
        print(f"\n  [Size {current_size}] Generating candidate extensions…", flush=True)
        t0         = time.time()
        candidates = generate_candidate_extensions(best_ig, n_celltypes)
        print(f"    {len(candidates)} unique candidates  [{time.time()-t0:.1f}s]")

        if not candidates:
            print(f"    No valid extensions found — stopping at size {current_size - 1}")
            break

        print(f"    Predicting counts…", flush=True)
        size_results = []
        best_this   = 0.0
        best_cand   = candidates[0]
        for i, ig_pat in enumerate(candidates):
            count = predict_count(model, ig_pat, G_ADJ, G_FEAT, G_MASK, n_sample, n_nodes)
            size_results.append((ig_pat, count))
            if count > best_this:
                best_this = count
                best_cand = ig_pat
            if (i + 1) % max(1, len(candidates) // 5) == 0 or (i + 1) == len(candidates):
                print(f"      {i+1}/{len(candidates)} ({time.time()-t0:.0f}s)  "
                      f"best: {best_this:.1f}", flush=True)

        save_size_results(size_results, args.outpath, current_size)
        best_ig    = best_cand
        best_count = best_this
        print(f"  Best size-{current_size}: labels={best_ig.vs['label']}  "
              f"count={best_count:.1f}")

    print(f"\n  Final best motif (size {best_ig.vcount()}):")
    print(f"    Labels: {best_ig.vs['label']}")
    print(f"    Edges:  {best_ig.get_edgelist()}")
    print(f"    Predicted count: {best_count:.1f}")
    return best_ig


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="TrimNN Replacement — GNN-based cellular community motif detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--function', '-function', required=True,
                   choices=['subgraph_matching', 'specific_size', 'all_size'],
                   help='Which TrimNN function to run')
    p.add_argument('--target', '-target', required=True,
                   help='Target graph: .gml or .csv (X,Y,cell_type columns)')
    p.add_argument('--outpath', '-outpath', required=True,
                   help='Output directory')
    p.add_argument('--motif', '-motif', default=None,
                   help='[Function 1] Motif GML file to match')
    p.add_argument('--size', '-size', type=int, default=3,
                   help='[Function 2/3] Pattern size (3–9)')
    p.add_argument('--celltype', '-celltype', type=int, default=None,
                   help='Number of cell types (auto-detected from graph if omitted)')
    p.add_argument('--k', '-k', type=int, default=2,
                   help='K-hop neighbourhood radius')
    p.add_argument('--search', '-search', default='greedy',
                   choices=['greedy'],
                   help='[Function 3] Search strategy')
    p.add_argument('--model', default=None,
                   help='Path to trained model .pt file')
    p.add_argument('--prune', type=lambda x: x.lower() != 'false',
                   default=True,
                   help='Prune outlier long edges from CSV graph (True/False)')
    p.add_argument('--eval_nodes', type=int, default=EVAL_NODES,
                   help='Number of nodes sampled for count estimation')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Override global constants if user changed them
    EVAL_NODES = args.eval_nodes
    K_HOP = args.k

    fn = args.function
    if fn == 'subgraph_matching':
        if not args.motif:
            print("Error: --motif required for subgraph_matching", file=sys.stderr)
            sys.exit(1)
        function1_subgraph_matching(args)
    elif fn == 'specific_size':
        function2_specific_size(args)
    elif fn == 'all_size':
        function3_all_size(args)
