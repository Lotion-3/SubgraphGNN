"""
TrimNN data preparation and fixed evaluation for autoresearch.

Generates (pattern, graph, count) training samples from demo_data.gml.
VF2 subgraph isomorphism counting provides the ground truth labels.
Data is cached to disk after first generation (~1 second).

DO NOT MODIFY — this file defines the task and the fixed evaluation metric.
"""

import os
import sys
import math
import time
import random
import pickle
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F
import igraph as ig

# ---------------------------------------------------------------------------
# Constants (fixed — do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET    = 300          # training time budget in seconds
K_HOP          = 2            # k-hop neighborhood for subgraph extraction
N_LABELS       = 8            # number of cell type labels
MAX_GRAPH_NODES  = 48         # padded size for target subgraphs (covers p99)
MAX_PATTERN_NODES = 8         # padded size for pattern (triangles have 3 nodes)
VAL_FRAC       = 0.2          # fraction of data held out for validation

DEMO_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../TrimNN/demo_data"))
GRAPH_PATH  = os.path.join(DEMO_DIR, "demo_data.gml")
CACHE_DIR   = os.path.expanduser("~/.cache/autoresearch_trimnn")
CACHE_FILE  = os.path.join(CACHE_DIR, "data_k2.pkl")

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _load_demo_graph():
    g = ig.read(GRAPH_PATH)
    g.vs["label"] = [int(x) for x in g.vs["label"]]
    g.es["label"] = [0] * g.ecount()
    return g


def _generate_triangle_patterns():
    """All non-isomorphic triangles over N_LABELS cell types."""
    base = ig.Graph(n=3, edges=[[0, 1], [0, 2], [1, 2]])
    patterns, seen = [], []
    for combo in product(range(N_LABELS), repeat=3):
        p = base.copy()
        p.vs["label"] = list(combo)
        p.es["label"] = [0, 0, 0]
        if not any(p.isomorphic_vf2(s,
                                    color1=p.vs["label"],
                                    color2=s.vs["label"]) for s in seen):
            patterns.append(p)
            seen.append(p)
    return patterns


def _khop_subgraph(graph, center, k):
    """Return the induced subgraph of the k-hop neighborhood around `center`."""
    visited = {center}
    frontier = {center}
    for _ in range(k):
        nf = set()
        for v in frontier:
            for nb in graph.neighbors(v):
                if nb not in visited:
                    nf.add(nb)
                    visited.add(nb)
        frontier = nf
    nodes = sorted(visited)
    return graph.induced_subgraph(nodes)


def generate_data():
    """
    Return a list of dicts with keys: pattern (ig.Graph), graph (ig.Graph), count (int).
    Results are cached to CACHE_FILE.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        print(f"Data: loaded {len(data)} cached samples from {CACHE_FILE}")
        return data

    print("Data: generating training samples via VF2 (first run only, ~5s) ...")
    t0 = time.time()
    graph    = _load_demo_graph()
    patterns = _generate_triangle_patterns()

    data = []
    for center in range(graph.vcount()):
        subg = _khop_subgraph(graph, center, K_HOP)
        if subg.vcount() < 3:
            continue
        for pat in patterns:
            count = subg.count_subisomorphisms_vf2(
                pat,
                color1=subg.vs["label"],
                color2=pat.vs["label"],
            )
            data.append({"pattern": pat, "graph": subg, "count": int(count)})

    random.seed(0)
    random.shuffle(data)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"Data: generated {len(data)} samples in {time.time() - t0:.1f}s, cached to {CACHE_FILE}")
    return data


# ---------------------------------------------------------------------------
# Tensor conversion
# ---------------------------------------------------------------------------

def _graph_to_tensors(graph, max_nodes):
    """
    Convert an igraph graph to padded (adjacency, one-hot features, mask) tensors.

    Returns:
        adj:   float32 (max_nodes, max_nodes) — symmetric, 0-padded
        feat:  float32 (max_nodes, N_LABELS)  — one-hot cell type, 0-padded
        mask:  bool    (max_nodes,)            — True for real nodes
    """
    n = min(graph.vcount(), max_nodes)
    adj  = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    feat = np.zeros((max_nodes, N_LABELS),  dtype=np.float32)
    mask = np.zeros((max_nodes,),            dtype=bool)

    for (u, v) in graph.get_edgelist():
        if u < max_nodes and v < max_nodes:
            adj[u, v] = 1.0
            adj[v, u] = 1.0

    for i, lbl in enumerate(graph.vs["label"]):
        if i >= max_nodes:
            break
        feat[i, int(lbl) % N_LABELS] = 1.0
        mask[i] = True

    return (torch.from_numpy(adj),
            torch.from_numpy(feat),
            torch.from_numpy(mask))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SubgraphDataset(torch.utils.data.Dataset):
    """Pre-tensored dataset of (pattern, graph, vf2_count) triples."""

    def __init__(self, raw):
        self.samples = []
        for d in raw:
            p_adj, p_feat, p_mask = _graph_to_tensors(d["pattern"], MAX_PATTERN_NODES)
            g_adj, g_feat, g_mask = _graph_to_tensors(d["graph"],   MAX_GRAPH_NODES)
            count = torch.tensor([float(d["count"])], dtype=torch.float32)
            self.samples.append((p_adj, p_feat, p_mask, g_adj, g_feat, g_mask, count))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_dataloader(raw_data, batch_size, shuffle=True):
    """Return a DataLoader over SubgraphDataset."""
    ds = SubgraphDataset(raw_data)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=False,
    )


def split_data(data):
    """Return (train_data, val_data) using a fixed random split."""
    n_val = max(1, int(len(data) * VAL_FRAC))
    return data[n_val:], data[:n_val]


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_val_mse(model, val_loader, device):
    """
    Mean squared error between predicted and VF2-exact subgraph counts.
    Lower is better. This is the fixed ground-truth metric.
    """
    model.eval()
    total_sq_err = 0.0
    total_n      = 0
    for batch in val_loader:
        p_adj, p_feat, p_mask, g_adj, g_feat, g_mask, counts = [
            x.to(device) for x in batch
        ]
        pred = model(p_adj, p_feat, p_mask, g_adj, g_feat, g_mask)
        total_sq_err += F.mse_loss(pred.view(-1), counts.view(-1),
                                   reduction="sum").item()
        total_n += counts.shape[0]
    return total_sq_err / max(total_n, 1)


# ---------------------------------------------------------------------------
# Main — generate data when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data = generate_data()
    train_data, val_data = split_data(data)
    counts = [d["count"] for d in data]
    print(f"Train: {len(train_data)}  Val: {len(val_data)}")
    print(f"Count distribution — 0: {counts.count(0)}, "
          f">0: {sum(1 for c in counts if c > 0)}, "
          f"max: {max(counts)}, mean: {sum(counts)/len(counts):.3f}")
