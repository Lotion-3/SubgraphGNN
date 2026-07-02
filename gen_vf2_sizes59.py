"""
Generate VF2 ground-truth training data for pattern sizes 5-9.

Produces per-node (pattern, k-hop-subgraph, count) triples for the demo graph,
matching the format of prepare.py's generate_data() for sizes 3-4.

Topologies (growing from kite, matching TrimNN's style):
  size-5:  kite + triangle closing = 5 nodes, 7 edges
  size-6:  size-5 + node closing triangle = 6 nodes, 9 edges
  size-7:  size-7 + node closing triangle = 7 nodes, 11 edges
  size-8:  size-8 + pendant from one branch = 8 nodes, 12 edges
  size-9:  size-9 + pendant extension = 9 nodes, 13 edges

Also generates whole-graph VF2 counts (matching the intestinal CSV format) for
comparison with TrimNN's Predicted_occurrence_sizeN.csv files.
"""

import os, sys, time, pickle, random
from itertools import product, combinations_with_replacement
import igraph as ig
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
TRIMNN_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, "../TrimNN"))
DEMO_DIR     = os.path.join(TRIMNN_DIR, "demo_data")
GRAPH_PATH   = os.path.join(DEMO_DIR, "demo_data.gml")
CACHE_DIR    = os.path.expanduser("~/.cache/autoresearch_trimnn")
OUT_DIR      = os.path.join(DEMO_DIR, "vf2_sizes59")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

K_HOP           = 2
N_LABELS        = 8   # demo graph has 8 cell types (0-7)
MAX_GRAPH_NODES = 64  # slightly bigger than size-3/4 to cover larger patterns
MAX_PATTERN_NODES_S59 = 9  # up to size-9

# ---------------------------------------------------------------------------
# Canonical unlabeled topologies for sizes 5-9
# Built by growing from the kite topology (size-4)
# ---------------------------------------------------------------------------
# Size-4 kite: nodes {0,1,2,3}, edges {(0,1),(0,2),(1,2),(1,3),(2,3)}
# Growth rule: each new node connects to 2 nodes of the previous pattern
# forming a new triangle with an existing edge.
# ---------------------------------------------------------------------------

TOPOLOGIES = {
    3: {
        "n": 3, "edges": [(0,1),(0,2),(1,2)],
        "desc": "triangle K3"
    },
    4: {
        "n": 4, "edges": [(0,1),(0,2),(1,2),(1,3),(2,3)],
        "desc": "kite (two triangles sharing edge)"
    },
    5: {
        # node 4 connects to {1,3} — closes triangle {1,3,4} using existing edge (1,3)
        "n": 5, "edges": [(0,1),(0,2),(1,2),(1,3),(2,3),(1,4),(3,4)],
        "desc": "kite + triangle on (1,3)"
    },
    6: {
        # node 5 connects to {2,3} — closes triangle {2,3,5} using existing edge (2,3)
        "n": 6, "edges": [(0,1),(0,2),(1,2),(1,3),(2,3),(1,4),(3,4),(2,5),(3,5)],
        "desc": "kite + 2 triangles"
    },
    7: {
        # node 6 connects to {3,4} — closes triangle {3,4,6} using existing edge (3,4)
        "n": 7, "edges": [(0,1),(0,2),(1,2),(1,3),(2,3),(1,4),(3,4),(2,5),(3,5),(4,6),(3,6)],
        "desc": "kite + 3 triangles"
    },
    8: {
        # node 7 connects to {4,6} — closes triangle {4,6,7}
        "n": 8, "edges": [(0,1),(0,2),(1,2),(1,3),(2,3),(1,4),(3,4),(2,5),(3,5),(4,6),(3,6),(4,7),(6,7)],
        "desc": "kite + 4 triangles"
    },
    9: {
        # node 8 connects to {5,3} — closes triangle {3,5,8}
        "n": 9, "edges": [(0,1),(0,2),(1,2),(1,3),(2,3),(1,4),(3,4),(2,5),(3,5),(4,6),(3,6),(4,7),(6,7),(5,8),(3,8)],
        "desc": "kite + 5 triangles"
    },
}


def make_base_graph(size):
    """Build unlabeled igraph for a given size."""
    topo = TOPOLOGIES[size]
    g = ig.Graph(n=topo["n"], edges=topo["edges"])
    g.es["label"] = [0] * g.ecount()
    return g


def enumerate_labeled_patterns(size, n_labels=N_LABELS, max_per_size=None):
    """
    Enumerate all non-isomorphic labeled patterns for a given topology.
    Each node gets a cell-type label from [0, n_labels).
    Returns list of ig.Graph objects.
    Uses isomorphism filtering to avoid duplicates.
    """
    base = make_base_graph(size)
    n = TOPOLOGIES[size]["n"]
    patterns, seen = [], []

    t0 = time.time()
    count = 0
    for combo in product(range(n_labels), repeat=n):
        p = base.copy()
        p.vs["label"] = list(combo)
        # Check if isomorphic to any seen pattern
        is_dup = any(
            p.isomorphic_vf2(s, color1=p.vs["label"], color2=s.vs["label"])
            for s in seen
        )
        if not is_dup:
            patterns.append(p)
            seen.append(p)
            count += 1
            if max_per_size and count >= max_per_size:
                break
    elapsed = time.time() - t0
    print(f"  size-{size}: {len(patterns)} distinct patterns "
          f"(from {n_labels}^{n}={n_labels**n} combos, {elapsed:.1f}s)")
    return patterns


def load_demo_graph():
    g = ig.read(GRAPH_PATH)
    g.vs["label"] = [int(x) for x in g.vs["label"]]
    g.es["label"] = [0] * g.ecount()
    return g


def khop_subgraph(graph, center, k):
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
    return graph.induced_subgraph(sorted(visited))


def generate_per_node_data(graph, patterns, size):
    """
    For each node in graph, count each pattern in its k-hop subgraph.
    Returns list of dicts matching prepare.py's format.
    """
    data = []
    t0 = time.time()
    n_nodes = graph.vcount()
    for i, center in enumerate(range(n_nodes)):
        subg = khop_subgraph(graph, center, K_HOP)
        if subg.vcount() < size:
            continue
        for pat in patterns:
            count = subg.count_subisomorphisms_vf2(
                pat,
                color1=subg.vs["label"],
                color2=pat.vs["label"],
            )
            data.append({"pattern": pat, "graph": subg, "count": int(count),
                         "size": size})
        if (i+1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed
            eta = (n_nodes - i - 1) / rate
            print(f"    node {i+1}/{n_nodes} | {elapsed:.0f}s elapsed | ETA {eta:.0f}s",
                  end="\r", flush=True)
    print()
    return data


def generate_whole_graph_counts(graph, patterns, size):
    """
    Count each pattern across the WHOLE graph (not k-hop).
    Used to compare with TrimNN's Predicted_occurrence_sizeN.csv format.
    Returns DataFrame with columns: motif_str, label, occurrence_number
    """
    rows = []
    t0 = time.time()
    for i, pat in enumerate(patterns):
        count = graph.count_subisomorphisms_vf2(
            pat,
            color1=graph.vs["label"],
            color2=pat.vs["label"],
        )
        motif_str = str(pat)
        label_str = str(pat.vs["label"])
        rows.append({"motif": motif_str, "label": label_str,
                     "occurrence_number": count})
        if (i+1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed
            eta = (len(patterns) - i - 1) / rate
            print(f"    pattern {i+1}/{len(patterns)} | {elapsed:.0f}s | ETA {eta:.0f}s",
                  end="\r", flush=True)
    print()
    return pd.DataFrame(rows)


def main():
    print("Loading demo graph...")
    graph = load_demo_graph()
    print(f"  {graph.vcount()} nodes, {graph.ecount()} edges, "
          f"{N_LABELS} cell types (labels {set(graph.vs['label'])})")

    for size in range(5, 10):
        cache_pernode = os.path.join(CACHE_DIR, f"data_s{size}_k{K_HOP}.pkl")
        csv_whole     = os.path.join(OUT_DIR, f"Occurrence_number_size{size}.csv")

        print(f"\n=== SIZE {size} ({TOPOLOGIES[size]['desc']}) ===")

        # Enumerate patterns
        print(f"  Enumerating labeled patterns (size {size}, {N_LABELS} cell types)...")
        patterns = enumerate_labeled_patterns(size, n_labels=N_LABELS)

        # Per-node VF2 counts (for training)
        if os.path.exists(cache_pernode):
            print(f"  Per-node cache exists: {cache_pernode} — skipping")
        else:
            print(f"  Generating per-node VF2 training data ({graph.vcount()} nodes × {len(patterns)} patterns)...")
            data = generate_per_node_data(graph, patterns, size)
            random.seed(size)
            random.shuffle(data)
            with open(cache_pernode, "wb") as f:
                pickle.dump(data, f)
            print(f"  Saved {len(data)} samples to {cache_pernode}")

        # Whole-graph VF2 counts (for TrimNN comparison)
        if os.path.exists(csv_whole):
            print(f"  Whole-graph CSV exists: {csv_whole} — skipping")
        else:
            print(f"  Computing whole-graph VF2 counts ({len(patterns)} patterns on full graph)...")
            df = generate_whole_graph_counts(graph, patterns, size)
            df.to_csv(csv_whole, index=False)
            nonzero = (df["occurrence_number"] > 0).sum()
            print(f"  Saved {csv_whole}: {nonzero}/{len(patterns)} non-zero patterns")

    print("\nDone! Summary:")
    for size in range(5, 10):
        csv_whole = os.path.join(OUT_DIR, f"Occurrence_number_size{size}.csv")
        cache_pernode = os.path.join(CACHE_DIR, f"data_s{size}_k{K_HOP}.pkl")
        exists_csv = "✓" if os.path.exists(csv_whole) else "✗"
        exists_pkl = "✓" if os.path.exists(cache_pernode) else "✗"
        print(f"  size-{size}: csv={exists_csv} pernode_cache={exists_pkl}")


if __name__ == "__main__":
    main()
