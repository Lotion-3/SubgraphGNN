"""
Fast VF2 ground-truth generator for pattern sizes 5-9.

Uses random sampling (not exhaustive enumeration) — samples N random label
assignments per size, computes VF2 whole-graph counts on the demo graph.

Output: demo_data/vf2_sizes59/Occurrence_number_size{N}.csv
Format: label,occurrence_number  (matching sizes 3-4 CSVs)

Strategy: sample with replacement (no isomorphism dedup) — duplicates provide
harmless redundant signal; far outweighs the cost of quadratic dedup.
"""

import os, sys, time, random, json
import igraph as ig
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRIMNN_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../TrimNN"))
DEMO_DIR   = os.path.join(TRIMNN_DIR, "demo_data")
GRAPH_PATH = os.path.join(DEMO_DIR, "demo_data.gml")
OUT_DIR    = os.path.join(DEMO_DIR, "vf2_sizes59")
os.makedirs(OUT_DIR, exist_ok=True)

N_LABELS = 8   # demo graph has 8 cell types (0-7)

# Canonical topologies (same as gen_vf2_sizes59.py and eval_allsizes.py)
TOPOLOGIES = {
    5: {"n": 5, "edges": [(0,1),(0,2),(1,2),(1,3),(2,3),(1,4),(3,4)]},
    6: {"n": 6, "edges": [(0,1),(0,2),(1,2),(1,3),(2,3),(1,4),(3,4),(2,5),(3,5)]},
    7: {"n": 7, "edges": [(0,1),(0,2),(1,2),(1,3),(2,3),(1,4),(3,4),(2,5),(3,5),(4,6),(3,6)]},
    8: {"n": 8, "edges": [(0,1),(0,2),(1,2),(1,3),(2,3),(1,4),(3,4),(2,5),(3,5),(4,6),(3,6),(4,7),(6,7)]},
    9: {"n": 9, "edges": [(0,1),(0,2),(1,2),(1,3),(2,3),(1,4),(3,4),(2,5),(3,5),(4,6),(3,6),(4,7),(6,7),(5,8),(3,8)]},
}

# Number of samples per size — fewer for larger sizes (VF2 is slower)
N_SAMPLES = {5: 500, 6: 400, 7: 300, 8: 200, 9: 150}


def load_demo_graph():
    g = ig.read(GRAPH_PATH)
    g.vs["label"] = [int(x) for x in g.vs["label"]]
    g.es["label"] = [0] * g.ecount()
    return g


def make_pattern(size, labels):
    """Build igraph pattern with given labels for the canonical topology."""
    topo = TOPOLOGIES[size]
    g = ig.Graph(n=topo["n"], edges=topo["edges"])
    g.vs["label"] = list(labels)
    g.es["label"] = [0] * g.ecount()
    return g


def generate_for_size(graph, size, n_samples, rng):
    """Sample n_samples random labeled patterns and compute VF2 counts."""
    topo = TOPOLOGIES[size]
    n    = topo["n"]
    rows = []
    t0   = time.time()

    for i in range(n_samples):
        labels = [rng.randint(0, N_LABELS - 1) for _ in range(n)]
        pat    = make_pattern(size, labels)
        count  = graph.count_subisomorphisms_vf2(
            pat,
            color1=graph.vs["label"],
            color2=pat.vs["label"],
        )
        rows.append({"label": json.dumps(labels), "occurrence_number": int(count)})

        if (i + 1) % 50 == 0 or (i + 1) == n_samples:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed if elapsed > 0 else 0
            eta     = (n_samples - i - 1) / rate if rate > 0 else 0
            nonzero = sum(1 for r in rows if r["occurrence_number"] > 0)
            print(f"  size-{size}: {i+1}/{n_samples} | "
                  f"{nonzero} nonzero | {elapsed:.0f}s elapsed | ETA {eta:.0f}s",
                  end="\r", flush=True)

    print()
    return pd.DataFrame(rows)


def main():
    print("Loading demo graph...")
    graph = load_demo_graph()
    print(f"  {graph.vcount()} nodes, {graph.ecount()} edges, "
          f"{N_LABELS} cell types")

    for size in range(5, 10):
        out_csv = os.path.join(OUT_DIR, f"Occurrence_number_size{size}.csv")
        if os.path.exists(out_csv):
            df = pd.read_csv(out_csv)
            nz = (df["occurrence_number"] > 0).sum()
            print(f"size-{size}: already exists ({len(df)} patterns, {nz} nonzero) — skipping")
            continue

        n = N_SAMPLES[size]
        print(f"\nsize-{size}: sampling {n} patterns on {graph.vcount()}-node graph...")
        rng = random.Random(size * 1000 + 42)
        df  = generate_for_size(graph, size, n, rng)

        df.to_csv(out_csv, index=False)
        nz = (df["occurrence_number"] > 0).sum()
        print(f"  Saved {out_csv}: {nz}/{len(df)} nonzero")

    print("\n=== Summary ===")
    for size in range(5, 10):
        out_csv = os.path.join(OUT_DIR, f"Occurrence_number_size{size}.csv")
        if os.path.exists(out_csv):
            df = pd.read_csv(out_csv)
            nz = (df["occurrence_number"] > 0).sum()
            print(f"  size-{size}: {len(df)} patterns, {nz} nonzero ({100*nz/len(df):.1f}%)")
        else:
            print(f"  size-{size}: MISSING")


if __name__ == "__main__":
    main()
