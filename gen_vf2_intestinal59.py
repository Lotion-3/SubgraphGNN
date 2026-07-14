"""
Generate size 5-9 VF2 ground truth for intestinal tissue graphs.

Uses random label sampling (same strategy as gen_vf2_fast.py for demo graph).
Tissue graphs are ~17K nodes with 24-25 cell type labels — much richer signal
than the 743-node demo graph.

Output: /tmp/intestinal_vf2/{sample}_{size}.csv
Format: label,occurrence_number  (matching demo graph CSVs)

Run: python3 gen_vf2_intestinal59.py
"""

import os, sys, time, random, json
import igraph as ig
import pandas as pd

INT_BASE = os.path.join(os.path.dirname(__file__), "../TrimNN/intestinalOutputs")

# Tissues to generate VF2 for (B004_ascending = benchmark tissue)
TISSUES = [
    "B004_ascending_ct25",
    "B005_ascending_ct24",
    "B006_descendingSigmoid_ct23",
    "B008_ascending_ct22",
    "B010_ascending_ct25",
]

OUT_DIR = "/tmp/intestinal_vf2"
os.makedirs(OUT_DIR, exist_ok=True)

# Same kite-growing topology as gen_vf2_fast.py / train_trimnn.py
TOPOLOGIES = {
    5: {"n": 5, "edges": [(0,1),(0,2),(1,2),(1,3),(2,3),(1,4),(3,4)]},
    6: {"n": 6, "edges": [(0,1),(0,2),(1,2),(1,3),(2,3),(1,4),(3,4),(2,5),(3,5)]},
    7: {"n": 7, "edges": [(0,1),(0,2),(1,2),(1,3),(2,3),(1,4),(3,4),(2,5),(3,5),(4,6),(3,6)]},
}

# Fewer samples for larger sizes (VF2 is slower on larger graphs)
N_SAMPLES = {5: 300, 6: 200, 7: 100}


def load_tissue(sample_name):
    gml_path = os.path.join(INT_BASE, sample_name, f"{sample_name}.gml")
    g = ig.read(gml_path)
    g.vs["label"] = [int(x) for x in g.vs["label"]]
    n_labels = len(set(g.vs["label"]))
    g.es["label"] = [0] * g.ecount()
    return g, n_labels


def make_pattern(size, labels):
    topo = TOPOLOGIES[size]
    g = ig.Graph(n=topo["n"], edges=topo["edges"])
    g.vs["label"] = list(labels)
    g.es["label"] = [0] * g.ecount()
    return g


def generate_for_size(graph, n_labels, size, n_samples, rng):
    topo = TOPOLOGIES[size]
    n = topo["n"]
    rows = []
    t0 = time.time()

    for i in range(n_samples):
        labels = [rng.randint(0, n_labels - 1) for _ in range(n)]
        pat = make_pattern(size, labels)
        count = graph.count_subisomorphisms_vf2(
            pat,
            color1=graph.vs["label"],
            color2=pat.vs["label"],
        )
        rows.append({"label": json.dumps(labels), "occurrence_number": int(count)})

        if (i + 1) % 25 == 0 or (i + 1) == n_samples:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_samples - i - 1) / rate if rate > 0 else 0
            nonzero = sum(1 for r in rows if r["occurrence_number"] > 0)
            print(f"  size-{size}: {i+1}/{n_samples} | "
                  f"{nonzero} nonzero | {elapsed:.0f}s | ETA {eta:.0f}s",
                  end="\r", flush=True)

    print()
    return pd.DataFrame(rows)


def main():
    sizes = [5, 6, 7]

    for sample in TISSUES:
        print(f"\n=== {sample} ===")
        try:
            graph, n_labels = load_tissue(sample)
        except Exception as e:
            print(f"  ERROR loading {sample}: {e}")
            continue
        print(f"  {graph.vcount()} nodes, {graph.ecount()} edges, {n_labels} label types")

        for size in sizes:
            out_csv = os.path.join(OUT_DIR, f"{sample}_size{size}.csv")
            if os.path.exists(out_csv):
                df = pd.read_csv(out_csv)
                nz = (df["occurrence_number"] > 0).sum()
                print(f"  size-{size}: already exists ({len(df)} patterns, {nz} nonzero) — skip")
                continue

            n = N_SAMPLES[size]
            print(f"\n  size-{size}: sampling {n} patterns...")
            rng = random.Random(size * 1000 + 42)
            df = generate_for_size(graph, n_labels, size, n, rng)
            df.to_csv(out_csv, index=False)
            nz = (df["occurrence_number"] > 0).sum()
            print(f"  Saved: {nz}/{len(df)} nonzero ({100*nz/len(df):.1f}%)")

    print("\n=== Summary ===")
    for sample in TISSUES:
        for size in sizes:
            out_csv = os.path.join(OUT_DIR, f"{sample}_size{size}.csv")
            if os.path.exists(out_csv):
                df = pd.read_csv(out_csv)
                nz = (df["occurrence_number"] > 0).sum()
                print(f"  {sample} size-{size}: {len(df)} patterns, "
                      f"{nz} nonzero ({100*nz/len(df):.1f}%)")
            else:
                print(f"  {sample} size-{size}: MISSING")


if __name__ == "__main__":
    main()
