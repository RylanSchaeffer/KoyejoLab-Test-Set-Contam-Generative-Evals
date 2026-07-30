"""Turn `audit_inventory.py --gaps-only` output into a model list for the overtrained sweep.

Ordering matters: the sweep may be interrupted, so checkpoints are emitted smallest-model
first (cheapest, so the grid fills in fastest) and within a model size ordered by overtrain
multiplier then replica count. A partial run therefore yields complete accuracy-vs-replicas
curves for the small models rather than a scattered half-grid.

Usage:
    python scripts/audit_inventory.py --gaps-only > gaps.txt
    python scripts/scratch/build_overtrained_model_list.py gaps.txt out.txt
"""

import re
import sys

NAME_RE = re.compile(
    r"^mem_Qwen3-(?P<size>[\d.]+)M_minerva_math_rep_(?P<rep>\d+)_sbst_"
    r"(?P<sbst>[\d.]+)_epch_(?P<epch>\d+)_ot_(?P<ot>[\d.]+)$"
)


def main(gaps_path: str, out_path: str) -> None:
    with open(gaps_path) as f:
        lines = [line.strip() for line in f]

    entries = []
    for line in lines:
        match = NAME_RE.match(line)
        if match is None:
            continue
        overtrain = float(match.group("ot"))
        if overtrain <= 1.0:
            continue  # compute-optimal checkpoints are already evaluated
        entries.append(
            (
                float(match.group("size")),
                overtrain,
                int(match.group("rep")),
                line,
            )
        )

    entries.sort(key=lambda e: (e[0], e[1], e[2]))

    seen = set()
    ordered = []
    for _, _, _, name in entries:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(f"RylanSchaeffer/{name}")

    with open(out_path, "w") as f:
        f.write("\n".join(ordered) + "\n")

    print(f"Wrote {len(ordered)} checkpoints to {out_path}")
    by_size = {}
    for size, overtrain, _, _ in entries:
        by_size.setdefault((size, overtrain), 0)
        by_size[(size, overtrain)] += 1
    for (size, overtrain), count in sorted(by_size.items()):
        print(f"  {size:6.0f}M  ot={overtrain:<6g}  {count} checkpoints")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
