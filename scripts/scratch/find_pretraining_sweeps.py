"""Search every visible W&B project for notebook 11's pretraining sweep IDs.

An earlier pass checked four guessed project paths, found none of the 15 sweeps, and concluded
they were "unreachable". That conclusion outran the evidence: absence from four guesses is not
absence from the account. This checks every project of every entity the API key can see, and
asks W&B for each sweep ID directly rather than enumerating runs (much faster, and it finds
sweeps whose runs were deleted).
"""

import wandb

NOTEBOOK_11_PT_SWEEPS = [
    "rkx5xfde", "g31f7bsb", "ehxxzk5n",              # 34M
    "u7dxxphm", "o6aoejzc", "1nwyun1z", "xbiu535y",  # 62M
    "ho49sshi", "x8gmmzlo", "u5xcf726",              # 93M
    "sl086kx0",                                       # 153M
    "09c432gh", "gsx7gisg", "6f9ah90l", "r9fixoce",  # 344M
]

EXTRA_ENTITIES = ["jkazdan", "stellaathena"]


def main() -> None:
    api = wandb.Api(timeout=600)

    entities = [api.default_entity]
    for team in getattr(api.viewer, "teams", []) or []:
        if team not in entities:
            entities.append(team)
    for entity in EXTRA_ENTITIES:
        if entity not in entities:
            entities.append(entity)

    project_paths = []
    for entity in entities:
        try:
            for project in api.projects(entity):
                project_paths.append(f"{entity}/{project.name}")
        except Exception as e:
            print(f"[{entity}] cannot list projects: {type(e).__name__}")
    print(f"Searching {len(project_paths)} projects for {len(NOTEBOOK_11_PT_SWEEPS)} sweep IDs\n")

    found = {}
    for sweep_id in NOTEBOOK_11_PT_SWEEPS:
        hits = []
        for path in project_paths:
            try:
                sweep = api.sweep(f"{path}/{sweep_id}")
            except Exception:
                continue
            # A sweep object can come back for a path that merely accepts the id; confirm it
            # has runs before counting it as a real hit.
            try:
                n_runs = len(list(sweep.runs))
            except Exception:
                n_runs = -1
            hits.append((path, n_runs))
        found[sweep_id] = hits
        status = (
            ", ".join(f"{p} ({n} runs)" for p, n in hits) if hits else "NOT FOUND anywhere"
        )
        print(f"  {sweep_id}: {status}")

    n_found = sum(1 for hits in found.values() if hits)
    print(f"\n{n_found}/{len(NOTEBOOK_11_PT_SWEEPS)} sweeps located.")
    if n_found:
        print("\nProjects that hold them:")
        holders = {}
        for hits in found.values():
            for path, n_runs in hits:
                holders.setdefault(path, 0)
                holders[path] += 1
        for path, count in sorted(holders.items(), key=lambda kv: -kv[1]):
            print(f"  {path}: {count} of the sweeps")


if __name__ == "__main__":
    main()
