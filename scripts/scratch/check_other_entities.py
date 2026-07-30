"""Check whether Table 1's runs live under a different W&B entity or team.

`07_acknowledgements.tex` credits Table 1's compute to TensorPool, and `src/globals.py` still
defaults to a `jkazdan/...` checkpoint, so the runs may have been logged by a collaborator
rather than to `rylan/*`. This asks the API directly what entities and teams the key can see,
then looks for rephrased/perturbed eval runs in each — much faster than walking every project
of every entity, because it answers the specific question first.
"""

import json

import wandb

NEEDLES = ("rephras", "perturb")
CANDIDATE_ENTITIES = ["jkazdan", "stellaathena", "tensorpool", "koyejolab"]


def mentions_modified(config: dict) -> bool:
    try:
        blob = json.dumps(config, default=str).lower()
    except (TypeError, ValueError):
        blob = str(config).lower()
    return any(needle in blob for needle in NEEDLES)


def main() -> None:
    api = wandb.Api(timeout=600)

    print("=== Identity visible to this API key ===")
    viewer = api.viewer
    print(f"  username       : {getattr(viewer, 'username', '?')}")
    print(f"  default_entity : {getattr(viewer, 'entity', '?')}")
    teams = list(getattr(viewer, "teams", []) or [])
    print(f"  teams          : {teams}")

    entities = list(dict.fromkeys(teams + CANDIDATE_ENTITIES))
    print(f"\n=== Probing entities: {entities} ===")

    for entity in entities:
        try:
            projects = list(api.projects(entity))
        except Exception as e:
            print(f"\n[{entity}] NOT VISIBLE: {type(e).__name__}: {str(e)[:120]}")
            continue

        print(f"\n[{entity}] {len(projects)} visible projects")
        for project in projects:
            name = project.name
            if not any(
                key in name.lower() for key in ("mem", "contam", "eval", "math", "scoring")
            ):
                continue
            try:
                runs = api.runs(f"{entity}/{name}", per_page=200)
                n_runs = 0
                n_hits = 0
                sizes = set()
                for run in runs:
                    n_runs += 1
                    if mentions_modified(run.config):
                        n_hits += 1
                        model_config = run.config.get("model_config")
                        model_name = (
                            model_config.get("model", "")
                            if isinstance(model_config, dict)
                            else str(model_config)
                        )
                        sizes.add(model_name)
                print(f"    {name}: {n_runs} runs, {n_hits} mention a modified set")
                for model_name in sorted(sizes)[:20]:
                    print(f"        {model_name}")
            except Exception as e:
                print(f"    {name}: error {type(e).__name__}: {str(e)[:100]}")


if __name__ == "__main__":
    main()
