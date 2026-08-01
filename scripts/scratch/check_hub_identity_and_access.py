"""Check which HuggingFace identity is active and whether project checkpoints are readable.

Motivated by reviews/2026_neurips/HF_TOKEN_INCIDENT.md: HF_HOME points at a shared
cache whose world-readable token file belongs to another user, so the ambient
identity on skampere1 can silently be someone else. Pretraining derives its upload
namespace from whoami(), and checkpoints are pushed with hub_private_repo=True, so a
wrong identity breaks both writes (wrong namespace) and reads (private repos 401).

Read-only: resolves identity and queries model metadata. Uploads nothing.

Usage:
    uv run python scripts/scratch/check_hub_identity_and_access.py
"""

import os

from huggingface_hub import HfApi
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

EXPECTED_USER = "RylanSchaeffer"

PROBE_MODELS = [
    "RylanSchaeffer/mem_Qwen3-344M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1",
    "RylanSchaeffer/mem_Qwen3-34M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1",
    "jkazdan/mem_Qwen3-344M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1_sft",
]


def main() -> None:
    print("Environment:")
    print(f"  HF_HOME                = {os.environ.get('HF_HOME', '<unset>')}")
    print(
        f"  HF_TOKEN               = {'<set>' if os.environ.get('HF_TOKEN') else '<unset>'}"
    )
    print(
        f"  HUGGING_FACE_HUB_TOKEN = "
        f"{'<set>' if os.environ.get('HUGGING_FACE_HUB_TOKEN') else '<unset>'}"
    )
    print()

    api = HfApi()
    try:
        who = api.whoami()
        name = who.get("name")
    except Exception as exc:  # noqa: BLE001 - want the reason, whatever it is
        print(f"whoami() FAILED: {exc}")
        name = None

    print(f"Active identity: {name!r} (expected {EXPECTED_USER!r})")
    if name != EXPECTED_USER:
        print("  ==> WRONG IDENTITY. Uploads would land in the wrong namespace, and")
        print("      private project checkpoints may be unreadable. Export a real")
        print(
            "      HF_TOKEN before training. See reviews/2026_neurips/HF_TOKEN_INCIDENT.md"
        )
    print()

    print("Read access to project checkpoints:")
    for model_id in PROBE_MODELS:
        try:
            info = api.model_info(model_id)
            print(f"  OK      private={info.private!s:<5} {model_id}")
        except RepositoryNotFoundError:
            print(f"  DENIED  (404/401 - private or absent)   {model_id}")
        except GatedRepoError:
            print(f"  GATED                                    {model_id}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR   {type(exc).__name__}: {exc}   {model_id}")


if __name__ == "__main__":
    main()
