"""Restore RylanSchaeffer/math_rephrased on the HF Hub from the local cache snapshot.

The dataset stopped resolving on the Hub (verified 2026-07-30: anonymous dataset_info returns
RepositoryNotFoundError / 401), but the exact artifact survives in the shared HF cache:
5,000 rows, columns [idx, original_problem, problem, answer, level, type, solution], plus the
README. This script re-uploads those exact bytes so reviewers fetch the identical dataset the
experiments used.

Token handling is deliberately strict because of reviews/2026_neurips/HF_TOKEN_INCIDENT.md:
the shared cache's token file belongs to another user, so this script NEVER falls back to
HF_HOME token resolution. It requires an explicit token from $HF_TOKEN or ~/.hf_token and
refuses to proceed unless that token authenticates as RylanSchaeffer.

Usage:
    # one-time, in your own terminal (keeps the token out of shell history):
    read -s -p "HF write token: " HFT && umask 077 && printf %s "$HFT" > ~/.hf_token && unset HFT

    python scripts/reupload_math_rephrased.py
"""

import glob
import os
import sys

from huggingface_hub import HfApi

REPO_ID = "RylanSchaeffer/math_rephrased"
EXPECTED_USER = "RylanSchaeffer"
CACHE_GLOB = (
    "/lfs/skampere1/0/shared_hf_cache/hub/datasets--RylanSchaeffer--math_rephrased/snapshots/*"
)


def get_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    token_path = os.path.expanduser("~/.hf_token")
    if os.path.isfile(token_path):
        with open(token_path) as f:
            return f.read().strip()
    sys.exit(
        "No token found. Set $HF_TOKEN or create ~/.hf_token (chmod 600) with a write-scoped "
        "token for the RylanSchaeffer account. Do NOT use the shared-cache token; see "
        "reviews/2026_neurips/HF_TOKEN_INCIDENT.md."
    )


def main() -> None:
    snapshots = sorted(glob.glob(CACHE_GLOB))
    if not snapshots:
        sys.exit(f"No cached snapshot found at {CACHE_GLOB}")
    snapshot = snapshots[-1]
    readme = os.path.join(snapshot, "README.md")
    parquet = os.path.join(snapshot, "data", "test-00000-of-00001.parquet")
    for path in (readme, parquet):
        if not os.path.isfile(path) and not os.path.islink(path):
            sys.exit(f"Missing cached file: {path}")

    token = get_token()
    api = HfApi(token=token)
    identity = api.whoami()["name"]
    if identity != EXPECTED_USER:
        sys.exit(
            f"Token authenticates as '{identity}', not '{EXPECTED_USER}'. Refusing to upload "
            f"into the wrong namespace (see HF_TOKEN_INCIDENT.md)."
        )

    api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True, private=False)
    for local, remote in ((readme, "README.md"), (parquet, "data/test-00000-of-00001.parquet")):
        api.upload_file(
            path_or_fileobj=os.path.realpath(local),
            path_in_repo=remote,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Restore dataset from local cache after Hub repo loss",
        )
        print(f"Uploaded {remote}")

    anon = HfApi(token=False)
    info = anon.dataset_info(REPO_ID)
    print(f"Verified: {REPO_ID} resolves anonymously, private={info.private}")
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    fetched = hf_hub_download(
        REPO_ID, "data/test-00000-of-00001.parquet", repo_type="dataset", token=False
    )
    table = pq.read_table(fetched)
    assert table.num_rows == 5000, f"Expected 5000 rows, got {table.num_rows}"
    print(f"Verified: anonymous download returns {table.num_rows} rows, "
          f"columns {table.column_names}")


if __name__ == "__main__":
    main()
