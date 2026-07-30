# The HuggingFace token on skampere1 belongs to someone else

Found 2026-07-29 while preparing the paraphrased-contamination runs. **Nothing of this project's
leaked**, but the exposure is real and the credentials should be rotated.

## What happened

`scripts/pretrain_language_model.py` builds its push target from the ambient identity:

```python
hf_username = get_hf_username()          # HfApi().whoami()["name"]
hub_model_id = f"{hf_username}/{pted_model_hf_name}"
```

On skampere1 that resolves to **`ruili0`**, not `RylanSchaeffer`:

| | |
|---|---|
| `HF_HOME` | `/lfs/skampere1/0/shared_hf_cache` (**shared**, owned by `brando9`) |
| Token file | `/lfs/skampere1/0/shared_hf_cache/token`, mode **`-rw-rw-rw-`** |
| Resolves to | user `ruili0`, email `3303lr@gmail.com`, org `Apollo-LMMs` |
| Token scope | **write** |
| Token label | `aaa`, created 2026-01-15, file last written 2026-05-14 |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | unset — so the file is what wins |

`stored_tokens` in the same directory holds four named profiles — `Upload to HuggingFace Google
2024 Summer`, `aaa`, `master2`, `master_token` — also mode `-rw-rw-rw-`.

So any job on this node that inherits `HF_HOME` and calls `push_to_hub()` publishes to `ruili0`.

## Blast radius: nothing of ours leaked

- `ruili0` owns **0** models matching `mem_*`.
- `RylanSchaeffer` owns **196**; `jkazdan` owns **72**. The paper's checkpoints are intact and in
  the right places.

The paper's pretraining ran in January 2026 and the shared token file was last written
2026-05-14, so the historical runs pushed under a correct token. The mismatch is recent, and the
paraphrased runs launched 2026-07-29 would have been the **first** affected. They were not:
the push is gated behind `PRETRAIN_SKIP_HUB_PUSH=1` (see `scripts/pretrain_language_model_v1.py`),
and `trainer.save_model()` runs before the push, so the checkpoints are on local disk under
`models/pt_language_model/` and can be pushed later to the correct account.

## Two separate problems

**1. Ours — pushes go to the wrong namespace.** Fix by exporting a real token before any run that
uploads:

```bash
export HF_TOKEN=hf_...            # RylanSchaeffer's own token; takes precedence over the file
# verify before trusting it:
python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])"   # want RylanSchaeffer
```

A private `HF_HOME` would also work but gives up the shared 513 GB dataset cache, which is worth
keeping. `HF_TOKEN` alone is enough — it overrides the token file.

Worth asserting the identity in the script rather than trusting the environment, so a wrong token
fails loudly instead of publishing to a stranger.

**2. Not ours — four HF tokens, at least one write-scoped, are world-readable on a shared node.**
Anyone with access to skampere1 can read `/lfs/skampere1/0/shared_hf_cache/token` and write to
`ruili0`'s HuggingFace account. Rui Li should **revoke and rotate** that token, and the shared
cache should hold datasets, not credentials. This is worth telling Rui and brando9 directly; it
is their exposure, not ours, and it is not something to fix by editing another user's file.
