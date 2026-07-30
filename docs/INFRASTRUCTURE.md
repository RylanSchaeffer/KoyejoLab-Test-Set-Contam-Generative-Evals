# Infrastructure: Cluster, Environments, W&B

Everything here was **verified from `wandb-metadata.json` on finished runs**, not from documentation.
Last verified: 2026-07-27.

> Note: this file records internal Stanford SNAP cluster paths and hostnames (no credentials).
> If you'd rather not have that in a public repo, `git rm --cached docs/INFRASTRUCTURE.md` and add it
> to `.gitignore` — nothing else depends on it.

---

## Which node ran what

| W&B project | Host | GPUs | Repo checkout | Python environment |
|---|---|---|---|---|
| `memorization-scoring-vs-sampling-eval` | **skampere1** | 8× A100-SXM4-80GB | `/lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization` | `mem_scoring_vs_sampling_env` (CPython 3.11.5) |
| `memorization-scoring-vs-sampling-eval-teacher-forcing` | **skampere1** | 8× A100-SXM4-80GB | same as above | same as above |
| `memorization-scoring-vs-sampling-sft` | **skampere1** | 8× A100-SXM4-80GB | same as above | same as above |
| `memorization-scoring-vs-sampling-pt` (v1, **deleted**) | skampere2 | 8× H200 | **this repo**, pre-`934546a` | borrowed `pt_var_env` |
| `memorization-scoring-vs-sampling-pt-v2` | skampere2 | 8× H200 | **this repo**, `/lfs/skampere2/0/rschaef/KoyejoLab-Memorization-Scoring-vs-Sampling` | borrowed `pt_var_env` (CPython 3.12.12) |
| `scaling-memorization-pt` | skampere2 | 8× H200 | `/lfs/skampere2/0/rschaef/KoyejoLab-Scaling-Memorization` | `scaling_mem_env` |

**For this paper's rebuttal work, the answer is `skampere1`.** All generative evaluation, teacher-forced
evaluation, and SFT for the submitted results ran there, out of a single checkout with a single environment.
`skampere3` shows up in shell history but no logged run for this project came from it.

> **Corrected 2026-07-29.** An earlier version of this table attributed pretraining to the repo
> `KoyejoLab-Pretraining-Variance`. That was a misreading of `wandb-metadata.json`: the `executable`
> field points at `KoyejoLab-Pretraining-Variance/pt_var_env/bin/python3`, but the `program` field —
> the code that actually ran — is
> `/lfs/skampere2/0/rschaef/KoyejoLab-Memorization-Scoring-vs-Sampling/scripts/pretrain_language_model.py`,
> with git remote `RylanSchaeffer/KoyejoLab-Memorization-Scoring-vs-Sampling.git`. That is **this
> repository**, under its former name, checked out on skampere2; only the Python interpreter was
> borrowed from the sibling project. `KoyejoLab-Pretraining-Variance` is a different experiment and
> its pretraining script cannot even emit `eval_after/eval_benchmark_loss` (it passes a bare
> `eval_dataset` to the Trainer). **All contamination pretraining came from this repo.**
>
> Read `program` and `git.remote`, not `executable`, when attributing a run.

**Pretraining is versioned by commit `934546a` (2026-01-19 11:36).** Everything before it is "v1"
(`warmup_steps: 250`, `weight_decay: 0`, Adam betas left at HuggingFace defaults) and logged to
`...-pt`; everything after is "v2" (`adam_beta1/2`, `warmup_ratio: 0.2`, `full_determinism`,
`weight_decay: 0.01`) and logged to `...-pt-v2`. **The published Fig. 3 runs are v1.** To reproduce
that setup use `git show 934546a^:scripts/pretrain_language_model.py` and
`git show 934546a^:src/globals.py`; the current script will `KeyError` on v1 sweep YAMLs because it
requires the four v2 keys. See `reviews/2026_neurips/MISSING_PRETRAINING_DATA.md`.

## Node hardware and filesystem layout

| Node | GPUs | Notes |
|---|---|---|
| skampere1 | 8× A100-SXM4-80GB | This paper's evals/SFT |
| skampere2 | 8× H200 | Pretraining (from other repo checkouts) |
| skampere3 | 8× B200 | Blackwell — verify vLLM/CUDA support before relying on it |

All three are heavily contended; expect roughly one fully free GPU per node at any moment. Check with
`nvidia-smi` before launching, and prefer many short single-GPU jobs over whole-node jobs.

Three filesystems, and the distinction matters:

| Mount | Scope | Size | Speed |
|---|---|---|---|
| `/afs/cs.stanford.edu/u/rschaef` | every machine | **5 GB quota** | fast |
| `/lfs/<host>/0/rschaef` | **per-machine** | hundreds of GB | fast |
| `/dfs/scratch0/rschaef` | every machine | massive | **very slow** |

Anything large or hot belongs on `/lfs`. AFS fills at 5 GB and then fails in confusing ways — `tar`
reports `Unknown system error -122`, which is `EDQUOT`, and tools that don't check write truncated files.
AFS also does not reliably serve `mmap` for large executables. DFS is shared but too slow for anything
latency-sensitive.

`~/.bashrc.lfs` derives `LFS_HOME` from the hostname, so the same rc file works on every node.

## Installing Claude Code on a node

Claude Code ships as a Bun single-file executable that mmaps itself at startup, so it must live on `/lfs`,
never AFS — on AFS you get `Bus error` at launch, either from a quota-truncated binary or from AFS's mmap
behaviour. `/lfs` is per-machine, so run this once per node:

```bash
bash scripts/setup_claude_node.sh
```

Idempotent. Installs nvm + Node 24 + Claude Code under `$LFS_HOME/nvm`, verifies the binary is not
truncated, and does not touch any shell rc file. To update later, on that node:

```bash
npm install -g @anthropic-ai/claude-code@latest
```

Installed and verified on skampere1, skampere2, and skampere3 (2026-07-27, v2.1.220). Note there is also a
root-owned `/usr/local/bin/claude` (v2.1.75) on these machines that you cannot update; the `/lfs` install
shadows it because nvm's bin comes earlier on `PATH`.

## Getting on the node

SSH aliases are defined in `~/.bashrc` on the local workstation:

```bash
snapskampere1     # rschaef@skampere1.stanford.edu   <- this paper's evals/SFT
snapskampere2     # rschaef@skampere2.stanford.edu   <- pretraining (other repos)
snapskampere3     # rschaef@skampere3.stanford.edu
snaphyperturing1  # rschaef@hyperturing1.stanford.edu
snaphyperturing2  # rschaef@hyperturing2.stanford.edu
```

> ⚠️ Those aliases currently embed a plaintext password via `sshpass` in `~/.bashrc`. Consider replacing
> them with SSH keys and a `~/.ssh/config` block.

Once on skampere1:

```bash
cd /lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization
source mem_scoring_vs_sampling_env/bin/activate     # uv venv, NOT conda
```

The environment is a **uv venv**, not a conda env — `conda activate` will not find it. Confirmed via the
logged interpreter path `.../mem_scoring_vs_sampling_env/bin/python`.

### Gotcha: `source activate` silently does nothing when the interpreter symlink dangles

On 2026-07-27 `source mem_scoring_vs_sampling_env/bin/activate` left `which python` pointing at
miniconda, and every script failed with `ModuleNotFoundError: No module named 'editdistance'`. The venv
was fine — its 11 GB of `site-packages` was intact — but `bin/python` was a dangling symlink into
`/afs/.../.local/share/uv/python/cpython-3.11.5-linux-x86_64-gnu/`, which had been emptied. `activate`
prepends a `bin/` whose `python` does not resolve, so the shell falls through to the next `PATH` entry
and reports no error.

`UV_PYTHON_INSTALL_DIR` is now `/lfs/skampere1/0/rschaef/uv-python`, so the venv was repointed there:

```bash
uv python install 3.11.5     # lands in $UV_PYTHON_INSTALL_DIR, not AFS
ln -sf /lfs/skampere1/0/rschaef/uv-python/cpython-3.11.5-linux-x86_64-gnu/bin/python3.11 \
       mem_scoring_vs_sampling_env/bin/python
# and edit `home = ...` in mem_scoring_vs_sampling_env/pyvenv.cfg to match
```

Diagnose with `ls -la mem_scoring_vs_sampling_env/bin/python` — if the target does not exist, this is
the failure. Invoking `./mem_scoring_vs_sampling_env/bin/python` by absolute path is a reliable way to
run scripts without depending on `activate` succeeding.

## Weights & Biases projects

Entity is `rylan` throughout.

| Project | Contents | Status |
|---|---|---|
| `memorization-scoring-vs-sampling-eval` | Generative (Math Verify) evals | 1,296 runs; 1,270 finished |
| `memorization-scoring-vs-sampling-eval-teacher-forcing` | Teacher-forced NLL evals | active |
| `memorization-scoring-vs-sampling-sft` | SFT runs | active |
| `memorization-scoring-vs-sampling-pt-v2` | Pretraining (v2) | active |
| `scaling-memorization-pt` / `scaling-memorization-eval` | `scale_mem_*` sweeps (separate project) | active |
| `memorization-scoring-vs-sampling-pt` | — | **Does not exist under `rylan`** — but see below; it exists under `jkazdan`. Referenced 16× in `notebooks/10_*` and `sweeps/pt/*.yaml`. |

### The "missing" pretraining project is a different entity — but that is not the whole story

Verified 2026-07-27. The notebooks resolve project paths against `wandb.api.default_entity`
(= `rylan`), which is why `memorization-scoring-vs-sampling-pt` looks absent:

| Path | Runs | Contains notebook 11's 15 PT sweep IDs? |
|---|---|---|
| `rylan/memorization-scoring-vs-sampling-pt` | — | project does not exist |
| `jkazdan/memorization-scoring-vs-sampling-pt` | **236** across 57 sweeps | **No — 0 of 15** |
| `joshteam/memorization-scoring-vs-sampling-pt` | 0 | no |
| `rylan/memorization-scoring-vs-sampling-pt-v2` | 68 across 6 sweeps | no — 0 of 15 |

So repointing the entity alone will **not** fix notebook 11: the specific sweeps it names
(`rkx5xfde`, `g31f7bsb`, `09c432gh`, …) are not in any project this API key can reach. Its
pretraining configs survive only in the local cache at
`notebooks/11_math_qwen3_pt_math_verify/data/c39ba9b590fe96b52183328d3d4c7323_runs_configs.csv`.
Treat that cache as the last copy and do not delete it.

Entities this API key can see, for future searches: teams `harvardparkesateams`, `kreiman-sdm`,
`fiete-lab`, `projectdeus`, `joshteam`, `brando-su`, plus `rylan`, `jkazdan`, `stellaathena`.
`tensorpool` and `koyejolab` return zero visible projects — relevant because
`07_acknowledgements.tex` credits Table 1's compute to TensorPool.

### Gotcha: the local `wandb` API is broken under Anaconda

On the local workstation, `import wandb` fails with a protobuf descriptor error. Prefix commands:

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python your_script.py
```

Also: this `wandb` version has **no `run.metadata` attribute**. To read host/environment info, download
the file instead:

```python
md = json.load(open(run.file("wandb-metadata.json").download(root=..., replace=True).name))
md["host"], md["executable"], md["gpu"], md["gpu_count"]
```

Large `api.runs(...)` iterations time out at the default 60s. Pass `wandb.Api(timeout=120)` and page with
`per_page=...`, or the sweep-expansion step will hang.

## Sweep ID reference

Sweep IDs are recorded inline in the notebooks that consume them. The ones that matter:

| Sweep | What | Consumed by |
|---|---|---|
| `qx2c4702` (+ per-size siblings) | 4-shot boxed-required generative eval, pretrained models | `notebooks/11_*` |
| `2zpwcnek` | 4-shot boxed-required generative eval, **SFT** models | `notebooks/13_*` |
| `onaspopu` | Teacher-forced perturbed-MATH eval of SFT checkpoints (34 runs) | `notebooks/16_*` |

Notebooks keep older pre-4-shot sweep IDs commented out directly above the active list. **The 0-shot
results are superseded**; don't mix them with 4-shot numbers.

## HuggingFace Hub

Checkpoints auto-upload at end of training (`hub_strategy: end`, `save_strategy: no` — so there are **no
intermediate checkpoints**, only final). Requires `HF_TOKEN`. All under the `RylanSchaeffer/` namespace.

Evaluation datasets used as memorization controls:
- `RylanSchaeffer/math_rephrased` — same math, different wording
- `RylanSchaeffer/math_perturbed` — same wording, different numbers

Both supersede the earlier `stellaathena/*` versions, which had severe quality problems (see
`reviews/2026_icml/REVIEWER_6RQA/`). Do not use the `stellaathena/*` datasets.
