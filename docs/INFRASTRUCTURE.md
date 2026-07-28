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
| `memorization-scoring-vs-sampling-pt-v2` | skampere2 | 8× H200 | `/lfs/skampere2/0/rschaef/KoyejoLab-Pretraining-Variance` | `pt_var_env` (CPython 3.12.12) |
| `scaling-memorization-pt` | skampere2 | 8× H200 | `/lfs/skampere2/0/rschaef/KoyejoLab-Scaling-Memorization` | `scaling_mem_env` |

**For this paper's rebuttal work, the answer is `skampere1`.** All generative evaluation, teacher-forced
evaluation, and SFT for the submitted results ran there, out of a single checkout with a single environment.
`skampere3` shows up in shell history but no logged run for this project came from it.

Note that two of the five projects were driven from **different repositories** (`KoyejoLab-Pretraining-Variance`,
`KoyejoLab-Scaling-Memorization`) with their own environments. That is why pretraining configs and eval
configs don't share a Python version.

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

## Weights & Biases projects

Entity is `rylan` throughout.

| Project | Contents | Status |
|---|---|---|
| `memorization-scoring-vs-sampling-eval` | Generative (Math Verify) evals | 1,296 runs; 1,270 finished |
| `memorization-scoring-vs-sampling-eval-teacher-forcing` | Teacher-forced NLL evals | active |
| `memorization-scoring-vs-sampling-sft` | SFT runs | active |
| `memorization-scoring-vs-sampling-pt-v2` | Pretraining (v2) | active |
| `scaling-memorization-pt` / `scaling-memorization-eval` | `scale_mem_*` sweeps (separate project) | active |
| `memorization-scoring-vs-sampling-pt` | — | **DOES NOT EXIST.** Referenced 16× in `notebooks/10_*` and `sweeps/pt/*.yaml`; the API returns "Could not find project". Notebook 10 cannot refresh as written. |

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
