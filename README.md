# KoyejoLab-Scoring-vs-Sampling-Memorization

# Setup

0. Install `uv`: `conda install conda-forge::uv`
1. Create the virtual environment: `uv venv -p 3.11.5 mem_scoring_vs_sampling_env`
2. Activate the virtual environment: `source mem_scoring_vs_sampling_env/bin/activate`
3. Install the requirements: `uv pip install -r requirements.txt`
4. Then install Eleuther AI's Language Model (LM) Evaluation Harness:

```shell
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
uv pip install -e .[math]
# Newer versions didn't work for me. See: https://github.com/Dao-AILab/flash-attention/issues/1708.
#uv pip install flash_attn==2.7.4.post1 --no-build-isolation
```

# BM's Setup

This is my setup because my repos are in DFS and I point to them from LFS.

0. Install `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh`
1. Create local env in lfs: `mkdir -p $HOME/uv_envs`
2. Create the virtual environment: `uv venv -p 3.11.5 $HOME/uv_envs/mem_scoring_vs_sampling_env`
3. Activate the virtual environment: `source $HOME/uv_envs/mem_scoring_vs_sampling_env/bin/activate`
3. Install the requirements: `uv pip install -r requirements.txt`
4. Install this repo: `uv pip install -e .`
5. Then install Eleuther AI's Language Model (LM) Evaluation Harness:
```shell
cd $HOME
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd $HOME/lm-evaluation-harness
uv pip install -e .[math]
```