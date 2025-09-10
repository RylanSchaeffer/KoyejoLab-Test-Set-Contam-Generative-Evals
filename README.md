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
# Newer versions of Flash Attention didn't work for me on SNAP. See: https://github.com/Dao-AILab/flash-attention/issues/1708.
uv pip install flash-attn==2.7.2.post1 --no-build-isolation
```

