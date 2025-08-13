
# How a sweep works
# 1) You create a sweep from a YAML → W&B returns a SWEEP_ID.
# 2) You launch one or more agents with that SWEEP_ID.
# 3) Each agent pulls the next unclaimed config from W&B, runs it to completion, reports metrics, then pulls another, serially.
# 4) To run in parallel, you start multiple agents (one per GPU or node). The W&B backend handles assigning distinct runs to each agent.

# Activate your env
source $HOME/uv_envs/mem_scoring_vs_sampling_env/bin/activate

# Create a sweep
cd $HOME/KoyejoLab-Memorization-Scoring-vs-Sampling
wandb sweep sweeps/dataset=math_debug-sgd.yaml --name sgd_test
# copy paste it manually
export sweep_id=c9fpt236

# Launch agent from sweep_id
source $HOME/uv_envs/mem_scoring_vs_sampling_env/bin/activate
echo $sweep_id
cd $HOME/KoyejoLab-Memorization-Scoring-vs-Sampling
export CUDA_VISIBLE_DEVICES=3
wandb agent brando-su/memorization-scoring-vs-sampling/$sweep_id

