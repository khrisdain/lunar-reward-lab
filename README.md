# lunar-reward-lab

Reward shaping and reward hacking experiments on Gymnasium's `LunarLander-v3`, trained
with PPO (Stable-Baselines3).

The plan: train an agent to land, deliberately shape the reward so it learns to hover
instead (reward hacking), detect that automatically, and fix it with potential-based
reward shaping. Results are compared across multiple seeds and always evaluated on the
unmodified environment.

Work in progress. See [ROADMAP.md](ROADMAP.md) for the phases.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and `swig` (needed to build Box2D).

```bash
brew install uv swig
uv sync
uv run python -c "import gymnasium, stable_baselines3"
```
