# Roadmap

The resume claim this project earns:

> Trained PPO agents to land a rocket in Box2D, deliberately induced reward hacking
> (hovering to farm a shaped bonus), built an automatic detector for it, and fixed it
> with potential-based reward shaping. Results across 5 seeds with 95% confidence bands.

Who writes what: **you** write everything marked ✍️. Claude sets up tooling, reviews
your code, and writes the plumbing marked 🔧.

Every phase ends with **interview questions**. Answer them out loud before moving on.
If you can't, the phase isn't done.

---

## Phase 0: Setup ✅

`uv` project, Python 3.12, Gymnasium 1.4 (Box2D), Stable-Baselines3 2.9, PyTorch.

```bash
uv run python -c "import gymnasium, stable_baselines3"
```

## Phase 1: Watch it fail ✅

Write `scripts/random_agent.py`:

1. Make `LunarLander-v3` with `render_mode="human"`.
2. Run 5 episodes of random actions (`env.action_space.sample()`).
3. For each episode, print: total reward, number of steps, and whether it ended by
   `terminated` or `truncated`.
4. On the first step of episode 1, print the 8 observation numbers with labels.
5. Pass `seed=` to the first `reset` so a run can be repeated exactly.

Run it with `uv run python scripts/random_agent.py`.

Interview questions:
- What's the difference between `terminated` and `truncated`? Why does PPO care?
- Roughly what does a random agent score? Why is it negative?
- Which observation numbers are continuous and which are booleans?

## Phase 2: Baseline PPO ✍️ + 🔧 ← you are here

- ✍️ `src/lunar_lab/train.py`: PPO with `MlpPolicy`, 8 parallel envs, a `--seed` flag,
  saves the model and the Monitor logs under `runs/<variant>/seed<N>/`.
- 🔧 Config file and CLI wiring.
- Train 1M steps. Time it. Use `device="cpu"`: for a 64x64 MLP, CPU beats MPS.

Interview questions: What does PPO's clip range do? Why do parallel envs help?
What is `ep_rew_mean` averaged over?

## Phase 3: Evaluation harness ✍️

`src/lunar_lab/evaluate.py` always evaluates on the **unwrapped** env, so your own
bonuses can't inflate the score. It runs 100 episodes and reports:

| Metric | Why it matters |
|---|---|
| mean ± std return | the standard "solved" metric (≥ 200) |
| landing rate | share of episodes ending with a safe landing |
| crash rate | share ending with a crash |
| **timeout rate** | share hitting the 1,000-step limit: **the hover signature** |
| mean episode length | hovering makes this long |
| fuel used | main and side engine firings |

## Phase 4: Reward wrappers ✍️

In `src/lunar_lab/rewards.py`, each a `gymnasium.Wrapper`:

1. `TimePenalty`: small cost every frame. Should still land.
2. `UprightBonus`: +1 per frame when upright and slow. **The hack.** Expect hovering.
3. `PotentialShaping`: the fix. Add `F = γ·Φ(s') − Φ(s)` with Φ = "upright and slow".
   Ng, Harada & Russell (1999) prove this kind of shaping cannot change which policy
   is optimal. This is the most important idea in the project.

🔧 Unit tests that pin each wrapper's arithmetic on hand-built observations.

Interview questions: Why does UprightBonus make hovering optimal? (Do the 1000 vs
300 arithmetic.) Why can't potential-based shaping be hacked? What happens to the
shaping terms when you add them up over a whole episode?

## Phase 5: Experiments 🔧 + ✍️

- 4 variants (baseline, time penalty, hack, fix) × 5 seeds.
- 🔧 Runner script and learning-curve plots with 95% confidence bands.
- ✍️ Read the results and write the analysis. That part can't be delegated.

## Phase 6: Reward-hack detector ✍️

A function that takes a trained run and flags it when the **shaped** return climbs
while the **true** return or landing rate stays flat, or when the timeout rate goes
above a threshold. Test it on your hack and fix runs: it should catch one and pass
the other.

## Phase 7: Ship it 🔧 + ✍️

- GitHub Actions: ruff + pytest + a 5k-step smoke train.
- Clips: random crash, baseline landing, hack hover, fix landing.
- A write-up page: the problem, the arithmetic, the curves, the clips.
- ✍️ README in your own words, plus the resume bullet.
