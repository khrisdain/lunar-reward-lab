"""Phase 1: watch a random agent crash.

Runs a few episodes of LunarLander-v3 with uniformly random actions and prints,
for each episode, the total reward, the number of steps, and how it ended.

    uv run python scripts/random_agent.py              # opens the game window
    uv run python scripts/random_agent.py --no-render  # text only, much faster
"""

import argparse

import gymnasium as gym

# What each of the 8 observation numbers means (see the Gymnasium docs).
OBS_LABELS = [
    "x position (0 = pad centre)",
    "y position (height above pad)",
    "x velocity",
    "y velocity",
    "angle (radians)",
    "angular velocity",
    "left leg touching ground",
    "right leg touching ground",
]


def run_episode(env: gym.Env, seed: int | None, print_first_obs: bool) -> dict:
    """Play one episode with random actions and return a summary of it."""
    obs, _info = env.reset(seed=seed)

    if print_first_obs:
        print("First observation:")
        for label, value in zip(OBS_LABELS, obs, strict=True):
            print(f"  {label:<32} {value: .3f}")
        print()

    total_reward = 0.0
    steps = 0
    terminated = truncated = False

    # An episode ends one of two ways:
    #   terminated: the game itself ended it (crash, landing and coming to rest, or flying
    #               off-screen). There is no future after this state.
    #   truncated:  the 1,000-step time limit cut it off. The rocket could have kept going.
    # Phase 3 relies on this distinction: lots of truncations is the signature of hovering.
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _info = env.step(action)
        total_reward += float(reward)
        steps += 1

    return {
        "reward": total_reward,
        "steps": steps,
        "ended_by": "terminated" if terminated else "truncated",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-render", action="store_true", help="skip the game window")
    args = parser.parse_args()

    env = gym.make("LunarLander-v3", render_mode=None if args.no_render else "human")
    # Seed the action sampler too, so the whole run is repeatable, not only the terrain.
    env.action_space.seed(args.seed)

    results = []
    for episode in range(args.episodes):
        # Seed only the first reset. Later resets continue the same random stream,
        # so each episode differs, but the run as a whole is reproducible.
        seed = args.seed if episode == 0 else None
        result = run_episode(env, seed, print_first_obs=episode == 0)
        results.append(result)
        print(
            f"Episode {episode + 1}: reward {result['reward']:8.1f} | "
            f"steps {result['steps']:4d} | {result['ended_by']}"
        )

    env.close()

    mean_reward = sum(r["reward"] for r in results) / len(results)
    print(f"\nMean reward over {len(results)} episodes: {mean_reward:.1f} (solved is >= 200)")


if __name__ == "__main__":
    main()
