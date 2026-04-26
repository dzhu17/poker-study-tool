"""
Smoke test for PokerPreflopEnv.
Runs random episodes and confirms: observations have correct shape,
episodes always terminate, rewards are finite.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from poker_env import PokerPreflopEnv, NUM_ACTIONS

NUM_EPISODES = 1000


def run_tests():
    env = PokerPreflopEnv()
    obs_shape = env.observation_space.shape

    episode_lengths = []
    rewards_seen = []

    for ep in range(NUM_EPISODES):
        obs, info = env.reset(seed=ep)

        assert obs.shape == obs_shape, f"Bad obs shape: {obs.shape}"
        assert not np.any(np.isnan(obs)), "NaN in initial obs"

        steps = 0
        terminated = False
        while not terminated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == obs_shape, f"Bad obs shape at step {steps}"
            assert np.isfinite(reward), f"Non-finite reward: {reward}"
            steps += 1

            if steps > 50:
                raise RuntimeError(f"Episode {ep} didn't terminate after 50 steps")

        episode_lengths.append(steps)
        rewards_seen.append(reward)

    print(f"Ran {NUM_EPISODES} episodes — all terminated successfully.")
    print(f"Episode length  min={min(episode_lengths)}  max={max(episode_lengths)}  "
          f"mean={np.mean(episode_lengths):.2f}")
    print(f"Final reward    min={min(rewards_seen):.2f}  max={max(rewards_seen):.2f}  "
          f"mean={np.mean(rewards_seen):.2f}")

    # Verify observation space bounds registration
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Spot-check render doesn't crash
    env.reset(seed=42)
    env.render()

    print("\nAll tests passed.")


if __name__ == "__main__":
    run_tests()