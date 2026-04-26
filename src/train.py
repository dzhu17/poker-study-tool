"""
Self-play PPO training for PokerPreflopEnv.

The learning agent alternates positions each episode (SB and BB) so it learns
both seats. The opponent is a frozen copy of the agent network, updated every
--opponent_update_freq episodes to match the latest weights.
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from poker_env import PokerPreflopEnv
from ppo_agent import PPOAgent, ActorCritic


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Self-play PPO training for preflop poker")
    p.add_argument("--episodes",              type=int,   default=50000)
    p.add_argument("--rollout_size",          type=int,   default=512)
    p.add_argument("--lr",                    type=float, default=3e-4)
    p.add_argument("--gamma",                 type=float, default=0.99)
    p.add_argument("--gae_lambda",            type=float, default=0.95)
    p.add_argument("--clip_eps",              type=float, default=0.2)
    p.add_argument("--entropy_coef",          type=float, default=0.01)
    p.add_argument("--value_coef",            type=float, default=0.5)
    p.add_argument("--update_epochs",         type=int,   default=4)
    p.add_argument("--minibatch_size",        type=int,   default=64)
    p.add_argument("--dropout",               type=float, default=0.1)
    p.add_argument("--opponent_update_freq",  type=int,   default=500,
                   help="Episodes between opponent weight refreshes")
    p.add_argument("--checkpoint_freq",       type=int,   default=5000,
                   help="Episodes between checkpoint saves")
    p.add_argument("--log_freq",              type=int,   default=500,
                   help="Episodes between log lines")
    p.add_argument("--early_stop_patience",   type=int,   default=20,
                   help="Evaluation windows with no improvement before stopping")
    p.add_argument("--early_stop_min_delta",  type=float, default=0.005)
    p.add_argument("--eval_window",           type=int,   default=500,
                   help="Episodes per early-stopping evaluation window")
    p.add_argument("--raise_bonus",            type=float, default=0.0)
    p.add_argument("--call_penalty",           type=float, default=0.0)
    # Scenario-specific shaping (GTO-informed)
    p.add_argument("--sb_open_raise_bonus",    type=float, default=0.0,
                   help="Bonus for SB opening raise (GTO: raise 78%)")
    p.add_argument("--sb_open_call_penalty",   type=float, default=0.0,
                   help="Penalty for SB limping (GTO: never limp HU)")
    p.add_argument("--pair_raise_bonus",       type=float, default=0.0,
                   help="Extra bonus for SB opening any pocket pair (GTO: always raise)")
    p.add_argument("--bb_rfi_raise_bonus",     type=float, default=0.0,
                   help="Bonus for BB 3betting vs SB open (GTO: ~5-6%)")
    p.add_argument("--bb_rfi_call_penalty",    type=float, default=0.0,
                   help="Penalty for BB calling SB open (overcome sunk blind bias)")
    p.add_argument("--sv3_raise_bonus",        type=float, default=0.0,
                   help="Bonus for SB 4betting vs BB 3bet (GTO: premiums only)")
    p.add_argument("--sv3_call_penalty",       type=float, default=0.0,
                   help="Penalty for SB calling BB 3bet (GTO: mostly fold or 4bet)")
    p.add_argument("--lr_schedule",           type=int,   default=1,
                   help="1=cosine annealing LR schedule, 0=constant LR")
    p.add_argument("--seed",                  type=int,   default=42)
    p.add_argument("--output_dir",            type=str,   default="models")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _opponent_action(network: ActorCritic, obs: np.ndarray) -> int:
    obs_t = torch.FloatTensor(obs).unsqueeze(0)
    with torch.no_grad():
        action_t, _, _ = network.get_action(obs_t)
    return action_t.item()


def run_episode(env: PokerPreflopEnv, agent: PPOAgent,
                opponent: ActorCritic, learning_player: int):
    """
    Run one episode.

    The learning agent controls `learning_player` (0 or 1); the frozen
    opponent controls the other seat.

    Returns:
        transitions: list of (obs, action, log_prob, value, reward, done)
                     containing only the learning agent's steps.
        episode_reward: scalar terminal reward for the learning agent.
    """
    obs, _ = env.reset()
    transitions = []

    while True:
        current = env.current_player

        if current == learning_player:
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, done, _, _ = env.step(action)

            if done:
                transitions.append((obs, action, log_prob, value, reward, True))
                return transitions, reward
            else:
                # Non-terminal raise: carry env reward (raise_bonus if set)
                transitions.append((obs, action, log_prob, value, reward, False))
                obs = next_obs

        else:
            opp_action = _opponent_action(opponent, obs)
            next_obs, opp_reward, done, _, _ = env.step(opp_action)

            if done:
                # Zero-sum: learning agent's payoff = -opponent's payoff
                p_reward = -opp_reward
                if transitions:
                    o, a, lp, v, prev_r, _ = transitions[-1]
                    # Accumulate: preserve raise_bonus from the prior raise step
                    transitions[-1] = (o, a, lp, v, prev_r + p_reward, True)
                # If transitions is empty (opponent ended before learning agent
                # ever acted), return empty list — nothing to train on.
                return transitions, p_reward
            else:
                obs = next_obs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _smooth(values, window=100):
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def save_training_curves(episode_rewards, loss_log, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("PPO Self-Play Training Curves", fontsize=14)

    # --- Reward ---
    ax = axes[0, 0]
    ax.plot(episode_rewards, alpha=0.2, color="steelblue")
    smoothed = _smooth(episode_rewards, 200)
    if len(smoothed) > 0:
        offset = len(episode_rewards) - len(smoothed)
        ax.plot(range(offset, offset + len(smoothed)), smoothed,
                color="steelblue", label="Smoothed (200)")
    ax.set_title("Episode Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (bb)")
    ax.legend()

    # --- Policy loss ---
    ax = axes[0, 1]
    ax.plot(loss_log["policy_loss"], color="tomato")
    ax.set_title("Policy Loss")
    ax.set_xlabel("PPO Update")

    # --- Value loss ---
    ax = axes[1, 0]
    ax.plot(loss_log["value_loss"], color="orange")
    ax.set_title("Value Loss")
    ax.set_xlabel("PPO Update")

    # --- Entropy ---
    ax = axes[1, 1]
    ax.plot(loss_log["entropy"], color="seagreen")
    ax.set_title("Entropy (exploration)")
    ax.set_xlabel("PPO Update")

    # --- LR schedule (shown on entropy axis if using cosine schedule) ---
    if loss_log.get("lr") and len(set(loss_log["lr"])) > 1:
        ax2 = ax.twinx()
        ax2.plot(loss_log["lr"], color="purple", alpha=0.5, linestyle="--", label="LR")
        ax2.set_ylabel("Learning Rate", color="purple")
        ax2.tick_params(axis="y", labelcolor="purple")

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Training curves saved to {path}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    env = PokerPreflopEnv(
        raise_bonus=args.raise_bonus,
        call_penalty=args.call_penalty,
        sb_open_raise_bonus=args.sb_open_raise_bonus,
        sb_open_call_penalty=args.sb_open_call_penalty,
        pair_raise_bonus=args.pair_raise_bonus,
        bb_rfi_raise_bonus=args.bb_rfi_raise_bonus,
        bb_rfi_call_penalty=args.bb_rfi_call_penalty,
        sv3_raise_bonus=args.sv3_raise_bonus,
        sv3_call_penalty=args.sv3_call_penalty,
    )

    # Estimate total PPO updates for LR scheduling (episodes / steps-per-update)
    estimated_updates = max(1, (args.episodes * 2) // args.rollout_size)
    agent = PPOAgent(
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        dropout=args.dropout,
        lr_schedule_steps=estimated_updates if args.lr_schedule else 0,
    )

    # Frozen opponent — periodically synced to latest agent weights
    opponent = ActorCritic()
    opponent.load_state_dict(agent.network.state_dict())
    opponent.eval()

    episode_rewards = []
    loss_log = {"policy_loss": [], "value_loss": [], "entropy": [], "lr": []}

    best_avg_reward = -np.inf
    patience_counter = 0
    buffer_steps = 0
    update_count = 0

    print(f"Starting self-play PPO training")
    print(f"  Episodes:        {args.episodes}")
    print(f"  Rollout size:    {args.rollout_size}")
    print(f"  Opponent update: every {args.opponent_update_freq} episodes")
    print(f"  Output dir:      {args.output_dir}")
    print("-" * 60)

    for ep in range(args.episodes):
        learning_player = ep % 2  # alternate seats every episode

        transitions, ep_reward = run_episode(env, agent, opponent, learning_player)

        for (obs, action, log_prob, value, reward, done) in transitions:
            agent.buffer.add(obs, action, log_prob, reward, value, float(done))
            buffer_steps += 1

        episode_rewards.append(ep_reward)

        # PPO update
        if buffer_steps >= args.rollout_size:
            stats = agent.update(next_value=0.0)
            loss_log["policy_loss"].append(stats["policy_loss"])
            loss_log["value_loss"].append(stats["value_loss"])
            loss_log["entropy"].append(stats["entropy"])
            loss_log["lr"].append(stats.get("lr", args.lr))
            buffer_steps = 0
            update_count += 1

        # Sync opponent to latest weights
        if ep % args.opponent_update_freq == 0 and ep > 0:
            opponent.load_state_dict(agent.network.state_dict())
            opponent.eval()

        # Logging
        if (ep + 1) % args.log_freq == 0:
            recent = episode_rewards[-args.log_freq:]
            avg = np.mean(recent)
            print(f"Ep {ep+1:>6}/{args.episodes} | "
                  f"Avg reward: {avg:+.3f} | "
                  f"Updates: {update_count} | "
                  f"Entropy: {loss_log['entropy'][-1]:.3f}" if loss_log["entropy"]
                  else f"Ep {ep+1:>6}/{args.episodes} | Avg reward: {avg:+.3f}")

        # Checkpoint
        if (ep + 1) % args.checkpoint_freq == 0:
            ckpt = os.path.join(args.output_dir, f"checkpoint_ep{ep+1}.pth")
            agent.save(ckpt)
            print(f"  Checkpoint saved to {ckpt}")

        # Early stopping
        if (ep + 1) % args.eval_window == 0:
            avg_reward = np.mean(episode_rewards[-args.eval_window:])
            if avg_reward > best_avg_reward + args.early_stop_min_delta:
                best_avg_reward = avg_reward
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.early_stop_patience:
                    print(f"\nEarly stopping at episode {ep+1}: "
                          f"no improvement for {patience_counter} eval windows.")
                    break

    # --- Save final artifacts ---
    final_model = os.path.join(args.output_dir, "agent.pth")
    agent.save(final_model)
    print(f"\nFinal model saved to {final_model}")

    rewards_path = os.path.join(args.output_dir, "episode_rewards.npy")
    np.save(rewards_path, np.array(episode_rewards))
    print(f"Reward history saved to {rewards_path}")

    save_training_curves(episode_rewards, loss_log, args.output_dir)

    n = min(1000, len(episode_rewards))
    print(f"\nTraining complete | {len(episode_rewards)} episodes | "
          f"Final avg reward (last {n}): {np.mean(episode_rewards[-n:]):+.3f}")


if __name__ == "__main__":
    main()
