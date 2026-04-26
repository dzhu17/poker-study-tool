"""
Runs four reward-shaping configs sequentially, then runs a round-robin
tournament between all trained models (including the previous best) to
determine the true winner via head-to-head play.
"""

import subprocess
import sys
import os
import time
import numpy as np
import shutil
import torch

CONFIGS = [
    {
        "name": "optA_calibrated",
        "output_dir": "models/config17_optA",
        "args": [
            "--entropy_coef", "0.10", "--lr", "3e-4",
            "--sb_open_raise_bonus",  "0.25",
            "--sb_open_call_penalty", "1.00",
            "--bb_rfi_raise_bonus",   "0.06",
            "--bb_rfi_call_penalty",  "0.40",
            "--sv3_raise_bonus",      "0.25",
            "--sv3_call_penalty",     "0.45",
        ],
    },
    {
        "name": "optB_firm",
        "output_dir": "models/config18_optB",
        "args": [
            "--entropy_coef", "0.10", "--lr", "3e-4",
            "--sb_open_raise_bonus",  "0.25",
            "--sb_open_call_penalty", "1.25",
            "--bb_rfi_raise_bonus",   "0.06",
            "--bb_rfi_call_penalty",  "0.45",
            "--sv3_raise_bonus",      "0.25",
            "--sv3_call_penalty",     "0.55",
        ],
    },
    {
        "name": "optC_max",
        "output_dir": "models/config19_optC",
        "args": [
            "--entropy_coef", "0.10", "--lr", "3e-4",
            "--sb_open_raise_bonus",  "0.30",
            "--sb_open_call_penalty", "1.50",
            "--bb_rfi_raise_bonus",   "0.05",
            "--bb_rfi_call_penalty",  "0.40",
            "--sv3_raise_bonus",      "0.30",
            "--sv3_call_penalty",     "0.65",
        ],
    },
]

# Include previous two best models as tournament reference baselines
TOURNAMENT_CONFIGS = CONFIGS + [
    {
        "name": "prev_gto_light",
        "output_dir": "models/config14_gto_light",
    },
    {
        "name": "prev_best_moderate",
        "output_dir": "models/config11_moderate",
    },
]

EPISODES            = "500000"
EARLY_STOP_PATIENCE = "99999"
LOG_DIR             = "models/logs"
PYTHON              = sys.executable
TOURNAMENT_HANDS    = 10000  # hands per matchup


def run_config(config):
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{config['name']}.txt")

    cmd = [
        PYTHON, "src/train.py",
        "--episodes",            EPISODES,
        "--output_dir",          config["output_dir"],
        "--early_stop_patience", EARLY_STOP_PATIENCE,
    ] + config["args"]

    print(f"\n{'='*60}")
    print(f"Starting config: {config['name']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Log: {log_path}")
    print(f"{'='*60}\n")

    start = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - start

    print(f"Config '{config['name']}' finished in {elapsed/60:.1f} min (exit code {proc.returncode})")
    return proc.returncode


def best_by_reward():
    """Pick config with highest mean reward over last 10k episodes (training-only configs)."""
    results = {}
    for config in CONFIGS:
        path = os.path.join(config["output_dir"], "episode_rewards.npy")
        if not os.path.exists(path):
            print(f"  WARNING: no rewards file for {config['name']}, skipping")
            continue
        rewards = np.load(path)
        n = min(10000, len(rewards))
        mean = np.mean(rewards[-n:])
        results[config["name"]] = (mean, config)
        print(f"  {config['name']:<20} final avg reward (last {n}): {mean:+.4f}")

    if not results:
        return None
    best_name = max(results, key=lambda k: results[k][0])
    return results[best_name][1]


# ---------------------------------------------------------------------------
# Round-robin tournament
# ---------------------------------------------------------------------------

def _load_network(model_path):
    sys.path.insert(0, "src")
    from ppo_agent import ActorCritic
    checkpoint = torch.load(model_path, map_location="cpu")
    net = ActorCritic()
    net.load_state_dict(checkpoint["network_state"])
    net.eval()
    return net


def _play_matchup(net1, net2, num_hands):
    """Play net1 vs net2. Returns net1's mean reward (zero-sum)."""
    sys.path.insert(0, "src")
    from poker_env import PokerPreflopEnv
    env = PokerPreflopEnv()
    rewards = []

    for hand in range(num_hands):
        seat1 = hand % 2
        obs, _ = env.reset()
        while True:
            current = env.current_player
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                if current == seat1:
                    action, _, _ = net1.get_action(obs_t)
                else:
                    action, _, _ = net2.get_action(obs_t)
            obs, reward, done, _, _ = env.step(action.item())
            if done:
                rewards.append(reward if current == seat1 else -reward)
                break

    return float(np.mean(rewards))


def run_tournament():
    print("\n" + "="*60)
    print("Round-Robin Tournament")
    print(f"Hands per matchup: {TOURNAMENT_HANDS:,}")
    print("="*60)

    # Load all available models
    networks = {}
    for config in TOURNAMENT_CONFIGS:
        path = os.path.join(config["output_dir"], "agent.pth")
        if not os.path.exists(path):
            print(f"  Skipping {config['name']} — no model found")
            continue
        networks[config["name"]] = _load_network(path)
        print(f"  Loaded: {config['name']}")

    names = list(networks.keys())
    n = len(names)
    if n < 2:
        print("Need at least 2 models for a tournament.")
        return None

    scores = {name: 0.0 for name in names}
    wins   = {name: 0   for name in names}

    print(f"\nPlaying {n*(n-1)//2} matchups...\n")
    for i in range(n):
        for j in range(i + 1, n):
            name1, name2 = names[i], names[j]
            avg = _play_matchup(networks[name1], networks[name2], TOURNAMENT_HANDS)
            scores[name1] += avg
            scores[name2] -= avg
            winner = name1 if avg > 0 else (name2 if avg < 0 else "tie")
            if avg > 0:
                wins[name1] += 1
            elif avg < 0:
                wins[name2] += 1
            print(f"  {name1:<22} vs {name2:<22}  {avg:+.4f} bb/hand  -> {winner}")

    print("\n" + "-"*60)
    print(f"{'Rank':<5} {'Model':<24} {'Score':>10} {'Wins':>6}")
    print("-"*60)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    for rank, (name, score) in enumerate(ranked, 1):
        print(f"  {rank:<3}  {name:<24} {score:>+10.4f}  {wins[name]:>4}/{n-1}")

    tournament_winner = ranked[0][0]
    print(f"\nTournament winner: {tournament_winner}")
    return tournament_winner, {c["name"]: c for c in TOURNAMENT_CONFIGS if c["name"] in networks}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    total_start = time.time()
    print("="*60)
    print("PPO Self-Play Training — GTO Fine-Tuned Sweep")
    print(f"Episodes per config: {int(EPISODES):,}")
    print("="*60)

    for config in CONFIGS:
        run_config(config)

    print("\n" + "="*60)
    print("All configs complete. Selecting best by training reward...")
    best = best_by_reward()

    if best:
        print(f"\nBest by reward: {best['name']}")
        src = os.path.join(best["output_dir"], "agent.pth")
        dst = os.path.join("models", "agent.pth")
        rewards_src = os.path.join(best["output_dir"], "episode_rewards.npy")
        rewards_dst = os.path.join("models", "episode_rewards.npy")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Reward-best model copied to {dst}")
        if os.path.exists(rewards_src):
            shutil.copy2(rewards_src, rewards_dst)

    # Run head-to-head tournament for true ranking
    result = run_tournament()
    if result:
        tournament_winner, config_map = result
        if tournament_winner in config_map:
            winner_config = config_map[tournament_winner]
            src = os.path.join(winner_config["output_dir"], "agent.pth")
            dst = os.path.join("models", "agent.pth")
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"\nTournament winner '{tournament_winner}' copied to {dst}")

    total_mins = (time.time() - total_start) / 60
    print(f"\nTotal time: {total_mins:.1f} min")
    print("Run src/evaluate.py to see full results.")


if __name__ == "__main__":
    main()
