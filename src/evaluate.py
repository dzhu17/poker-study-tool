"""
Evaluation suite for the trained PPO poker agent.

Covers:
  1. Random baseline comparison (win rate / bb per hand)
  2. GTO accuracy vs preflop chart data
  3. Confusion matrix (PPO action vs GTO action)
  4. Error analysis heatmap (13x13 hand grid)
  5. Action distribution by scenario
  6. Training curves from saved reward history
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(__file__))
from poker_env import PokerPreflopEnv, FOLD, CALL, RAISE_33, RAISE_75, RAISE_150, ALL_IN
from ppo_agent import PPOAgent, ActorCritic

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCENARIOS = ["SB_RFI", "BB_vs_RFI", "SB_vs_3bet"]
SCENARIO_LABELS = {
    "SB_RFI":    "SB Open (vs no action)",
    "BB_vs_RFI": "BB vs SB Open",
    "SB_vs_3bet": "SB vs BB 3-Bet",
}
ACTION_NAMES = ["Fold", "Call", "Raise33", "Raise75", "Raise150", "All-In"]
GTO_ACTIONS  = ["fold", "call", "raise"]  # 3-category buckets

RANK_NAMES = {14: "A", 13: "K", 12: "Q", 11: "J", 10: "T",
              9: "9", 8: "8", 7: "7", 6: "6", 5: "5", 4: "4", 3: "3", 2: "2"}

OUTPUT_DIR = "models"
DATA_DIR   = "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hand_name(r1, r2, suited):
    n1, n2 = RANK_NAMES[r1], RANK_NAMES[r2]
    if r1 == r2:
        return f"{n1}{n2}"
    return f"{n1}{n2}s" if suited else f"{n1}{n2}o"


def hand_to_cards(r1, r2, suited):
    """Convert hand description to two specific card indices (no collision)."""
    from poker_env import card_to_idx
    if r1 == r2:
        return [card_to_idx(r1, 0), card_to_idx(r2, 1)]   # pair: different suits
    if suited:
        return [card_to_idx(r1, 0), card_to_idx(r2, 0)]   # same suit
    return [card_to_idx(r1, 0), card_to_idx(r2, 1)]        # different suits


def agent_action_to_gto(action_idx):
    """Map 6-action index to 3-category GTO label."""
    if action_idx == FOLD:
        return "fold"
    if action_idx == CALL:
        return "call"
    return "raise"


# ---------------------------------------------------------------------------
# GTO reference data generation
# ---------------------------------------------------------------------------

def _gto_sb_rfi(r1, r2, suited):
    """
    GTO SB open-raise action based on GGPoker 200NL 100bb HU chart.
    Fold 21.61% / Raise 2.5x 78.39% — NO limping (call never occurs).
    Fold region: weak offsuit hands with low ranks and poor connectivity.
    """
    is_pair = (r1 == r2)
    if is_pair or suited:
        return "raise"  # all pairs and suited hands raise

    # Offsuit folds — ~24 hand types = ~288 combos = 21.7% (matches 21.61%)
    # Low disconnected offsuit hands that can't profitably open at 2.5x
    fold_offsuit = {
        (9,2),(9,3),(9,4),(9,5),
        (8,2),(8,3),(8,4),(8,5),(8,6),
        (7,2),(7,3),(7,4),(7,5),(7,6),
        (6,2),(6,3),(6,4),(6,5),
        (5,2),(5,3),(5,4),
        (4,2),(4,3),
        (3,2),
    }
    if (r1, r2) in fold_offsuit:
        return "fold"
    return "raise"


def _gto_bb_vs_rfi(r1, r2, suited):
    """
    GTO BB action vs SB 2.5x open based on GGPoker 200NL 100bb HU chart.
    Target: Fold ~40% / Call ~34% / 3Bet ~26% (by combos).
    Note: real GTO uses mixed strategies; this is a pure-strategy approximation
    based on the dominant action visible in the GGPoker chart.
    """
    is_pair = (r1 == r2)

    if is_pair:
        if r1 >= 11:   return "raise"   # JJ+: 3bet
        if r1 >= 4:    return "call"    # 44-TT: call
        return "fold"                   # 22-33: fold

    if suited:
        if r1 == 14:                    # Axs
            if r2 >= 10:   return "raise"  # ATs+: 3bet
            if r2 >= 6:    return "call"   # A6s-A9s: call
            return "raise"                 # A2s-A5s: bluff 3bet
        if r1 == 13:                    # Kxs
            if r2 >= 12:   return "raise"  # KQs: 3bet
            if r2 >= 5:    return "call"   # K5s-KJs: call
            return "fold"
        if r1 == 12:                    # Qxs
            if r2 >= 7:    return "call"   # Q7s+: call
            return "fold"
        if r1 == 11:                    # Jxs
            if r2 >= 7:    return "call"   # J7s+: call
            return "fold"
        if r1 == 10:
            if r2 >= 6:    return "call"   # T6s+: call
            return "fold"
        if r1 == 9:
            if r2 >= 6:    return "call"   # 96s+: call
            return "fold"
        if r1 == 8 and r2 >= 5:  return "call"   # 85s+
        if r1 == 7 and r2 >= 4:  return "call"   # 74s+
        if r1 == 6 and r2 >= 4:  return "call"   # 64s+
        if r1 == 5 and r2 >= 3:  return "call"   # 53s+
        return "fold"

    # Offsuit
    if r1 == 14:
        if r2 >= 13:   return "raise"  # AKo: 3bet
        if r2 >= 7:    return "call"   # A7o-AQo: call
        return "fold"
    if r1 == 13:
        if r2 >= 9:    return "call"   # K9o+: call
        return "fold"
    if r1 == 12:
        if r2 >= 9:    return "call"   # Q9o+: call
        return "fold"
    if r1 == 11:
        if r2 >= 8:    return "call"   # J8o+: call
        return "fold"
    if r1 == 10:
        if r2 >= 7:    return "call"   # T7o+: call
        return "fold"
    if r1 == 9 and r2 >= 7:   return "call"   # 97o+
    if r1 == 8 and r2 >= 7:   return "call"   # 87o
    return "fold"


def _gto_sb_vs_3bet(r1, r2, suited):
    """
    GTO SB action vs BB 3-bet based on GGPoker 200NL 100bb HU chart.
    Target: Fold ~44% / Call ~37% / 4Bet ~19% (by combos).
    """
    is_pair = (r1 == r2)

    if is_pair:
        if r1 >= 13:   return "raise"  # KK+: 4bet jam
        if r1 >= 7:    return "call"   # 77-QQ: call
        return "fold"                  # 22-66: fold

    if suited:
        if r1 == 14:
            if r2 >= 13:   return "raise"  # AKs: 4bet
            if r2 >= 6:    return "call"   # A6s-AQs: call
            return "raise"                 # A2s-A5s: 4bet bluff (blocker)
        if r1 == 13:
            if r2 >= 10:   return "call"   # KTs+: call
            return "fold"
        if r1 == 12:
            if r2 >= 9:    return "call"   # Q9s+: call
            return "fold"
        if r1 == 11:
            if r2 >= 8:    return "call"   # J8s+: call
            return "fold"
        if r1 == 10 and r2 >= 8: return "call"   # T8s+
        if r1 == 9  and r2 >= 7: return "call"   # 97s+
        return "fold"

    # Offsuit
    if r1 == 14:
        if r2 >= 13:   return "raise"  # AKo: 4bet
        if r2 >= 9:    return "call"   # A9o-AQo: call
        return "fold"
    if r1 == 13:
        if r2 >= 11:   return "call"   # KJo+: call
        return "fold"
    if r1 == 12:
        if r2 >= 11:   return "call"   # QJo: call
        return "fold"
    return "fold"


def generate_gto_data(csv_path):
    """Build GTO reference DataFrame and write to CSV."""
    rows = []
    for r1 in range(14, 1, -1):
        for r2 in range(r1, 1, -1):
            suits = [False] if r1 == r2 else [True, False]
            for suited in suits:
                hand = hand_name(r1, r2, suited)
                for scenario, fn in [
                    ("SB_RFI",    _gto_sb_rfi),
                    ("BB_vs_RFI", _gto_bb_vs_rfi),
                    ("SB_vs_3bet",_gto_sb_vs_3bet),
                ]:
                    rows.append({
                        "hand": hand,
                        "rank1": r1,
                        "rank2": r2,
                        "suited": int(suited),
                        "scenario": scenario,
                        "gto_action": fn(r1, r2, suited),
                    })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"GTO data written to {csv_path} ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Scenario observation constructor
# ---------------------------------------------------------------------------

def make_scenario_obs(env, r1, r2, suited, scenario):
    """
    Construct the observation vector for a specific hand + scenario.
    Returns None if the scenario terminates unexpectedly.
    """
    cards = hand_to_cards(r1, r2, suited)
    env.reset()

    if scenario == "SB_RFI":
        env.hole_cards[0] = cards
        return env._get_obs()

    elif scenario == "BB_vs_RFI":
        env.hole_cards[1] = cards
        obs, _, done, _, _ = env.step(RAISE_75)  # SB opens
        if done:
            return None
        return obs  # now current_player == 1 (BB)

    elif scenario == "SB_vs_3bet":
        env.hole_cards[0] = cards
        _, _, done, _, _ = env.step(RAISE_75)   # SB opens
        if done:
            return None
        obs, _, done, _, _ = env.step(RAISE_75) # BB 3bets
        if done:
            return None
        return obs  # now current_player == 0 (SB)

    return None


# ---------------------------------------------------------------------------
# 1. Random baseline comparison
# ---------------------------------------------------------------------------

def evaluate_vs_random(agent, num_hands=10000):
    """
    Run PPO agent against a random-action opponent.
    Returns dict with reward statistics (bb per hand).
    """
    env = PokerPreflopEnv()
    ppo_rewards  = []
    rand_rewards = []

    for hand in range(num_hands):
        # --- PPO as player 0, random as player 1 ---
        obs, _ = env.reset()
        seat = hand % 2  # alternate seats to remove positional bias

        while True:
            current = env.current_player
            if current == seat:
                action, _, _ = agent.select_action(obs)
            else:
                action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            if done:
                if current == seat:
                    ppo_rewards.append(reward)
                else:
                    ppo_rewards.append(-reward)
                break

    # Baseline: random vs random
    for _ in range(num_hands // 2):
        obs, _ = env.reset()
        while True:
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            if done:
                rand_rewards.append(reward)
                break

    results = {
        "ppo_mean_reward":  np.mean(ppo_rewards),
        "rand_mean_reward": np.mean(rand_rewards),
        "ppo_win_rate":     np.mean(np.array(ppo_rewards) > 0),
        "rand_win_rate":    np.mean(np.array(rand_rewards) > 0),
        "ppo_rewards":      ppo_rewards,
        "rand_rewards":     rand_rewards,
    }
    return results


# ---------------------------------------------------------------------------
# 2. GTO accuracy evaluation
# ---------------------------------------------------------------------------

def evaluate_gto_accuracy(agent, gto_df):
    """
    For each hand × scenario in gto_df, get the agent's top action and compare.
    Returns (DataFrame with agent_action and correct columns, list of inference times in seconds).
    """
    env = PokerPreflopEnv()
    records = []
    inference_times = []

    for _, row in gto_df.iterrows():
        r1, r2, suited = int(row["rank1"]), int(row["rank2"]), bool(row["suited"])
        scenario = row["scenario"]

        obs = make_scenario_obs(env, r1, r2, suited, scenario)
        if obs is None:
            continue

        t0 = time.perf_counter()
        probs = agent.get_probs(obs)
        inference_times.append(time.perf_counter() - t0)
        fold_p  = float(probs[FOLD])
        call_p  = float(probs[CALL])
        raise_p = float(probs[RAISE_33]) + float(probs[RAISE_75]) + \
                  float(probs[RAISE_150]) + float(probs[ALL_IN])

        # Classify by whichever 3-category bucket has highest aggregate prob.
        # Using per-slot argmax would split raise probability across 4 slots,
        # making fold win even when total raise intent is the majority.
        cat_probs = {"fold": fold_p, "call": call_p, "raise": raise_p}
        agent_action_cat = max(cat_probs, key=cat_probs.get)
        agent_action_idx = int(np.argmax(probs))
        gto_action = row["gto_action"]

        records.append({
            "hand":         row["hand"],
            "rank1":        r1,
            "rank2":        r2,
            "suited":       int(suited),
            "scenario":     scenario,
            "gto_action":   gto_action,
            "agent_action": agent_action_cat,
            "agent_raw":    agent_action_idx,
            "correct":      agent_action_cat == gto_action,
            "fold_prob":    fold_p,
            "call_prob":    call_p,
            "raise_prob":   raise_p,
        })

    return pd.DataFrame(records), inference_times


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_training_curves(output_dir, rewards_path=None):
    path = rewards_path or os.path.join(output_dir, "episode_rewards.npy")
    if not os.path.exists(path):
        print("No episode_rewards.npy found — skipping training curve plot.")
        return

    rewards = np.load(path)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rewards, alpha=0.2, color="steelblue")

    window = min(500, len(rewards) // 10)
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smoothed,
                color="steelblue", label=f"Smoothed ({window})")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Training Reward Over Time")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (bb)")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(output_dir, "eval_training_curve.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Training curve saved to {out}")


def plot_baseline_comparison(baseline_results, output_dir):
    ppo_r  = baseline_results["ppo_rewards"]
    rand_r = baseline_results["rand_rewards"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Distribution of rewards
    ax = axes[0]
    ax.hist(ppo_r,  bins=40, alpha=0.6, color="steelblue", label="PPO vs Random")
    ax.hist(rand_r, bins=40, alpha=0.6, color="tomato",    label="Random vs Random")
    ax.axvline(np.mean(ppo_r),  color="steelblue", linestyle="--")
    ax.axvline(np.mean(rand_r), color="tomato",    linestyle="--")
    ax.set_title("Reward Distribution")
    ax.set_xlabel("Reward (bb)")
    ax.legend()

    # Summary bar chart
    ax = axes[1]
    labels = ["PPO vs Random", "Random vs Random"]
    means  = [np.mean(ppo_r), np.mean(rand_r)]
    colors = ["steelblue", "tomato"]
    bars = ax.bar(labels, means, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:+.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Mean Reward per Hand")
    ax.set_ylabel("Mean Reward (bb)")

    plt.tight_layout()
    out = os.path.join(output_dir, "eval_baseline_comparison.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Baseline comparison saved to {out}")


def plot_confusion_matrix(results_df, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Agent Action vs GTO Action (by Scenario)", fontsize=13)

    for ax, scenario in zip(axes, SCENARIOS):
        sub = results_df[results_df["scenario"] == scenario]
        if sub.empty:
            ax.set_visible(False)
            continue

        labels = GTO_ACTIONS
        y_true = sub["gto_action"].values
        y_pred = sub["agent_action"].values

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(SCENARIO_LABELS[scenario])
        ax.set_xlabel("Agent Action")
        ax.set_ylabel("GTO Action")

    plt.tight_layout()
    out = os.path.join(output_dir, "eval_confusion_matrix.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Confusion matrix saved to {out}")


def plot_action_distribution(results_df, output_dir):
    """Bar chart comparing agent vs GTO action frequencies per scenario."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Action Distribution: Agent vs GTO", fontsize=13)

    for ax, scenario in zip(axes, SCENARIOS):
        sub = results_df[results_df["scenario"] == scenario]
        if sub.empty:
            ax.set_visible(False)
            continue

        x = np.arange(len(GTO_ACTIONS))
        width = 0.35
        gto_counts   = [np.mean(sub["gto_action"]   == a) for a in GTO_ACTIONS]
        agent_counts = [np.mean(sub["agent_action"] == a) for a in GTO_ACTIONS]

        ax.bar(x - width/2, gto_counts,   width, label="GTO",   color="seagreen", alpha=0.8)
        ax.bar(x + width/2, agent_counts, width, label="Agent", color="steelblue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(GTO_ACTIONS)
        ax.set_title(SCENARIO_LABELS[scenario])
        ax.set_ylabel("Frequency")
        ax.set_ylim(0, 1)
        ax.legend()

    plt.tight_layout()
    out = os.path.join(output_dir, "eval_action_distribution.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Action distribution saved to {out}")


def plot_error_heatmap(results_df, output_dir):
    """
    13x13 heatmap of GTO accuracy by hand type.
    Rows = rank1 (high card), columns = rank2 (low card).
    Suited hands in upper triangle, pairs on diagonal, offsuit in lower.
    """
    ranks = list(range(14, 1, -1))  # A down to 2
    rank_labels = [RANK_NAMES[r] for r in ranks]

    for scenario in SCENARIOS:
        sub = results_df[results_df["scenario"] == scenario]
        if sub.empty:
            continue

        grid = np.full((13, 13), np.nan)

        for _, row in sub.iterrows():
            r1i = ranks.index(int(row["rank1"]))
            r2i = ranks.index(int(row["rank2"]))
            grid[r1i][r2i] = float(row["correct"])

        fig, ax = plt.subplots(figsize=(9, 8))
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "rg", ["tomato", "gold", "seagreen"]
        )
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(13))
        ax.set_yticks(range(13))
        ax.set_xticklabels(rank_labels)
        ax.set_yticklabels(rank_labels)
        ax.set_xlabel("Low Card Rank")
        ax.set_ylabel("High Card Rank")
        ax.set_title(f"GTO Accuracy by Hand Type — {SCENARIO_LABELS[scenario]}")
        plt.colorbar(im, ax=ax, label="Correct (1=yes, 0=no)")
        plt.tight_layout()
        out = os.path.join(output_dir, f"eval_error_heatmap_{scenario}.png")
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"  Error heatmap saved to {out}")


# ---------------------------------------------------------------------------
# Range charts
# ---------------------------------------------------------------------------

def plot_range_charts(results_df, output_dir):
    """
    13x13 preflop range chart for each scenario, GTO vs Agent side-by-side.
    Upper-right triangle = suited, diagonal = pairs, lower-left = offsuit.
    """
    ranks = list(range(14, 1, -1))  # [14..2], index 0=A, 12=2
    rank_labels = [RANK_NAMES[r] for r in ranks]

    action_to_int = {"fold": 0, "call": 1, "raise": 2}
    cmap = mcolors.ListedColormap(["tomato", "steelblue", "seagreen"])

    for scenario in SCENARIOS:
        sub = results_df[results_df["scenario"] == scenario]
        if sub.empty:
            continue

        gto_grid   = np.full((13, 13), np.nan)
        agent_grid = np.full((13, 13), np.nan)

        for _, row in sub.iterrows():
            r1i = ranks.index(int(row["rank1"]))
            r2i = ranks.index(int(row["rank2"]))
            suited = bool(int(row["suited"]))

            if r1i == r2i:          # pair → diagonal
                gi, gj = r1i, r2i
            elif suited:            # suited → upper-right (r1i < r2i)
                gi, gj = r1i, r2i
            else:                   # offsuit → lower-left (swap)
                gi, gj = r2i, r1i

            gto_grid[gi][gj]   = action_to_int[row["gto_action"]]
            agent_grid[gi][gj] = action_to_int[row["agent_action"]]

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f"Preflop Range Chart — {SCENARIO_LABELS[scenario]}", fontsize=14)

        for ax, grid, title in zip(axes, [gto_grid, agent_grid], ["GTO Reference", "PPO Agent"]):
            ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")
            ax.set_xticks(range(13))
            ax.set_yticks(range(13))
            ax.set_xticklabels(rank_labels, fontsize=8)
            ax.set_yticklabels(rank_labels, fontsize=8)
            ax.set_xlabel("Low Card Rank")
            ax.set_ylabel("High Card Rank")
            ax.set_title(title, fontsize=12)

            for gi in range(13):
                for gj in range(13):
                    if np.isnan(grid[gi][gj]):
                        continue
                    if gi == gj:
                        label = f"{rank_labels[gi]}{rank_labels[gj]}"
                    elif gi < gj:   # upper triangle = suited
                        label = f"{rank_labels[gi]}{rank_labels[gj]}s"
                    else:           # lower triangle = offsuit
                        label = f"{rank_labels[gj]}{rank_labels[gi]}o"
                    ax.text(gj, gi, label, ha="center", va="center",
                            fontsize=5, color="white", fontweight="bold")

        from matplotlib.patches import Patch
        legend_els = [Patch(color="tomato",    label="Fold"),
                      Patch(color="steelblue", label="Call"),
                      Patch(color="seagreen",  label="Raise")]
        axes[1].legend(handles=legend_els, loc="lower right", fontsize=9)

        plt.tight_layout()
        out = os.path.join(output_dir, f"range_chart_{scenario}.png")
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"  Range chart saved to {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default=os.path.join(OUTPUT_DIR, "agent.pth"))
    p.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    args = p.parse_args()

    model_path = args.model_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(DATA_DIR, "gto_ranges.csv")

    # Load or generate GTO data
    if os.path.exists(csv_path):
        gto_df = pd.read_csv(csv_path)
        print(f"Loaded GTO data from {csv_path} ({len(gto_df)} rows)")
    else:
        gto_df = generate_gto_data(csv_path)

    # Load agent
    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}.")
        print("Run train.py first, or evaluating with untrained weights for demo.")
    agent = PPOAgent()
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print("Using untrained agent (all evaluations will reflect random policy).")

    print("\n--- 1. Training Curves ---")
    model_rewards = os.path.join(os.path.dirname(model_path), "episode_rewards.npy")
    plot_training_curves(output_dir, rewards_path=model_rewards)

    print("\n--- 2. Random Baseline Comparison ---")
    baseline = evaluate_vs_random(agent, num_hands=10000)
    print(f"  PPO  mean reward : {baseline['ppo_mean_reward']:+.4f} bb/hand")
    print(f"  Rand mean reward : {baseline['rand_mean_reward']:+.4f} bb/hand")
    print(f"  PPO  win rate    : {baseline['ppo_win_rate']:.1%}")
    print(f"  Rand win rate    : {baseline['rand_win_rate']:.1%}")
    plot_baseline_comparison(baseline, output_dir)

    print("\n--- 3. GTO Accuracy ---")
    results_df, inference_times = evaluate_gto_accuracy(agent, gto_df)

    overall_acc = results_df["correct"].mean()
    print(f"  Overall accuracy : {overall_acc:.1%}")
    for scenario in SCENARIOS:
        sub = results_df[results_df["scenario"] == scenario]
        acc = sub["correct"].mean()
        print(f"  {SCENARIO_LABELS[scenario]:<30} {acc:.1%}")

    if inference_times:
        mean_ms = np.mean(inference_times) * 1000
        throughput = 1000 / mean_ms
        print(f"\n--- Inference Performance ---")
        print(f"  Hands evaluated  : {len(inference_times)}")
        print(f"  Mean latency     : {mean_ms:.3f} ms/hand")
        print(f"  Throughput       : {throughput:.0f} hands/sec")
        print(f"  p95 latency      : {np.percentile(inference_times, 95)*1000:.3f} ms")

    results_path = os.path.join(output_dir, "gto_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"  Detailed results saved to {results_path}")

    print("\n--- 4. Visualizations ---")
    plot_confusion_matrix(results_df, output_dir)
    plot_action_distribution(results_df, output_dir)
    plot_error_heatmap(results_df, output_dir)
    plot_range_charts(results_df, output_dir)

    print("\n--- 5. Error Analysis Summary ---")
    wrong = results_df[~results_df["correct"]]
    if len(wrong) > 0:
        top_errors = (
            wrong.groupby(["scenario", "hand", "gto_action", "agent_action"])
                 .size()
                 .reset_index(name="count")
                 .sort_values("count", ascending=False)
                 .head(10)
        )
        print("  Top mismatched hands:")
        print(top_errors.to_string(index=False))
    else:
        print("  No mismatches found.")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
