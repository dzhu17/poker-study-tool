# Poker Preflop GTO Study Tool

A heads-up No-Limit Texas Hold'em preflop study tool built on a custom PPO reinforcement learning agent trained entirely via self-play. The agent learns preflop strategy from scratch — with no hard-coded poker rules — and is evaluated against professional GTO solver benchmarks across three distinct positional scenarios.

---

## What it Does

This project builds a complete reinforcement learning pipeline for heads-up NLHE preflop poker. A custom OpenAI Gymnasium environment (`poker_env.py`) models the full preflop betting tree — six discrete actions (fold, call, raise 33%/75%/150% pot, all-in), position-aware state, and an equity-based showdown reward using a hand strength heuristic. A Proximal Policy Optimization agent with an actor-critic neural network architecture learns strategy through self-play: two copies of the same agent play against each other, with no external poker knowledge provided. Training uses position-specific reward shaping to guide the agent toward GTO-like behavior — separate bonuses and penalties for SB opening, BB 3-betting, and SB responding to 3-bets — motivated by the insight that each street has fundamentally different GTO frequencies. After training, the agent is evaluated against GGPoker 200NL heads-up preflop charts across 169 hand combinations × 3 scenarios. The best model is served through a Flask web app where users can study per-hand probability breakdowns in Study mode or test their own GTO intuition in Quiz mode.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/dzhu17/poker-study-tool.git
cd poker-study-tool

# 2. Create and activate virtual environment (requires Python 3.11)
python3.11 -m venv venv      # Mac/Linux
py -3.11 -m venv venv        # Windows (if multiple Python versions installed)
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the web app (uses pre-computed results — no GPU needed)
python app.py
# Open http://localhost:5000
```

To retrain from scratch:
```bash
python src/train.py \
  --episodes 500000 \
  --output_dir models/my_run \
  --entropy_coef 0.10 \
  --sb_open_raise_bonus 0.25 \
  --sb_open_call_penalty 1.00 \
  --bb_rfi_raise_bonus 0.06 \
  --bb_rfi_call_penalty 0.40 \
  --sv3_raise_bonus 0.25 \
  --sv3_call_penalty 0.45
```

To evaluate a trained model against GTO benchmarks:
```bash
python src/evaluate.py \
  --model_path models/my_run/agent.pth \
  --output_dir models/my_run
```

To run the full multi-config sweep + round-robin tournament:
```bash
python run_training.py
```

---

## Technical Implementation

### Neural Network Architecture

The agent uses a custom **actor-critic network** (`src/ppo_agent.py`) with a shared trunk and separate policy and value heads:

```
Input (116-dim observation)
    │
    ▼
Shared Trunk: Linear(116→256) → ReLU → Dropout(0.1)
             → Linear(256→128) → ReLU → Dropout(0.1)
    │
    ├──▶ Actor Head:  Linear(128→64) → ReLU → Linear(64→6)  → Categorical policy
    └──▶ Critic Head: Linear(128→64) → ReLU → Linear(64→1)  → state value V(s)
```

All weights initialized with **orthogonal initialization** (gain=√2 for trunk, gain=0.01 for actor output to encourage exploration early, gain=1.0 for critic output).

### PPO Algorithm

Custom implementation of **Proximal Policy Optimization** (`src/ppo_agent.py`) with:

| Component | Implementation | Value |
|---|---|---|
| Advantage estimation | Generalized Advantage Estimation (GAE-λ) | λ = 0.95 |
| Clipped surrogate objective | PPO-clip | ε = 0.2 |
| Discount factor | γ | 0.99 |
| Entropy bonus | Exploration regularization | coef = 0.10 (best model) |
| Value function loss | MSE | coef = 0.5 |
| Update epochs | Minibatch iterations per rollout | 4 |
| Minibatch size | Transitions sampled per update | 64 |
| Rollout buffer | Transitions collected before update | 512 |
| Gradient clipping | Prevents exploding gradients | max norm = 0.5 |
| LR schedule | Cosine annealing | 3e-4 → 1e-6 |
| Optimizer | Adam | lr = 3e-4 |

**Regularization techniques used:** dropout (p=0.1 in shared trunk), early stopping (patience=20 evaluation windows of 500 episodes each, min delta=0.005), gradient clipping, and entropy bonus.

### Self-Play Training

The agent trains through **self-play** (`src/train.py`): two copies of the same agent play against each other with no external poker knowledge provided.

- A **frozen opponent** (copy of the current agent weights) controls the opposing seat
- Opponent weights sync to latest agent every 500 episodes, preventing the agent from overfitting to a fixed opponent
- The learning agent **alternates seats** each episode (SB/BB) to ensure it learns both positions
- Transitions from the opponent's turns are discarded; only the learning agent's trajectory is used for updates

### Custom Gymnasium Environment

`src/poker_env.py` implements a full heads-up NLHE preflop betting tree as a custom Gymnasium environment:

**Observation space (116 dimensions):**
```
obs[0:52]    — Hole card 1 (52-dim one-hot over all 52 cards)
obs[52:104]  — Hole card 2 (52-dim one-hot)
obs[104]     — Position (0 = SB/Button, 1 = BB)
obs[105]     — Pot size / starting_stack
obs[106]     — My stack / starting_stack
obs[107]     — Opponent stack / starting_stack
obs[108]     — Amount to call / starting_stack
obs[109:115] — Last action (6-dim one-hot over action space)
obs[115]     — Number of raises so far / 4.0
```

**Action space (6 discrete actions):** fold, call, raise 33% pot, raise 75% pot, raise 150% pot, all-in.

**Reward function:** Terminal reward is equity-based expected value:

```
EV = equity(hand) × pot − committed_chips
```

where `equity` is computed from a hand-strength heuristic calibrated to HU preflop equity:

```
strength = rank1 + rank2
         + 18 (if pocket pair)
         + 2  (if suited)
         + connectivity bonus (0-4 based on card gap)
equity ≈ strength / 46.0
```

**Position-specific reward shaping (6 parameters)** guides the agent toward GTO-like frequencies without providing explicit hand rankings:

| Parameter | Best Model Value | GTO Motivation |
|---|---|---|
| `sb_open_raise_bonus` | 0.25 | GTO: SB raises ~78% of hands HU |
| `sb_open_call_penalty` | 1.00 | GTO: limping is never correct HU |
| `pair_raise_bonus` | 0.40 | All pocket pairs should open-raise |
| `bb_rfi_raise_bonus` | 0.06 | GTO: BB 3-bets only ~5-6% of hands |
| `bb_rfi_call_penalty` | 0.40 | Overcomes sunk blind bias toward calling |
| `sv3_call_penalty` | 0.45 | GTO: SB mostly folds or 4-bets vs 3-bet |

### Hyperparameter Search and Model Selection

Trained **20 configurations** across 500,000 episodes each, systematically varying all 6 reward shaping parameters to find the best GTO approximation. Model selection used two criteria:

1. **GTO accuracy** vs GGPoker solver benchmarks (fold/call/raise accuracy over 507 hand/scenario combinations)
2. **Round-robin tournament** — head-to-head play between all 20 trained models (10,000 hands per matchup) to identify the strongest agent independent of GTO accuracy

Key finding from the ablation: `sb_open_call_penalty ≥ 1.0` was necessary to suppress limping, and `bb_rfi_raise_bonus ≤ 0.06` was critical to prevent over-3-betting. The most GTO-accurate model (43.8%) was *not* the tournament winner — a concrete demonstration of the GTO vs. exploitative strategy tradeoff.

### Production Deployment

The Flask web app (`app.py`) uses pre-computed results from the best trained model (sub-millisecond response, no live inference at request time). Production considerations implemented:

1. **Caching:** All 507 hand/scenario predictions pre-computed and loaded into memory at startup from `gto_results.csv` — eliminates per-request model inference entirely
2. **Structured logging:** Timestamped request logs (`INFO`) and error logs (`ERROR`) via Python `logging` module
3. **HTTP error handlers:** JSON-formatted 404 (not found) and 500 (server error) responses
4. **Input validation:** Rank bounds (2–14) and scenario whitelist checked before any lookup, returning 400 with descriptive error messages

---

## Video Links

- **Project Demo :** [_\[add link after recording\]_]https://drive.google.com/file/d/1W9u0XFirChTEp4XjTjgs7gzk1qE9Vowu/view?usp=drive_link
- **Technical Walkthrough :** https://drive.google.com/file/d/1N0STp5ouXQHF_KM0tqWcvHqzIo1xMSNh/view?usp=drive_link
---

## Evaluation

### GTO Accuracy — Best Model (`BEST_GTO__optA_46pct_accuracy`)

The best model was selected from **20 training configurations** based on GTO accuracy against GGPoker 200NL heads-up preflop charts (507 hand/scenario combinations):

| Scenario | GTO Accuracy |
|---|---|
| SB Opening | 84.0% |
| BB vs SB Open | 28.4% |
| SB vs BB 3-Bet | 18.9% |
| **Overall** | **43.8%** |

A random baseline achieves ~33% (uniform over fold/call/raise), so the agent meaningfully outperforms chance, especially on SB opening where it reaches 84%.

### Key Quantitative Findings

**Premium hands learned correctly:**
| Hand | Raise Prob | Agent Action | Correct? |
|---|---|---|---|
| AA (SB open) | 99.99% | raise | ✅ |
| KK (SB open) | 99.99% | raise | ✅ |
| 72o (SB open) | 2.1% raise | fold | ✅ |

**Hardest scenario — BB vs SB Open:**
The BB calling range is the most nuanced: GTO calls ~34%, 3-bets ~5-6%, and folds ~60% with strong hand-rank dependence. The agent tends to over-fold marginal hands (53s, 64s) and under-3-bet borderline hands (99, TT), which accounts for most of the BB accuracy gap.

**GTO vs. Exploitative tradeoff (tournament finding):**
A round-robin tournament between all 20 trained configs revealed that the most GTO-accurate model (43.8% accuracy) is *not* the strongest head-to-head player. A less GTO-accurate model (config11, 38% accuracy) won the tournament by exploiting the tendencies of other agents — a concrete demonstration of the GTO vs. exploitative strategy tradeoff in game theory.

**Reward shaping ablation:**
Systematically varying 6 reward shaping parameters across 20 configs showed that `sb_open_call_penalty ≥ 1.0` was necessary to suppress SB limping (GTO: never limp HU), and `bb_rfi_raise_bonus ≤ 0.06` was critical to prevent BB over-3-betting.

### Evaluation Artifacts

All outputs saved in `models/BEST_GTO__optA_46pct_accuracy/`:
- `range_chart_SB_RFI.png` — 13×13 preflop range grid, SB opening
- `range_chart_BB_vs_RFI.png` — 13×13 preflop range grid, BB facing open
- `range_chart_SB_vs_3bet.png` — 13×13 preflop range grid, SB facing 3-bet
- `eval_confusion_matrix.png` — fold/call/raise confusion matrix vs GTO
- `eval_error_heatmap_*.png` — per-hand error heatmaps for each scenario
- `eval_action_distribution.png` — agent vs GTO action frequency comparison
- `training_curves.png` — reward over 500,000 training episodes
- `gto_results.csv` — full per-hand predictions with probabilities

---

## Project Structure

```
poker-study-tool/
├── app.py                  # Flask web server (study + quiz modes)
├── run_training.py         # Multi-config sweep + round-robin tournament
├── requirements.txt
├── src/
│   ├── poker_env.py        # Custom Gymnasium environment (6-action preflop)
│   ├── ppo_agent.py        # PPO actor-critic agent (custom implementation)
│   ├── train.py            # Self-play training loop with reward shaping
│   └── evaluate.py         # GTO accuracy eval, range charts, heatmaps
├── models/
│   └── BEST_GTO__optA_46pct_accuracy/   # Best model + all eval outputs
├── data/
│   └── gto_ranges.csv      # GTO benchmark (GGPoker 200NL HU preflop)
├── static/                 # Frontend JS + CSS
├── templates/              # Flask HTML templates
├── videos/                 # Demo and walkthrough videos
└── docs/                   # Additional documentation
```

---

## Individual Contributions

Solo project — all environment design, agent implementation, training infrastructure, evaluation pipeline, reward shaping research, and web app built by Daniel Zhu.
