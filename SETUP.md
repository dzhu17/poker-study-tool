# Setup Instructions

## System Requirements

- Python 3.11 specifically (3.12+ not compatible with pinned torch/numpy versions)
- Windows, macOS, or Linux
- No GPU required — all training runs on CPU (AMD/Intel compatible)
- ~2 GB disk space for models and eval outputs

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd poker-study-tool

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
venv\Scripts\activate        # Windows (Command Prompt / PowerShell)
source venv/bin/activate     # macOS / Linux

# 4. Install all dependencies
pip install -r requirements.txt
```

---

## Running the Web App (Recommended for Graders)

The web app uses pre-computed results from the best trained model — no training or GPU needed.

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

**Study mode:** Select a scenario tab (SB Opening / BB vs Open / SB vs 3-Bet). Click either card to pick its rank, then suit. The agent's recommended action and probability breakdown appear instantly.

**Quiz mode:** The app deals a random hand from the selected scenario. Choose FOLD, LIMP/CALL, or RAISE/3-BET/4-BET — the agent's answer is revealed after you click.

---

## Retraining from Scratch

The best model uses the following reward shaping parameters (trains in ~15 min on CPU):

```bash
python src/train.py \
  --episodes 500000 \
  --output_dir models/my_run \
  --entropy_coef 0.10 \
  --lr 3e-4 \
  --sb_open_raise_bonus 0.25 \
  --sb_open_call_penalty 1.00 \
  --pair_raise_bonus 0.40 \
  --bb_rfi_raise_bonus 0.06 \
  --bb_rfi_call_penalty 0.40 \
  --sv3_raise_bonus 0.25 \
  --sv3_call_penalty 0.45
```

Progress logs print every 500 episodes. Checkpoints saved every 5,000 episodes.

All reward shaping parameters and their GTO motivation:

| Parameter | Value | Why |
|---|---|---|
| `sb_open_raise_bonus` | 0.25 | GTO: SB raises ~78% of hands HU |
| `sb_open_call_penalty` | 1.00 | GTO: limping is never correct HU |
| `pair_raise_bonus` | 0.40 | All pocket pairs should open-raise |
| `bb_rfi_raise_bonus` | 0.06 | GTO: BB 3-bets only ~5-6% |
| `bb_rfi_call_penalty` | 0.40 | Overcomes sunk blind bias toward calling |
| `sv3_raise_bonus` | 0.25 | GTO: SB 4-bets only premiums |
| `sv3_call_penalty` | 0.45 | GTO: SB mostly folds to 3-bet |

---

## Evaluating a Trained Model

```bash
python src/evaluate.py \
  --model_path models/my_run/agent.pth \
  --output_dir models/my_run
```

Outputs generated:
- `gto_results.csv` — per-hand predictions with fold/call/raise probabilities
- `range_chart_SB_RFI.png` — 13×13 preflop range grid (SB opening)
- `range_chart_BB_vs_RFI.png` — 13×13 preflop range grid (BB vs open)
- `range_chart_SB_vs_3bet.png` — 13×13 preflop range grid (SB vs 3-bet)
- `eval_confusion_matrix.png` — fold/call/raise confusion vs GTO
- `eval_error_heatmap_*.png` — per-hand error heatmaps per scenario
- `eval_action_distribution.png` — agent vs GTO action frequency
- `eval_training_curve.png` — reward over training
- `eval_baseline_comparison.png` — agent vs random baseline

---

## Running the Full Multi-Config Sweep + Tournament

```bash
python run_training.py
```

This trains multiple reward-shaping configurations sequentially, then runs a round-robin head-to-head tournament to determine the strongest model. Results printed to console; best model copied to `models/agent.pth`.

---

## Project Structure

```
poker-study-tool/
├── app.py                  # Flask web server
├── run_training.py         # Multi-config training + round-robin tournament
├── requirements.txt        # Python dependencies
├── README.md
├── SETUP.md
├── ATTRIBUTION.md
├── src/
│   ├── poker_env.py        # Custom Gymnasium environment (6-action preflop NLHE)
│   ├── ppo_agent.py        # PPO actor-critic agent (custom PyTorch implementation)
│   ├── train.py            # Self-play training loop with reward shaping
│   ├── evaluate.py         # GTO accuracy evaluation, range charts, heatmaps
│   └── test_env.py         # Environment unit tests
├── models/
│   ├── BEST_GTO__optA_46pct_accuracy/   # Best model + full eval outputs
│   ├── PAIRFIX__config20_42pct_accuracy/ # Pair-fix variant model
│   ├── TOURNAMENT_WINNER__moderate_config11/ # Best head-to-head model
│   └── logs/               # Training logs for all configs
├── data/
│   └── gto_ranges.csv      # GTO benchmark (GGPoker 200NL HU preflop)
├── static/
│   ├── app.js              # Frontend logic (study/quiz modes, card picker)
│   └── style.css           # Poker table UI styles
├── templates/
│   └── index.html          # Main app template
├── videos/                 # Demo and walkthrough videos
└── docs/                   # Additional documentation
```
