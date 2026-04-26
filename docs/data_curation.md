# GTO Benchmark Dataset — Curation Methodology

## Source

Preflop ranges sourced from **GGPoker 200NL (100bb, 6-max → heads-up) solver charts**, which represent near-optimal game-theoretic equilibrium play in a heads-up no-limit Texas Hold'em setting. These charts are publicly available from GGPoker's solver suite and are widely used as a professional reference benchmark.

## Coverage

The dataset covers all **169 canonical hand combinations** × **3 positional scenarios** = **507 labeled examples**.

Hand types:
- 13 pocket pairs (AA, KK, …, 22)
- 78 suited hands (e.g., AKs, T9s, 72s)
- 78 offsuit hands (e.g., AKo, T9o, 72o)

Scenarios:
| Scenario | Description | GTO Action Space |
|---|---|---|
| `SB_RFI` | Small blind opens (first to act, no prior raise) | fold / raise 2.5x |
| `BB_vs_RFI` | Big blind responds to SB 2.5x open | fold / call / 3-bet |
| `SB_vs_3bet` | Small blind responds to BB 3-bet | fold / call / 4-bet |

## Label Discretization

GTO solvers produce **mixed strategies** (e.g., "raise 63%, call 37%"). For evaluation, the dominant action (highest-probability bucket) is used as the ground-truth label:

```
dominant_action = argmax { fold_freq, call_freq, raise_freq }
```

This matches how the PPO agent is evaluated (3-category aggregate probability).

**Known limitation:** Mixed-strategy hands (e.g., KQo in BB vs RFI: raise 40% / call 60%) are assigned the call label, but the agent may actually be correct in raising them — the binary correct/incorrect metric underestimates agent quality on borderline hands.

## Data Format

File: `data/gto_ranges.csv`

| Column | Type | Description |
|---|---|---|
| `hand` | string | Canonical hand name (e.g., `AKs`, `72o`, `AA`) |
| `rank1` | int | High card rank (2–14, Ace=14) |
| `rank2` | int | Low card rank (2–14) |
| `suited` | int | 1 if suited, 0 if not (always 0 for pairs) |
| `scenario` | string | One of `SB_RFI`, `BB_vs_RFI`, `SB_vs_3bet` |
| `gto_action` | string | Dominant GTO action: `fold`, `call`, or `raise` |

## Preprocessing

No preprocessing was required — the ranges were encoded directly as categorical labels. The only non-trivial step was the **discretization of mixed strategies** to a single dominant action, as described above.

The observation encoding (feature engineering) is handled in `src/poker_env.py` — see `docs/feature_engineering.md` for details.

## Class Balance

| Scenario | Fold | Call | Raise |
|---|---|---|---|
| SB Opening | ~22% | 0% | ~78% |
| BB vs Open | ~36% | ~49% | ~15% |
| SB vs 3-Bet | ~44% | ~37% | ~19% |

The SB Opening scenario is highly imbalanced toward raise (78%), reflecting the GTO principle that limping is never correct heads-up. This imbalance motivated the `sb_open_call_penalty` reward shaping parameter.
