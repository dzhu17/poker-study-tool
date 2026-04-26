# Feature Engineering — Observation Space Design

## Overview

The agent's input at each decision point is a **116-dimensional float32 vector** constructed from the current game state. All features are normalized to roughly [0, 1] to stabilize training.

## Observation Vector Structure

```
obs[0:104]    — Card encoding   (104 dims)
obs[104]      — Position        (1 dim)
obs[105]      — Pot size        (1 dim, normalized)
obs[106]      — My stack        (1 dim, normalized)
obs[107]      — Opponent stack  (1 dim, normalized)
obs[108]      — Amount to call  (1 dim, normalized)
obs[109:115]  — Last action OHE (6 dims)
obs[115]      — Num raises      (1 dim, normalized)
              Total: 116 dims
```

## Card Encoding (104 dims)

Each hole card is represented as a **52-dimensional one-hot vector** (one entry per card in a standard deck). Two cards → 104 dimensions.

```python
# Card index: rank (2–14) × 4 suits
card_idx = (rank - 2) * 4 + suit   # 0–51

obs[0:52]   = encode_card(hole_cards[0])   # first hole card
obs[52:104] = encode_card(hole_cards[1])   # second hole card
```

**Why one-hot over a compact representation:** A one-hot encoding lets the network learn separate weights for every card, allowing it to discover non-linear hand strength patterns (e.g., that A♠ and A♥ have the same strategic value despite different indices). A compact rank/suit encoding would require the network to learn this invariance implicitly.

**Note:** The agent sees its own cards only (private information), not the opponent's.

## Position (1 dim)

```python
obs[104] = float(current_player)   # 0 = SB/Button, 1 = BB
```

Position is the most strategically important scalar feature — preflop GTO play differs radically between SB (acts first) and BB.

## Pot and Stack Sizes (3 dims, normalized)

```python
obs[105] = pot_size   / starting_stack    # typically 0.015–2.0
obs[106] = my_stack   / starting_stack    # 0.0–1.0
obs[107] = opp_stack  / starting_stack    # 0.0–1.0
```

Normalizing by starting stack (100bb) keeps all scalar features in similar magnitude to the one-hot card features.

## Amount to Call (1 dim)

```python
obs[108] = amount_to_call / starting_stack   # 0.0–1.0
```

Encodes pot odds directly. When facing a large 3-bet, `amount_to_call` is large, training the agent to be more selective about calling/4-betting.

## Last Action One-Hot (6 dims)

```python
obs[109:115] = one_hot(last_action, num_classes=6)
# Actions: 0=fold, 1=call, 2=raise33, 3=raise75, 4=raise150, 5=all-in
```

The opponent's last action is the most direct signal for scenario detection. If `obs[109]=1` (opponent raised), the agent is facing a raise scenario. Without this feature, the agent cannot distinguish "BB's first decision" from "SB facing a 3-bet."

## Number of Raises (1 dim)

```python
obs[115] = num_raises / 4.0   # 4 = practical maximum (cap)
```

Distinguishes street level: 0 raises = first decision, 1 raise = facing open, 2 raises = facing 3-bet, 3+ = deep re-raise territory. This feature prevents the agent from treating "SB opening" and "SB vs 3-bet" as equivalent situations.

## Design Decisions

| Decision | Rationale |
|---|---|
| 52-dim one-hot per card vs compact rank/suit encoding | Gives network maximum flexibility to learn non-linear hand strength without inductive bias |
| Private cards only (no opponent cards) | Matches real game information structure |
| Normalization by starting stack | Keeps scalars in the same order of magnitude as one-hot features |
| Last-action OHE vs scalar action index | Ordinal encoding would imply raise150 > call by 4 units; OHE avoids this |
| 116 total dims | Empirically sufficient for 169 distinct hands across 3 scenarios |
