# Attribution

## AI Development Tools

This project was built with assistance from Claude Code (Anthropic) as an AI pair-programming tool. Below is a substantive account of how AI assistance was used, what was generated vs. modified, and what required significant human debugging and rework.

### What was AI-generated (scaffolding / boilerplate)
- Initial Flask route structure (`app.py`): basic `@app.route` patterns and `jsonify` response formatting
- HTML/CSS layout for the poker table UI (`templates/index.html`, `static/style.css`): the oval felt design, card positioning, and color scheme
- Gymnasium environment boilerplate: the `reset()` / `step()` / `render()` method signatures and `spaces.Box` / `spaces.Discrete` setup

### What was substantially directed and designed by the student
- **All core algorithmic decisions**: the 6-action discrete action space (fold, call, raise 33%/75%/150%, all-in), the 116-dimensional observation encoding (104 one-hot card features + 6 scalar game state features + 6 last-action OHE), and the equity-based showdown reward
- **The self-play training loop**: the decision to train two agents against each other by alternating seats, how episode transitions are stored, and PPO update logic
- **Position-specific reward shaping**: the design of separate bonuses/penalties for SB opening, BB 3-betting, and SB vs 3-bet — motivated by studying GTO preflop frequencies and recognizing that a flat call_penalty couldn't distinguish between three fundamentally different poker situations
- **The round-robin tournament**: the design of head-to-head matchup evaluation as a model selection criterion beyond training reward
- **GTO evaluation methodology**: the decision to use 3-category aggregate probability (fold/call/raise) instead of raw 6-slot argmax, which was identified and fixed after discovering it caused the model to appear to fold pocket pairs that it was actually raising 55%+ of the time

### What was generated but substantially debugged and reworked
- **Reward accumulation bug**: AI-generated training code overwrote intermediate raise rewards with terminal rewards. Identified that `transitions[-1] = (o, a, lp, v, p_reward, True)` should instead accumulate: `prev_r + p_reward`. This bug caused all reward shaping for non-terminal raises to be silently discarded.
- **Hand strength heuristic**: The original `_hand_strength()` formula gave pocket pairs a `+10` bonus, causing 22 to estimate only 41% equity vs ATo (real: ~53%). Redesigned the formula with `+18` pair bonus and updated normalization to correctly reflect HU equity for small pairs.
- **Evaluation classification bug**: The original `argmax(probs)` over 6 actions caused fold to win when raise probability was split across 4 raise-size slots. Reworked to use category-level argmax over `{fold_prob, call_prob, raise_prob}`.
- **`sb_open_call_penalty` sign issue**: Negative CLI args caused argparse parsing errors. Reworked the interface to pass penalty magnitudes as positive floats and negate internally.

### AI-generated code comments in source files
Per course policy, functions with significant AI scaffolding are noted in code comments at the function level in `src/poker_env.py`, `src/ppo_agent.py`, and `app.py`.

---

## External Libraries

| Library | Version | Use |
|---|---|---|
| [PyTorch](https://pytorch.org/) | 2.x | Actor-critic neural network, PPO gradient updates |
| [Gymnasium](https://gymnasium.farama.org/) | 1.3.0 | Custom RL environment interface |
| [Flask](https://flask.palletsprojects.com/) | 3.1.x | Web application server |
| [pandas](https://pandas.pydata.org/) | 2.x | CSV data handling and evaluation |
| [NumPy](https://numpy.org/) | 1.x | Observation encoding, numerical operations |
| [matplotlib](https://matplotlib.org/) | 3.x | Training curves, 13×13 range charts, error heatmaps |
| [scikit-learn](https://scikit-learn.org/) | 1.x | Confusion matrix computation |

---

## Datasets

**GTO Benchmark (`data/gto_ranges.csv`):** Preflop ranges sourced from GGPoker 200NL heads-up preflop solver charts, used under fair use for academic evaluation. Covers all 169 hand combinations across three positional scenarios (SB open, BB vs open, SB vs 3-bet). This dataset serves as the ground truth for all GTO accuracy metrics reported in the project.

---

## References

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). OpenAI.
- Farama Foundation. [Gymnasium Documentation](https://gymnasium.farama.org/). Custom environment API.
- Chen, B. & Ankenman, J. (2006). *The Mathematics of Poker.* ConJelCo. — Chen formula referenced as inspiration for the preflop hand strength heuristic.
- Silver, D. et al. (2016). [Mastering the Game of Go with Deep Neural Networks and Tree Search](https://www.nature.com/articles/nature16961). Nature. — Conceptual reference for self-play as a training paradigm.
