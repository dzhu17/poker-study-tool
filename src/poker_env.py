import gymnasium as gym
import numpy as np
from gymnasium import spaces


RANKS = list(range(2, 15))  # 2–14 (14 = Ace)
SUITS = [0, 1, 2, 3]        # clubs, diamonds, hearts, spades
NUM_CARDS = 52

# Action indices
FOLD    = 0
CALL    = 1
RAISE_33 = 2
RAISE_75 = 3
RAISE_150 = 4
ALL_IN  = 5
NUM_ACTIONS = 6

# Raise size multipliers relative to pot
RAISE_FRACS = {RAISE_33: 0.33, RAISE_75: 0.75, RAISE_150: 1.50}


def card_to_idx(rank, suit):
    return (rank - 2) * 4 + suit


def encode_card(idx):
    """One-hot encode a card index into a 52-dim vector."""
    v = np.zeros(NUM_CARDS, dtype=np.float32)
    v[idx] = 1.0
    return v


class PokerPreflopEnv(gym.Env):
    """
    Heads-up No-Limit Texas Hold'em preflop environment.

    Players: 0 = SB/Button (acts first preflop), 1 = BB.
    Stacks: 100bb each. Blinds: SB posts 0.5bb, BB posts 1bb.
    Episode ends when a player folds, both players call (no raise pending),
    or an all-in is reached.

    Observation (from perspective of the acting player):
        [104 card features | position | pot | my_stack | opp_stack |
         amount_to_call | last_action_ohe(6) | num_raises]
    Total: 104 + 1 + 1 + 1 + 1 + 1 + 6 + 1 = 116 dims
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, big_blind=1.0, starting_stack=100.0,
                 # SB open (first action, no raises yet)
                 sb_open_raise_bonus=0.0,   # GTO: raise 78% of hands
                 sb_open_call_penalty=0.0,  # GTO: limping is NEVER correct HU
                 pair_raise_bonus=0.0,      # extra bonus for opening any pocket pair from SB
                 # BB vs SB open (facing 1 raise)
                 bb_rfi_raise_bonus=0.0,    # GTO: 3bet ~5-6% of hands
                 bb_rfi_call_penalty=0.0,   # mild — calling ~34% is GTO
                 # SB vs BB 3bet (facing 2 raises)
                 sv3_raise_bonus=0.0,       # GTO: 4bet premiums
                 sv3_call_penalty=0.0,      # GTO: mostly fold, some calls
                 # Generic fallback for deeper streets
                 raise_bonus=0.0,
                 call_penalty=0.0):
        super().__init__()
        self.big_blind = big_blind
        self.small_blind = big_blind / 2
        self.starting_stack = starting_stack
        self.sb_open_raise_bonus  = sb_open_raise_bonus
        self.sb_open_call_penalty = sb_open_call_penalty
        self.pair_raise_bonus     = pair_raise_bonus
        self.bb_rfi_raise_bonus   = bb_rfi_raise_bonus
        self.bb_rfi_call_penalty  = bb_rfi_call_penalty
        self.sv3_raise_bonus      = sv3_raise_bonus
        self.sv3_call_penalty     = sv3_call_penalty
        self.raise_bonus          = raise_bonus
        self.call_penalty         = call_penalty

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(116,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._deck = [card_to_idx(r, s) for r in RANKS for s in SUITS]
        self.reset()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        # Deal hole cards
        deck = self._deck.copy()
        rng.shuffle(deck)
        self.hole_cards = [deck[0:2], deck[2:4]]  # player 0, player 1

        # Stacks and blinds
        self.stacks = [self.starting_stack, self.starting_stack]
        self.stacks[0] -= self.small_blind
        self.stacks[1] -= self.big_blind
        self.pot = self.small_blind + self.big_blind
        self.committed = [self.small_blind, self.big_blind]

        # Betting state
        self.current_player = 0      # SB acts first preflop
        self.last_action = -1        # none yet
        self.num_raises = 0
        self.done = False
        self.winner = None           # 0 or 1 or None (split)

        # SB has already acted (posted), so BB is last aggressor
        # Amount to call for current player
        self._last_raise_to = self.big_blind  # BB raise level

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        assert not self.done, "Episode is over. Call reset()."
        action = int(action)

        if not self._is_legal(action):
            action = FOLD  # treat illegal actions as fold

        reward = 0.0
        terminated = False

        p = self.current_player
        opp = 1 - p

        amount_to_call = self.committed[opp] - self.committed[p]
        amount_to_call = max(0.0, min(amount_to_call, self.stacks[p]))

        if action == FOLD:
            # Opponent wins the pot
            reward = -self.committed[p]  # lose what was put in
            self.winner = opp
            terminated = True

        elif action == CALL:
            call_amt = min(amount_to_call, self.stacks[p])
            self.stacks[p] -= call_amt
            self.committed[p] += call_amt
            self.pot += call_amt
            self.last_action = CALL

            # Episode ends: both players have matched commitments
            terminated = True
            reward = self._showdown_reward(p)
            if call_amt > 0:
                # Scenario-specific call penalty
                if self.num_raises == 0 and p == 0:
                    reward -= self.sb_open_call_penalty   # SB limping
                elif self.num_raises == 1 and p == 1:
                    reward -= self.bb_rfi_call_penalty    # BB calling SB open
                elif self.num_raises == 2 and p == 0:
                    reward -= self.sv3_call_penalty       # SB calling BB 3bet
                else:
                    reward -= self.call_penalty

        elif action == ALL_IN:
            all_in_amt = self.stacks[p]
            self.stacks[p] = 0.0
            self.committed[p] += all_in_amt
            self.pot += all_in_amt
            self.num_raises += 1
            self.last_action = ALL_IN
            self._last_raise_to = self.committed[p]

            # If opponent can't re-raise (also all-in or pot is settled), end
            if self.stacks[opp] == 0:
                terminated = True
                reward = self._showdown_reward(p)
            else:
                self.current_player = opp

        else:
            # Raise actions (33%, 75%, 150% pot)
            frac = RAISE_FRACS[action]
            raise_to = self.committed[opp] + frac * self.pot
            raise_to = min(raise_to, self.stacks[p] + self.committed[p])
            additional = raise_to - self.committed[p]
            additional = min(additional, self.stacks[p])

            self.stacks[p] -= additional
            self.committed[p] += additional
            self.pot += additional
            self.num_raises += 1
            self.last_action = action
            self._last_raise_to = self.committed[p]
            self.current_player = opp
            # Scenario-specific raise bonus (num_raises already incremented)
            if self.num_raises == 1 and p == 0:
                reward = self.sb_open_raise_bonus    # SB opening
                r1_p = self.hole_cards[p][0] // 4 + 2
                r2_p = self.hole_cards[p][1] // 4 + 2
                if r1_p == r2_p:
                    reward += self.pair_raise_bonus  # pocket pair: always open
            elif self.num_raises == 2 and p == 1:
                reward = self.bb_rfi_raise_bonus     # BB 3betting
            elif self.num_raises == 3 and p == 0:
                reward = self.sv3_raise_bonus        # SB 4betting
            else:
                reward = self.raise_bonus

        if not terminated:
            pass  # current_player already swapped above for raise/all-in

        self.done = terminated
        obs = self._get_obs()
        return obs, float(reward), terminated, False, {}

    def render(self):
        p = self.current_player
        cards0 = self._card_str(self.hole_cards[0])
        cards1 = self._card_str(self.hole_cards[1])
        print(
            f"Player {p}'s turn | "
            f"P0: {cards0} stack={self.stacks[0]:.1f} | "
            f"P1: {cards1} stack={self.stacks[1]:.1f} | "
            f"Pot: {self.pot:.1f} | Committed: {self.committed}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self):
        p = self.current_player
        opp = 1 - p

        # Encode both hole cards as one-hot (52 dims each)
        c1 = encode_card(self.hole_cards[p][0])
        c2 = encode_card(self.hole_cards[p][1])

        amount_to_call = max(0.0, self.committed[opp] - self.committed[p])
        amount_to_call = min(amount_to_call, self.stacks[p])

        last_action_ohe = np.zeros(NUM_ACTIONS, dtype=np.float32)
        if self.last_action >= 0:
            last_action_ohe[self.last_action] = 1.0

        scalar_features = np.array([
            float(p),                              # position (0=SB, 1=BB)
            self.pot / self.starting_stack,        # normalized pot
            self.stacks[p] / self.starting_stack,  # normalized my stack
            self.stacks[opp] / self.starting_stack, # normalized opp stack
            amount_to_call / self.starting_stack,  # normalized call amount
            float(self.num_raises),
        ], dtype=np.float32)

        obs = np.concatenate([c1, c2, scalar_features, last_action_ohe])
        return obs  # shape (116,)

    def _is_legal(self, action):
        p = self.current_player
        amount_to_call = max(0.0, self.committed[1 - p] - self.committed[p])
        if action == CALL and amount_to_call == 0:
            # Check is legal (treated as call with 0 amount)
            return True
        if action in (RAISE_33, RAISE_75, RAISE_150):
            frac = RAISE_FRACS[action]
            raise_to = self.committed[1 - p] + frac * self.pot
            additional = raise_to - self.committed[p]
            return additional <= self.stacks[p] and self.stacks[p] > 0
        if action == ALL_IN:
            return self.stacks[p] > 0
        return True  # FOLD always legal

    def _showdown_reward(self, acting_player):
        """
        Simplified equity-based reward using preflop hand strength.
        Returns reward for acting_player relative to their investment.
        """
        p = acting_player
        opp = 1 - p

        eq_p = _preflop_equity(self.hole_cards[p], self.hole_cards[opp])
        # EV = equity * pot - amount_committed
        ev = eq_p * self.pot - self.committed[p]
        return ev

    def _card_str(self, cards):
        rank_names = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T'}
        suit_names = ['c', 'd', 'h', 's']
        out = []
        for c in cards:
            r = c // 4 + 2
            s = c % 4
            rn = rank_names.get(r, str(r))
            out.append(f"{rn}{suit_names[s]}")
        return " ".join(out)

    def get_hand_type(self, player=None):
        """Return (rank1, rank2, suited) for hole cards — used by evaluation."""
        if player is None:
            player = self.current_player
        cards = self.hole_cards[player]
        r1, r2 = cards[0] // 4 + 2, cards[1] // 4 + 2
        s1, s2 = cards[0] % 4, cards[1] % 4
        if r1 < r2:
            r1, r2 = r2, r1
        suited = (s1 == s2)
        return r1, r2, suited


# ------------------------------------------------------------------
# Preflop equity approximation (Chen formula proxy)
# ------------------------------------------------------------------

def _hand_strength(cards):
    """
    Approximate preflop hand strength using a simplified rank/suit heuristic.
    Returns a score in [0, 1].
    """
    r1 = cards[0] // 4 + 2
    r2 = cards[1] // 4 + 2
    s1 = cards[0] % 4
    s2 = cards[1] % 4
    if r1 < r2:
        r1, r2 = r2, r1
    suited = (s1 == s2)

    # Base score: high card + pair bonus
    score = r1 + r2
    if r1 == r2:
        score += 18  # pairs dominate HU (was 10, but 22 needs ~52% equity vs unpaired)
    if suited:
        score += 2
    gap = r1 - r2
    if gap == 0:
        score += 4  # connected
    elif gap == 1:
        score += 2
    elif gap == 2:
        score += 1

    # Normalize to [0, 1]: theoretical max ~50 (AA suited), min ~4 (32o)
    return np.clip((score - 4) / 46.0, 0.0, 1.0)


def _preflop_equity(cards_p, cards_opp):
    """
    Estimate heads-up preflop equity for player given opponent's cards.
    Uses relative hand strength as a fast approximation.
    """
    s_p   = _hand_strength(cards_p)
    s_opp = _hand_strength(cards_opp)
    total = s_p + s_opp
    if total == 0:
        return 0.5
    return s_p / total