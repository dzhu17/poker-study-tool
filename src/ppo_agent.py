import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# Hyperparameters (defaults — overridable at construction time)
# ---------------------------------------------------------------------------
DEFAULT_LR          = 3e-4
DEFAULT_GAMMA       = 0.99
DEFAULT_GAE_LAMBDA  = 0.95
DEFAULT_CLIP_EPS    = 0.2
DEFAULT_ENTROPY_COEF = 0.01
DEFAULT_VALUE_COEF  = 0.5
DEFAULT_MAX_GRAD_NORM = 0.5
DEFAULT_UPDATE_EPOCHS = 4
DEFAULT_MINIBATCH   = 64
DEFAULT_DROPOUT     = 0.1


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """
    Shared-trunk actor-critic network.

    Trunk: 116 → 256 → 128 (ReLU + Dropout for regularization)
    Actor head: 128 → 64 → 6  (action logits → Categorical policy)
    Critic head: 128 → 64 → 1 (state value estimate)
    """

    def __init__(self, obs_dim=116, action_dim=6, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # smaller gain for output layers
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)

    def forward(self, obs):
        features = self.trunk(obs)
        logits = self.actor_head(features)
        value  = self.critic_head(features).squeeze(-1)
        return logits, value

    def get_action(self, obs):
        """Sample an action and return (action, log_prob, value)."""
        logits, value = self.forward(obs)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(self, obs, actions):
        """Return (log_probs, values, entropy) for a batch of (obs, action) pairs."""
        logits, value = self.forward(obs)
        dist      = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, value, entropy

    def get_action_probs(self, obs):
        """Return action probability vector (for inference / web app)."""
        with torch.no_grad():
            logits, _ = self.forward(obs)
            return F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores one batch of environment transitions for a PPO update."""

    def __init__(self):
        self.observations  = []
        self.actions       = []
        self.log_probs     = []
        self.rewards       = []
        self.values        = []
        self.dones         = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """
    PPO agent wrapping ActorCritic with:
      - GAE advantage estimation
      - Clipped surrogate objective
      - Value function loss
      - Entropy bonus (exploration regularization)
      - Gradient clipping
      - Checkpoint save / load
    """

    def __init__(
        self,
        obs_dim=116,
        action_dim=6,
        lr=DEFAULT_LR,
        gamma=DEFAULT_GAMMA,
        gae_lambda=DEFAULT_GAE_LAMBDA,
        clip_eps=DEFAULT_CLIP_EPS,
        entropy_coef=DEFAULT_ENTROPY_COEF,
        value_coef=DEFAULT_VALUE_COEF,
        max_grad_norm=DEFAULT_MAX_GRAD_NORM,
        update_epochs=DEFAULT_UPDATE_EPOCHS,
        minibatch_size=DEFAULT_MINIBATCH,
        dropout=DEFAULT_DROPOUT,
        device=None,
        lr_schedule_steps=0,
    ):
        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.clip_eps     = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef   = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size

        self.device = device or torch.device("cpu")
        self.network = ActorCritic(obs_dim, action_dim, dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        # Cosine annealing LR schedule: decays LR from lr → eta_min over training
        self.scheduler = None
        if lr_schedule_steps > 0:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=lr_schedule_steps, eta_min=1e-6
            )

        self.buffer = RolloutBuffer()

    # ------------------------------------------------------------------
    # Action selection (used during rollout collection)
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray):
        """
        obs: numpy array of shape (obs_dim,)
        Returns: action (int), log_prob (float), value (float)
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        self.network.eval()
        with torch.no_grad():
            action_t, log_prob_t, value_t = self.network.get_action(obs_t)
        self.network.train()
        return action_t.item(), log_prob_t.item(), value_t.item()

    def get_probs(self, obs: np.ndarray) -> np.ndarray:
        """Return action probability array — used by Flask app."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        probs = self.network.get_action_probs(obs_t)
        return probs.squeeze(0).numpy()

    # ------------------------------------------------------------------
    # GAE — Generalized Advantage Estimation
    # ------------------------------------------------------------------

    def compute_gae(self, next_value: float) -> tuple:
        """
        Compute advantages (GAE-λ) and discounted returns from the buffer.

        GAE formula:
            δ_t  = r_t + γ·V(s_{t+1})·(1-done) − V(s_t)
            A_t  = δ_t + (γλ)·A_{t+1}

        Returns:
            advantages: np.ndarray shape (T,)
            returns:    np.ndarray shape (T,)  (used as value targets)
        """
        rewards  = self.buffer.rewards
        values   = self.buffer.values
        dones    = self.buffer.dones
        T = len(rewards)

        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            next_val   = next_value if t == T - 1 else values[t + 1]
            next_done  = dones[t]
            delta      = rewards[t] + self.gamma * next_val * (1.0 - next_done) - values[t]
            gae        = delta + self.gamma * self.gae_lambda * (1.0 - next_done) * gae
            advantages[t] = gae

        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, next_value: float = 0.0) -> dict:
        """
        Run PPO update epochs over the current buffer.
        Returns a dict of loss statistics for logging.
        """
        advantages, returns = self.compute_gae(next_value)

        # Normalize advantages (reduces variance)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert buffer to tensors
        obs_t     = torch.FloatTensor(np.array(self.buffer.observations)).to(self.device)
        act_t     = torch.LongTensor(self.buffer.actions).to(self.device)
        old_lp_t  = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        adv_t     = torch.FloatTensor(advantages).to(self.device)
        ret_t     = torch.FloatTensor(returns).to(self.device)

        T = len(self.buffer)
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        num_updates       = 0

        for _ in range(self.update_epochs):
            # Shuffle indices for minibatch sampling
            indices = np.random.permutation(T)

            for start in range(0, T, self.minibatch_size):
                mb_idx = indices[start: start + self.minibatch_size]

                mb_obs  = obs_t[mb_idx]
                mb_act  = act_t[mb_idx]
                mb_adv  = adv_t[mb_idx]
                mb_ret  = ret_t[mb_idx]
                mb_old_lp = old_lp_t[mb_idx]

                log_probs, values, entropy = self.network.evaluate_actions(mb_obs, mb_act)

                # PPO clipped surrogate objective
                ratio  = torch.exp(log_probs - mb_old_lp)
                surr1  = ratio * mb_adv
                surr2  = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss
                value_loss = F.mse_loss(values, mb_ret)

                # Entropy bonus (encourages exploration)
                entropy_loss = entropy.mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy_loss.item()
                num_updates       += 1

        self.buffer.clear()

        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss":  total_value_loss  / num_updates,
            "entropy":     total_entropy     / num_updates,
            "lr":          self.optimizer.param_groups[0]["lr"],
        }

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.network.eval()
