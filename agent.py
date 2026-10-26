"""Compact masked DQN with uniform replay, implemented for TABasco."""
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn


def action_mask_from_observation(observation, n_tabs):
    """Derive exactly the environment's operation-validity mask from state."""
    obs = np.asarray(observation, dtype=np.float32)
    if obs.shape != (n_tabs * 8 + 5,):
        raise ValueError(f"expected {(n_tabs * 8 + 5)} observation values, got {obs.shape}")
    mask = np.zeros(n_tabs * 3 + 1, dtype=bool)
    slots = obs[:n_tabs * 8].reshape(n_tabs, 8)
    for i, slot in enumerate(slots):
        occupied, active = slot[0] >= .5, slot[1] >= .5
        if occupied:
            mask[3 * i] = True       # KEEP: no-op if active, restore if suspended
            mask[3 * i + 1] = active # SUSPEND only changes active tabs
            mask[3 * i + 2] = True   # CLOSE any occupied tab
    mask[-1] = True                 # WAIT guarantees a valid action
    return mask


def _batch_action_masks(observations, n_tabs):
    """Torch equivalent used to mask bootstrapped next-state values."""
    slots = observations[:, :n_tabs * 8].reshape(-1, n_tabs, 8)
    occupied = slots[:, :, 0] >= .5
    active = slots[:, :, 1] >= .5
    masks = torch.zeros((observations.shape[0], n_tabs * 3 + 1), dtype=torch.bool)
    masks[:, 0:n_tabs * 3:3] = occupied
    masks[:, 1:n_tabs * 3:3] = occupied & active
    masks[:, 2:n_tabs * 3:3] = occupied
    masks[:, -1] = True
    return masks


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(),
                                    nn.Linear(128, 128), nn.ReLU(),
                                    nn.Linear(128, n_actions))

    def forward(self, x):
        return self.layers(x)


class DQNAgent:
    def __init__(self, obs_dim, n_actions, seed=0, learning_rate=1e-3,
                 gamma=.97, batch_size=64, replay_capacity=50000,
                 target_update=500, epsilon_start=1., epsilon_end=.05,
                 epsilon_decay=20000, gradient_clip=10.):
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.n_tabs = (self.n_actions - 1) // 3
        if self.n_actions != self.n_tabs * 3 + 1 or self.obs_dim != self.n_tabs * 8 + 5:
            raise ValueError("DQN dimensions must match TABasco's 8N+5 observations and 3N+1 actions")
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update = int(target_update)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.gradient_clip = float(gradient_clip)
        self.steps = 0
        self.updates = 0
        self.config = {"learning_rate": learning_rate, "gamma": gamma,
                       "batch_size": batch_size, "replay_capacity": replay_capacity,
                       "target_update": target_update, "epsilon_start": epsilon_start,
                       "epsilon_end": epsilon_end, "epsilon_decay": epsilon_decay,
                       "gradient_clip": gradient_clip}
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device("cpu")
        self.online = QNetwork(self.obs_dim, self.n_actions).to(self.device)
        self.target = QNetwork(self.obs_dim, self.n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optim = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=replay_capacity)
        self.rng = random.Random(seed)

    @property
    def epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-self.steps / self.epsilon_decay)

    def action_mask(self, observation):
        return action_mask_from_observation(observation, self.n_tabs)

    def act(self, obs, explore=True, action_mask=None):
        mask = self.action_mask(obs) if action_mask is None else np.asarray(action_mask, dtype=bool)
        if mask.shape != (self.n_actions,) or not mask.any():
            raise ValueError("action mask must match the action space and contain a valid action")
        valid = np.flatnonzero(mask)
        if explore:
            self.steps += 1
            if self.rng.random() < self.epsilon:
                return int(self.rng.choice(valid.tolist()))
        with torch.no_grad():
            q = self.online(torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)).squeeze(0)
            q[~torch.as_tensor(mask)] = -torch.inf
            return int(q.argmax().item())

    def remember(self, state, action, reward, next_state, done, next_action_mask=None):
        mask = (self.action_mask(next_state) if next_action_mask is None
                else np.asarray(next_action_mask, dtype=bool))
        if mask.shape != (self.n_actions,) or not mask.any():
            raise ValueError("next action mask must match the action space and be nonempty")
        self.memory.append((np.asarray(state, dtype=np.float32), int(action), float(reward),
                            np.asarray(next_state, dtype=np.float32), float(done), mask.copy()))

    def update(self):
        if len(self.memory) < self.batch_size:
            return None
        batch = self.rng.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        states = torch.as_tensor(np.stack(states), dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.long)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        next_states = torch.as_tensor(np.stack(next_states), dtype=torch.float32)
        dones = torch.as_tensor(dones, dtype=torch.float32)
        q_values = self.online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_states)
            next_masks = torch.as_tensor(np.stack(next_masks), dtype=torch.bool)
            next_q = next_q.masked_fill(~next_masks, -torch.inf)
            targets = rewards + self.gamma * (1. - dones) * next_q.max(1).values
        loss = nn.functional.smooth_l1_loss(q_values, targets)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.gradient_clip)
        self.optim.step()
        self.updates += 1
        if self.updates % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.online.state_dict(), "obs_dim": self.obs_dim,
                    "n_actions": self.n_actions, "metadata": metadata or {},
                    "agent_config": self.config, "steps": self.steps,
                    "updates": self.updates}, path)

    @classmethod
    def load(cls, path, seed=0, **kwargs):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = dict(checkpoint.get("agent_config", {}))
        config.update(kwargs)
        agent = cls(checkpoint["obs_dim"], checkpoint["n_actions"], seed=seed, **config)
        agent.online.load_state_dict(checkpoint["state_dict"])
        agent.target.load_state_dict(agent.online.state_dict())
        agent.metadata = checkpoint.get("metadata", {})
        agent.steps = checkpoint.get("steps", 0)
        agent.updates = checkpoint.get("updates", 0)
        return agent
