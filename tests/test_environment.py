import random

import numpy as np
import torch

from agent import DQNAgent, _batch_action_masks, action_mask_from_observation
from baselines import lfu_action, lru_action, random_valid_action
from environment import CLOSE, KEEP, SUSPEND, WAIT, TabManagementEnv


class NoEventsRNG:
    """Predictable random source for testing action effects without user events."""
    def random(self):
        return .99

    def integers(self, low, high=None):
        return int(low)

    def uniform(self, low, high):
        return (low + high) / 2

    def choice(self, values):
        return values[0]


class ZeroRNG(NoEventsRNG):
    def random(self):
        return 0.


class ExpireOnceRNG(NoEventsRNG):
    def __init__(self):
        self.values = iter([.99, .99, .99, 0.])

    def random(self):
        return next(self.values, .99)


class ClosedReturnWithoutSpaceRNG(NoEventsRNG):
    """Avoid user events, then force a closed-tab return draw."""
    def __init__(self):
        self.values = iter([.99, .99, .99, .0, .99, .99])

    def random(self):
        return next(self.values, .99)


class ScriptedEventsRNG(NoEventsRNG):
    def __init__(self, values):
        self.values = iter(values)

    def random(self):
        return next(self.values, .99)

    def uniform(self, low, high):
        return high


def test_reset_and_seeded_transitions_are_reproducible():
    a = TabManagementEnv(seed=4)
    b = TabManagementEnv(seed=4)
    obs_a = a.reset(20)
    obs_b = b.reset(20)
    assert obs_a.shape == a.observation_space_shape == (101,)
    assert np.array_equal(obs_a, obs_b)
    for _ in range(20):
        action = a.valid_actions()[0]
        xa, ra, ta, tra, ia = a.step(action)
        xb, rb, tb, trb, ib = b.step(action)
        assert np.array_equal(xa, xb)
        assert (ra, ta, tra, ia) == (rb, tb, trb, ib)


def test_mask_matches_environment_and_covers_every_nonterminal_state():
    env = TabManagementEnv(n_tabs=4, max_steps=20, seed=5)
    obs = env.reset(5)
    for _ in range(20):
        expected = env.action_mask(obs)
        agent_mask = action_mask_from_observation(obs, env.n_tabs)
        assert np.array_equal(expected, agent_mask)
        assert expected[-1]  # WAIT is always valid.
        assert np.flatnonzero(expected).tolist() == env.valid_actions()
        obs, _, _, truncated, _ = env.step(env.valid_actions()[0])
        if truncated:
            break


def test_every_masked_action_executes_and_empty_close_is_identified():
    probe = TabManagementEnv(n_tabs=4, max_steps=5, seed=7)
    initial = probe.reset(7)
    actions = probe.valid_actions()
    for action in actions:
        env = TabManagementEnv(n_tabs=4, max_steps=5, seed=7)
        obs = env.reset(7)
        env.rng = NoEventsRNG()
        assert env.action_mask(obs)[action]
        _, _, _, _, info = env.step(action)
        assert info["action_valid"]
    empty_slot = next((i for i in range(probe.n_tabs)
                       if initial[i * 8] == 0), None)
    if empty_slot is not None:
        env = TabManagementEnv(n_tabs=4, seed=7)
        env.reset(7)
        env.rng = NoEventsRNG()
        _, _, _, _, info = env.step(3 * empty_slot + CLOSE)
        assert not info["action_valid"]
        assert info["action_invalid_reason"] == "close_missing_tab"


def test_empty_slot_and_redundant_suspend_are_invalid_but_keep_noop_is_valid():
    env = TabManagementEnv(n_tabs=3, seed=11)
    env.reset(11)
    env.rng = NoEventsRNG()
    env.tabs[-1] = None
    obs = env._obs()
    assert not env.action_mask(obs)[3 * 2 + CLOSE]
    _, reward, _, _, info = env.step(3 * 2 + CLOSE)
    assert not info["action_valid"]
    assert info["action_invalid_reason"] == "close_missing_tab"
    assert info["action_cost"] == env.c["invalid"]
    assert reward == -env.c["invalid"]

    suspended = next(i for i, tab in enumerate(env.tabs) if tab is not None)
    env.tabs[suspended]["active"] = False
    assert not env.action_mask()[3 * suspended + SUSPEND]
    _, reward, _, _, info = env.step(3 * suspended + SUSPEND)
    assert not info["action_valid"]
    assert info["action_invalid_reason"] == "suspend_already_suspended"
    assert reward == -env.c["invalid"]

    active = next(i for i, tab in enumerate(env.tabs) if tab is not None)
    env.tabs[active]["active"] = True
    _, reward, _, _, info = env.step(3 * active + KEEP)
    assert info["action_valid"] and info["action_noop"]
    assert info["action_cost"] == env.c["action"]
    assert reward == -env.c["action"]


def test_dqn_exploration_and_greedy_selection_respect_action_mask():
    env = TabManagementEnv(n_tabs=3, seed=13)
    obs = env.reset(13)
    env.tabs[-1] = None
    obs = env._obs()
    mask = env.action_mask(obs)
    batch_masks = _batch_action_masks(torch.as_tensor(np.stack([obs, obs])), env.n_tabs).numpy()
    assert np.array_equal(batch_masks[0], mask)
    assert np.array_equal(batch_masks[1], mask)
    agent = DQNAgent(len(obs), env.action_space_n, seed=13)
    # Force the unmasked Q maximum onto an empty-slot CLOSE action.
    for parameter in agent.online.parameters():
        parameter.data.zero_()
    invalid = 3 * (env.n_tabs - 1) + CLOSE
    agent.online.layers[-1].bias.data[invalid] = 100.
    greedy = agent.act(obs, explore=False, action_mask=mask)
    assert mask[greedy]
    assert greedy != invalid
    for _ in range(500):
        action = agent.act(obs, explore=True, action_mask=mask)
        assert mask[action]
    # A train update must also have finite targets with all empty next-state slots.
    empty_obs = np.zeros_like(obs)
    empty_obs[-3] = 1.  # normalized remaining horizon; slot fields stay empty
    for _ in range(agent.batch_size):
        agent.remember(empty_obs, env.wait_action, -env.c["action"], empty_obs, False)
    loss = agent.update()
    assert loss is not None and np.isfinite(loss)


def test_suspend_frees_exact_active_memory_and_reward_components_sum():
    env = TabManagementEnv(n_tabs=4, seed=17)
    env.reset(17)
    idx = next(i for i, tab in enumerate(env.tabs) if tab is not None)
    size = env.tabs[idx]["size"]
    before = env._active_memory()
    env.rng = NoEventsRNG()
    _, reward, _, _, info = env.step(3 * idx + SUSPEND)
    assert np.isclose(info["active_memory"], before - size)
    assert info["active_memory"] >= 0
    assert env.tabs[idx] is not None and not env.tabs[idx]["active"]
    assert np.isclose(reward, -sum(info["reward_components"].values()))
    assert np.isclose(info["memory_penalty"], max(0., info["active_memory"] - env.memory_budget))


def test_active_keep_has_no_memory_benefit_and_closing_is_not_rewarded():
    env = TabManagementEnv(n_tabs=2, seed=19)
    env.reset(19)
    env.rng = NoEventsRNG()
    idx = next(i for i, tab in enumerate(env.tabs) if tab is not None)
    before = env._active_memory()
    _, keep_reward, _, _, keep_info = env.step(3 * idx + KEEP)
    assert keep_info["action_noop"] and keep_info["active_memory"] == before
    assert keep_reward < 0
    if env.tabs[idx] is not None:
        _, close_reward, terminated, truncated, close_info = env.step(3 * idx + CLOSE)
        assert close_reward <= 0
        assert not terminated and not truncated
        assert close_info["event"] == "close_active"
        assert env.tabs[idx] is None


def test_repeated_invalid_actions_never_generate_positive_reward_or_early_end():
    env = TabManagementEnv(n_tabs=3, max_steps=6, seed=21)
    env.reset(21)
    env.rng = NoEventsRNG()
    invalid_action = env.action_space_n + 5
    for step in range(5):
        _, reward, terminated, truncated, info = env.step(invalid_action)
        assert reward <= -env.c["invalid"]
        assert not info["action_valid"]
        assert not terminated and not truncated


def test_suspended_and_closed_returns_charge_configured_costs():
    env = TabManagementEnv(n_tabs=1, seed=23)
    env.reset(23)
    size = env.tabs[0]["size"]
    env.rng = ZeroRNG()
    _, _, _, _, suspended = env.step(SUSPEND)
    assert "suspended_revisit" in suspended["revisit_events"]
    assert np.isclose(suspended["restoration_cost"], env.c["restore"] * .22 * size)

    env = TabManagementEnv(n_tabs=2, seed=29)
    env.reset(29)
    idx = next(i for i, tab in enumerate(env.tabs) if tab is not None)
    size = env.tabs[idx]["size"]
    risk = env.tabs[idx]["risk"]
    env.rng = ZeroRNG()
    _, _, _, _, recovered = env.step(3 * idx + CLOSE)
    assert "closed_tab_recovered" in recovered["revisit_events"]
    assert np.isclose(recovered["restoration_cost"], env.c["restore"] * .55 * size)
    # Closing charges .65*risk; the immediate simulated return charges .45*risk.
    assert np.isclose(recovered["lost_work_cost"], env.c["lost"] * ( .65 + .45) * risk)


def test_natural_expiration_is_charged_and_retained_for_recovery():
    env = TabManagementEnv(n_tabs=1, seed=30)
    env.reset(30)
    env.tabs[0]["age"] = 25
    env.tabs[0]["risk"] = .8
    env.tabs[0]["p"] = .2
    env.rng = ExpireOnceRNG()
    _, _, _, _, info = env.step(env.wait_action)
    assert info["expired_tab_count"] == 1
    assert env.tabs[0] is None
    assert env.closed and env.closed[-1]["risk"] == .8
    assert np.isclose(info["lost_work_cost"], env.c["lost"] * .4 * .8)


def test_closed_history_is_not_consumed_when_browser_has_no_empty_slot():
    env = TabManagementEnv(n_tabs=2, seed=32)
    env.reset(32)
    env.tabs = [None, None]
    env._new_tab(0)
    env._new_tab(1)
    record = {"p": 1., "size": .5, "risk": .4, "task": env.task,
              "closed_at": env.t}
    env.closed = [record]
    env.rng = ClosedReturnWithoutSpaceRNG()
    _, _, _, _, info = env.step(env.wait_action)
    assert not info["recovered"]
    assert env.closed == [record]
    assert all(tab is not None for tab in env.tabs)


def test_empty_browser_waits_until_fixed_horizon_without_early_termination():
    env = TabManagementEnv(n_tabs=2, max_steps=3, seed=31)
    env.reset(31)
    env.tabs = [None, None]
    env.rng = NoEventsRNG()
    for step in range(3):
        assert env.valid_actions() == [env.wait_action]
        _, _, terminated, truncated, info = env.step(env.wait_action)
        assert not terminated
        assert truncated == (step == 2)
        assert info["action_valid"]
    with np.testing.assert_raises(RuntimeError):
        env.step(WAIT)


def test_baselines_only_emit_actions_in_same_environment_valid_set():
    env = TabManagementEnv(n_tabs=5, seed=37)
    env.reset(37)
    actions = [lru_action(env), lfu_action(env), random_valid_action(env, random.Random(37))]
    assert all(action in env.valid_actions() for action in actions)
    active_suspensions = {3 * i + SUSPEND for i, tab in enumerate(env.tabs)
                          if tab is not None and tab["active"]}
    if active_suspensions:
        assert actions[0] in active_suspensions
        assert actions[1] in active_suspensions
        rng = random.Random(3700)
        assert all(random_valid_action(env, rng) in active_suspensions for _ in range(100))
    env.tabs = [None] * env.n_tabs
    assert lru_action(env) == env.wait_action
    assert lfu_action(env) == env.wait_action
    assert random_valid_action(env, random.Random(37)) == env.wait_action


def test_constraint_aware_environment_never_exceeds_budget_across_seeded_rollouts():
    for seed in range(12):
        env = TabManagementEnv(n_tabs=12, memory_budget=4., max_steps=60,
                               seed=seed, constraint_aware=True)
        assert env._active_memory() <= env.memory_budget + 1e-8
        for _ in range(env.max_steps):
            obs = env._obs()
            mask = env.action_mask(obs)
            action = int(np.flatnonzero(mask)[seed % mask.sum()])
            next_obs, reward, terminated, truncated, info = env.step(action)
            assert info["active_memory"] <= env.memory_budget + 1e-8
            assert info["memory_excess"] == 0.
            assert max(info[key] for key in ("memory_after_action", "memory_after_revisits",
                                             "memory_after_closed_recovery", "memory_after_arrival",
                                             "memory_after_expiration")) <= env.memory_budget + 1e-8
            assert np.isclose(reward, -sum(info["reward_components"].values()))
            assert env.action_mask(next_obs).any()
            if terminated or truncated:
                break


def test_constraint_mask_disables_restore_that_cannot_fit():
    env = TabManagementEnv(n_tabs=2, memory_budget=1., seed=41, constraint_aware=True)
    env.reset(41)
    env.tabs = [
        {"active": True, "size": .9, "last": 0, "freq": 1, "task": env.task,
         "p": .2, "risk": .2, "age": 0},
        {"active": False, "size": .7, "last": 0, "freq": 1, "task": env.task,
         "p": .2, "risk": .2, "age": 0},
    ]
    assert not env.action_mask()[3 + KEEP]
    assert 3 + KEEP not in env.valid_actions()


def test_constraint_defers_over_budget_revisit_and_arrival():
    env = TabManagementEnv(n_tabs=2, memory_budget=1., seed=43, constraint_aware=True)
    env.reset(43)
    env.tabs = [
        {"active": True, "size": .9, "last": 0, "freq": 1, "task": env.task,
         "p": .2, "risk": .2, "age": 0},
        {"active": False, "size": .7, "last": 0, "freq": 1, "task": env.task,
         "p": .2, "risk": .2, "age": 0},
    ]
    env.rng = ScriptedEventsRNG([.99, .99, 0., .99])
    _, _, _, _, info = env.step(env.wait_action)
    assert "suspended_revisit_deferred" in info["revisit_events"]
    assert env.tabs[1] is not None and not env.tabs[1]["active"]
    assert info["active_memory"] <= env.memory_budget
    assert info["deferred_event_cost"] == env.c["deferred"]

    env = TabManagementEnv(n_tabs=2, memory_budget=.5, seed=47, constraint_aware=True)
    env.reset(47)
    env.tabs = [{"active": True, "size": .5, "last": 0, "freq": 1,
                 "task": env.task, "p": .2, "risk": .2, "age": 0}, None]
    env.rng = ScriptedEventsRNG([.99, .99, .0])
    _, _, _, _, info = env.step(env.wait_action)
    assert not info["arrival_created"]
    assert info["deferred_event_cost"] == env.c["deferred"]
    assert env._active_memory() <= env.memory_budget


def test_constraint_defers_closed_recovery_and_retains_history_when_full():
    env = TabManagementEnv(n_tabs=3, memory_budget=1., seed=53, constraint_aware=True)
    env.reset(53)
    env.tabs = [{"active": True, "size": .9, "last": 0, "freq": 1,
                 "task": env.task, "p": .2, "risk": .2, "age": 0}, None, None]
    record = {"size": .7, "risk": .4, "task": env.task, "p": 1., "freq": 2,
              "last": 0, "age": 4, "closed_at": 0}
    env.closed = [record]
    env.rng = ScriptedEventsRNG([.99, .99, 0., .99])
    _, _, _, _, info = env.step(env.wait_action)
    assert "closed_recovery_deferred" in info["revisit_events"]
    assert env.closed == [record]
    assert info["deferred_event_cost"] == env.c["deferred"]
    assert env._active_memory() <= env.memory_budget
