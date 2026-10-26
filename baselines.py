"""Deterministic tab-management baselines for evaluation only."""
from environment import SUSPEND


def _active_indices(env):
    return [i for i, tab in enumerate(env.tabs) if tab is not None and tab["active"]]


def lru_action(env):
    """Suspend the active, valid tab with oldest last visit; wait if none."""
    eligible = _active_indices(env)
    if not eligible:
        return env.wait_action
    i = min(eligible, key=lambda j: (env.tabs[j]["last"], j))
    action = 3 * i + SUSPEND
    assert action in env.valid_actions()
    return action


def lfu_action(env):
    """Suspend the active tab with fewest visits (then oldest); wait if none."""
    eligible = _active_indices(env)
    if not eligible:
        return env.wait_action
    i = min(eligible, key=lambda j: (env.tabs[j]["freq"], env.tabs[j]["last"], j))
    action = 3 * i + SUSPEND
    assert action in env.valid_actions()
    return action


def random_valid_action(env, rng):
    """Uniform suspension over the same eligible active tabs as LRU/LFU.

    When no tab is eligible to suspend, the baseline takes the shared valid
    WAIT action. Every candidate is checked against the environment mask.
    """
    choices = [3 * i + SUSPEND for i in _active_indices(env)]
    if not choices:
        choices = [env.wait_action]
    assert all(action in env.valid_actions() for action in choices)
    return int(rng.choice(choices))
