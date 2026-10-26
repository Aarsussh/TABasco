"""Synthetic browser-tab management MDP used by TABasco.

Observation: float32 vector of length 8*N + 5. Per-tab fields are
occupied, active, size/1.35, recency/100, frequency/20, current-task match,
estimated revisit probability, and unsaved-work risk. Globals are active
memory/budget, task/3, remaining horizon, elapsed fraction, and occupancy.

Actions 3*i + operation address KEEP, SUSPEND, or CLOSE for tab i. A final
WAIT action is always available, including during an empty-browser interval,
so every nonterminal state has a valid action. SUSPEND is valid only for an
active occupied tab; KEEP on an active tab is a harmless no-op; KEEP on a
suspended tab restores it. Empty-slot operations and redundant suspension are
invalid. The fixed horizon ends by truncation, never merely because tabs close.
Only active tabs count against the memory budget; suspension releases the full
tab size from active memory and retains its metadata at zero active-memory cost.
"""
from __future__ import annotations

import numpy as np

KEEP, SUSPEND, CLOSE, WAIT = 0, 1, 2, 3
OP_NAMES = ("KEEP", "SUSPEND", "CLOSE", "WAIT")


class TabManagementEnv:
    """Fixed-size, reproducible simulated browser session."""

    def __init__(self, n_tabs=12, memory_budget=8.0, max_steps=120, seed=None,
                 coefficients=None, constraint_aware=False):
        self.n_tabs = int(n_tabs)
        self.memory_budget = float(memory_budget)
        self.max_steps = int(max_steps)
        self.constraint_aware = bool(constraint_aware)
        if self.n_tabs < 1 or self.memory_budget <= 0 or self.max_steps < 1:
            raise ValueError("n_tabs, budget, and max_steps must be positive")
        self.observation_space_shape = (self.n_tabs * 8 + 5,)
        self.action_space_n = self.n_tabs * 3 + 1
        self.wait_action = self.n_tabs * 3
        # Costs are positive quantities and reward is their negative sum.
        self.c = {"memory": 1.0, "restore": 0.8, "lost": 1.2,
                  "disruption": 0.45, "action": 0.035, "invalid": 0.12,
                  "deferred": 0.25}
        if coefficients:
            self.c.update(coefficients)
        self.rng = np.random.default_rng(seed)
        self.reset(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.task = int(self.rng.integers(4))
        self.tabs = [None] * self.n_tabs
        self.closed = []
        self.memory_violations = 0
        initial_count = int(self.rng.integers(1, self.n_tabs + 1))
        for i in range(initial_count):
            self._new_tab(i)
            if self.constraint_aware and self._active_memory() > self.memory_budget:
                self.tabs[i]["active"] = False
        self._update_revisit_probabilities()
        self.initial_memory = self._active_memory()
        self.memory_violation_source = "initial_state" if self.initial_memory > self.memory_budget else None
        return self._obs()

    def _sample_tab(self, task=None):
        task = int(self.rng.integers(4)) if task is None else task
        return {"active": True, "size": float(self.rng.uniform(.45, 1.35)),
                "last": self.t, "freq": 1, "task": task,
                "p": float(self.rng.uniform(.12, .78)),
                "risk": float(self.rng.uniform(0, 1)), "age": 0}

    def _new_tab(self, i, task=None):
        self.tabs[i] = self._sample_tab(task)

    def _active_memory(self):
        return sum(tab["size"] for tab in self.tabs if tab and tab["active"])

    def _memory_source_after(self, before, after, crossing_source, existing_source):
        """Track the first stage that crossed the budget; clear after recovery."""
        if after <= self.memory_budget:
            return None, None
        if before <= self.memory_budget:
            return crossing_source, crossing_source
        return existing_source, None

    def _update_revisit_probabilities(self):
        for tab in self.tabs:
            if tab:
                age = max(0, self.t - tab["last"])
                affinity = 1.0 if tab["task"] == self.task else .28
                tab["p"] = float(np.clip((.12 + .68 * np.exp(-age / 22) +
                                          .08 * np.log1p(tab["freq"])) * affinity, .03, .92))

    def _obs(self):
        rows = []
        for tab in self.tabs:
            if tab is None:
                rows.extend([0.0] * 8)
            else:
                rows.extend([1., float(tab["active"]), tab["size"] / 1.35,
                             min(1., (self.t - tab["last"]) / 100),
                             min(1., tab["freq"] / 20), float(tab["task"] == self.task),
                             tab["p"], tab["risk"]])
        rows += [self._active_memory() / self.memory_budget, self.task / 3,
                 max(0., (self.max_steps - self.t) / self.max_steps),
                 self.t / self.max_steps,
                 sum(tab is not None for tab in self.tabs) / self.n_tabs]
        return np.asarray(rows, dtype=np.float32)

    def decode_action(self, action):
        try:
            a = int(action)
        except (TypeError, ValueError, OverflowError):
            return None, None
        if a != action or a < 0 or a >= self.action_space_n:
            return None, None
        if a == self.wait_action:
            return None, WAIT
        return a // 3, a % 3

    @staticmethod
    def mask_from_observation(observation, n_tabs, memory_budget=None, constraint_aware=False):
        """Return the bool action mask derivable from the public observation."""
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (n_tabs * 8 + 5,):
            raise ValueError(f"expected observation shape {(n_tabs * 8 + 5,)}, got {obs.shape}")
        mask = np.zeros(n_tabs * 3 + 1, dtype=bool)
        slots = obs[:n_tabs * 8].reshape(n_tabs, 8)
        active_memory = float(obs[n_tabs * 8]) * float(memory_budget or 1.)
        for i, slot in enumerate(slots):
            occupied = slot[0] >= .5
            active = slot[1] >= .5
            if occupied:
                can_restore = (not constraint_aware or active or
                               (memory_budget is not None and
                                active_memory + float(slot[2]) * 1.35 <= memory_budget + 1e-8))
                mask[3 * i + KEEP] = can_restore
                mask[3 * i + SUSPEND] = active
                mask[3 * i + CLOSE] = True
        mask[-1] = True  # WAIT guarantees an action in every state.
        return mask

    def valid_actions(self):
        return np.flatnonzero(self.action_mask()).tolist()

    def action_mask(self, observation=None):
        return self.mask_from_observation(self._obs() if observation is None else observation,
                                          self.n_tabs, self.memory_budget,
                                          self.constraint_aware)

    def step(self, action):
        if self.t >= self.max_steps:
            raise RuntimeError("step called after the episode horizon")
        i, op = self.decode_action(action)
        components = {"memory_penalty": 0., "restoration_cost": 0.,
                      "lost_work_cost": 0., "task_disruption_cost": 0.,
                      "action_cost": 0., "deferred_event_cost": 0.}
        pre_obs = self._obs()
        pre_mask = self.action_mask(pre_obs)
        memory_before_action = self._active_memory()
        memory_source = self.memory_violation_source
        memory_crossing_source = None
        in_range = i is not None or op == WAIT
        valid = bool(in_range and pre_mask[int(action)])
        event = None
        restored = False
        recovered = False
        no_op = False
        if not valid:
            components["action_cost"] = self.c["invalid"]
            if i is not None and self.tabs[i] is None:
                reason = "close_missing_tab" if op == CLOSE else "empty_slot"
            elif (i is not None and op == KEEP and self.tabs[i] is not None and
                  not self.tabs[i]["active"] and self.constraint_aware):
                reason = "restore_exceeds_budget"
            elif i is not None and op == SUSPEND and self.tabs[i] is not None:
                reason = "suspend_already_suspended"
            else:
                reason = "out_of_range_or_nonintegral"
        elif op == WAIT:
            components["action_cost"] = self.c["action"]
            no_op = True
            reason = "wait"
        else:
            tab = self.tabs[i]
            components["action_cost"] = self.c["action"]
            reason = "valid"
            if op == KEEP:
                if tab["active"]:
                    no_op = True
                else:
                    components["restoration_cost"] += self.c["restore"] * (.12 + .18 * tab["size"])
                    event = "restore_suspended"
                    restored = True
                    tab["active"] = True
            elif op == SUSPEND:
                tab["active"] = False
                event = "suspend"
                components["task_disruption_cost"] += self.c["disruption"] * tab["p"] * (
                    1.0 if tab["task"] == self.task else .35)
            elif op == CLOSE:
                components["task_disruption_cost"] += self.c["disruption"] * tab["p"] * (
                    1.0 if tab["task"] == self.task else .35)
                components["lost_work_cost"] += self.c["lost"] * .65 * tab["risk"]
                event = "close_active" if tab["active"] else "close_suspended"
                self.closed.append({"size": tab["size"], "risk": tab["risk"],
                                    "task": tab["task"], "p": tab["p"], "closed_at": self.t})
                self.tabs[i] = None

        memory_after_action = self._active_memory()
        memory_source, crossing = self._memory_source_after(
            memory_before_action, memory_after_action, "agent_action", memory_source)
        memory_crossing_source = crossing or memory_crossing_source
        memory_before_revisits = memory_after_action

        # Time evolution follows the management action. A real return may
        # immediately restore a suspended page, undoing the action's memory save.
        self.t += 1
        if self.rng.random() < .11:
            self.task = int(self.rng.integers(4))
        self._update_revisit_probabilities()
        revisit_events = []
        useful_revisits = 0
        candidates = [(j, tab) for j, tab in enumerate(self.tabs)
                      if tab and self.rng.random() < tab["p"]]
        if candidates:
            _, tab = max(candidates, key=lambda pair: pair[1]["p"])
            if not tab["active"]:
                if (self.constraint_aware and
                        self._active_memory() + tab["size"] > self.memory_budget + 1e-8):
                    revisit_events.append("suspended_revisit_deferred")
                    components["deferred_event_cost"] += self.c["deferred"]
                else:
                    components["restoration_cost"] += self.c["restore"] * .22 * tab["size"]
                    tab["active"] = True
                    restored = True
                    useful_revisits += int(tab["task"] == self.task)
                    revisit_events.append("suspended_revisit")
            else:
                revisit_events.append("active_revisit")
                useful_revisits += int(tab["task"] == self.task)
            tab["last"] = self.t
            tab["freq"] += 1
            tab["risk"] *= .75

        memory_after_revisits = self._active_memory()
        memory_source, crossing = self._memory_source_after(
            memory_before_revisits, memory_after_revisits, "suspended_revisit", memory_source)
        memory_crossing_source = crossing or memory_crossing_source

        # A closed page remains in simulated history for 30 steps. A return
        # recreates it and charges both restoration and lost-work risk.
        self.closed = [record for record in self.closed if self.t - record["closed_at"] < 30]
        closed_return_draw = self.rng.random() if self.closed else 1.
        empty = [j for j, tab in enumerate(self.tabs) if tab is None]
        if (self.closed and empty and
                closed_return_draw < max(record["p"] for record in self.closed) * .35):
            record = max(self.closed, key=lambda item: item["p"])
            fits_budget = (not self.constraint_aware or
                           self._active_memory() + record["size"] <= self.memory_budget + 1e-8)
            if fits_budget:
                self.closed.remove(record)
                components["restoration_cost"] += self.c["restore"] * .55 * record["size"]
                components["lost_work_cost"] += self.c["lost"] * .45 * record["risk"]
                if self.constraint_aware:
                    restored_tab = dict(record)
                    restored_tab.update(active=True, last=self.t, age=0,
                                        freq=int(record.get("freq", 1)))
                    restored_tab.pop("closed_at", None)
                    self.tabs[int(self.rng.choice(empty))] = restored_tab
                else:
                    self._new_tab(int(self.rng.choice(empty)), record["task"])
                useful_revisits += int(record["task"] == self.task)
                revisit_events.append("closed_tab_recovered")
                recovered = True
            else:
                revisit_events.append("closed_recovery_deferred")
                components["deferred_event_cost"] += self.c["deferred"]

        memory_after_closed_recovery = self._active_memory()
        memory_source, crossing = self._memory_source_after(
            memory_after_revisits, memory_after_closed_recovery, "closed_recovery", memory_source)
        memory_crossing_source = crossing or memory_crossing_source

        for tab in self.tabs:
            if tab:
                tab["age"] += 1
        arrival_created = False
        if self.rng.random() < .16:
            empty = [j for j, tab in enumerate(self.tabs) if tab is None]
            if empty:
                slot_index = int(self.rng.choice(empty))
                if self.constraint_aware:
                    arrival_tab = self._sample_tab()
                    if self._active_memory() + arrival_tab["size"] <= self.memory_budget + 1e-8:
                        self.tabs[slot_index] = arrival_tab
                        arrival_created = True
                    else:
                        components["deferred_event_cost"] += self.c["deferred"]
                else:
                    self._new_tab(slot_index)
                    arrival_created = True
        memory_after_arrival = self._active_memory()
        memory_source, crossing = self._memory_source_after(
            memory_after_closed_recovery, memory_after_arrival, "new_arrival", memory_source)
        memory_crossing_source = crossing or memory_crossing_source
        expired_tabs = 0
        for j, tab in enumerate(self.tabs):
            if tab and tab["age"] > 24 and self.rng.random() < .012:
                components["lost_work_cost"] += self.c["lost"] * .4 * tab["risk"]
                self.closed.append({"size": tab["size"], "risk": tab["risk"],
                                    "task": tab["task"], "p": tab["p"], "closed_at": self.t})
                self.tabs[j] = None
                expired_tabs += 1

        memory_before_expiration = memory_after_arrival
        memory_after_expiration = self._active_memory()
        memory_source, crossing = self._memory_source_after(
            memory_before_expiration, memory_after_expiration, "expiration", memory_source)
        memory_crossing_source = crossing or memory_crossing_source

        # Apply memory cost to final post-event active memory, including arrivals.
        excess = max(0., self._active_memory() - self.memory_budget)
        if self.constraint_aware and excess > 1e-7:
            raise RuntimeError("constraint-aware transition exceeded active-memory budget")
        components["memory_penalty"] = self.c["memory"] * excess
        if excess > 0:
            self.memory_violations += 1
        self.memory_violation_source = memory_source if excess > 0 else None
        reward = -sum(components.values())
        truncated = self.t >= self.max_steps
        info = {**components, "reward_components": dict(components),
                "action_valid": valid, "action_noop": no_op,
                "action_invalid_reason": None if valid else reason,
                "action": action, "action_name": OP_NAMES[op] if op is not None else "INVALID",
                "tab_index": i, "event": event,
                "revisit_events": tuple(revisit_events),
                "revisit_event_count": len(revisit_events),
                "expired_tab_count": expired_tabs,
                "revisit_occurred": bool(revisit_events),
                "restored": restored, "recovered": recovered,
                "useful_revisit_count": useful_revisits,
                "useful_revisit": int(useful_revisits > 0),
                "active_memory": self._active_memory(), "memory_budget": self.memory_budget,
                "memory_violation": excess > 0, "memory_excess": excess,
                "memory_before_action": memory_before_action,
                "memory_after_action": memory_after_action,
                "memory_after_revisits": memory_after_revisits,
                "memory_after_closed_recovery": memory_after_closed_recovery,
                "memory_after_arrival": memory_after_arrival,
                "memory_after_expiration": memory_after_expiration,
                "memory_violation_source": memory_source if excess > 0 else None,
                "memory_violation_crossing_source": memory_crossing_source,
                "arrival_created": arrival_created,
                "automatic_revisit_restored": "suspended_revisit" in revisit_events,
                "closed_recovery_created": recovered,
                "occupied_tabs": sum(tab is not None for tab in self.tabs),
                "valid_action_count": int(self.action_mask().sum()), "reward": reward}
        return self._obs(), float(reward), False, bool(truncated), info
