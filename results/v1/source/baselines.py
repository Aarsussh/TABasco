"""Deterministic tab policies for evaluation; never used for DQN labels."""
from environment import SUSPEND
import numpy as np

def lru_action(env):
    choices=[(x["last"],i) for i,x in enumerate(env.tabs) if x and x["active"]]
    return min(choices)[1]*3+SUSPEND if choices else 0
def lfu_action(env):
    choices=[(x["freq"],x["last"],i) for i,x in enumerate(env.tabs) if x and x["active"]]
    return min(choices)[2]*3+SUSPEND if choices else 0
def random_valid_action(env,rng):
    choices=[i*3+SUSPEND for i,x in enumerate(env.tabs) if x and x["active"]]
    return int(rng.choice(choices)) if choices else 0
