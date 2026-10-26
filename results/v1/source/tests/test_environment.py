import numpy as np
from environment import TabManagementEnv,KEEP,SUSPEND,CLOSE

class ZeroRNG:
 def random(self): return 0.0
 def integers(self,low,high=None): return int(low)
 def uniform(self,low,high): return (low+high)/2
 def choice(self,values): return values[0]

def test_reset_and_seeded_transitions_are_reproducible():
 a=TabManagementEnv(seed=4);b=TabManagementEnv(seed=4)
 assert a.reset(20).shape==a.observation_space_shape;b.reset(20)
 assert np.array_equal(a._obs(),b._obs())
 for action in [0,1,2,3,4,5]:
  x=a.step(action);y=b.step(action);assert np.array_equal(x[0],y[0]);assert x[1:4]==y[1:4];assert x[4]==y[4]

def test_actions_and_empty_slot_invalidity():
 e=TabManagementEnv(n_tabs=3,seed=1);e.reset(1)
 for action in range(e.action_space_n): e.reset(1);e.step(action)
 e.reset(1);e.tabs[-1]=None;empty=e.n_tabs-1;_,_,_,_,info=e.step(empty*3+CLOSE)
 assert not info['action_valid'];assert e.tabs[empty] is None

def test_suspend_close_memory_and_reward_breakdown():
 e=TabManagementEnv(n_tabs=4,seed=2);e.reset(2);i=next(i for i,x in enumerate(e.tabs) if x);x=e.tabs[i];size=x['size'];e.step(i*3+SUSPEND)
 assert not e.tabs[i]['active'] and e._active_memory()<=sum(y['size'] for y in e.tabs if y)
 _,r,_,_,info=e.step(i*3+CLOSE)
 assert e.tabs[i] is None and np.isclose(r,-sum(info['reward_components'].values()))
 assert info['restoration_cost']>=0

def test_memory_nonnegative_and_episode_ends():
 e=TabManagementEnv(n_tabs=2,max_steps=3,seed=3);e.reset(3)
 for _ in range(3):
  _,_,term,_,info=e.step(0);assert info['active_memory']>=0
 assert term

def test_revisiting_suspended_incurs_cost():
 e=TabManagementEnv(n_tabs=1,seed=5);e.reset(5);e.rng=ZeroRNG();_,_,_,_,info=e.step(SUSPEND)
 assert info['revisit_event']=='suspended_revisit' and info['restoration_cost']>0

def test_closing_creates_recoverable_history():
 e=TabManagementEnv(n_tabs=2,seed=8);e.reset(8);i=next(i for i,x in enumerate(e.tabs) if x);e.rng=ZeroRNG()
 _,_,_,_,info=e.step(i*3+CLOSE)
 assert info['event']=='closed_tab_recovered' and info['restoration_cost']>0 and info['lost_work_cost']>0
