import argparse,csv,random
from pathlib import Path
import numpy as np
from environment import TabManagementEnv
from agent import DQNAgent
from baselines import lru_action,lfu_action,random_valid_action

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='models/tabasco_dqn.pt');p.add_argument('--episodes',type=int,default=60);p.add_argument('--seed',type=int,default=90000);p.add_argument('--max-steps',type=int,default=120);p.add_argument('--memory-budget',type=float,default=8.);p.add_argument('--n-tabs',type=int,default=12);p.add_argument('--results-dir',default='results');a=p.parse_args()
 out=Path(a.results_dir);out.mkdir(parents=True,exist_ok=True);agent=DQNAgent.load(a.checkpoint,seed=a.seed); policies={'DQN':lambda e,r:agent.act(e._obs(),explore=False),'LRU':lambda e,r:lru_action(e),'LFU':lambda e,r:lfu_action(e),'Random':lambda e,r:random_valid_action(e,r)};rows=[]
 for ep in range(a.episodes):
  seed=a.seed+ep
  for name,choose in policies.items():
   env=TabManagementEnv(a.n_tabs,a.memory_budget,a.max_steps,seed);env.reset(seed);rng=random.Random(seed+777);ret=0.;mem=[];viol=0.;restore=lost=0.;useful=total=invalid=0
   for _ in range(a.max_steps):
    act=choose(env,rng);_,r,term,trunc,info=env.step(act);ret+=r;mem.append(info['active_memory']);viol+=info['memory_excess'];restore+=info['restoration_cost'];lost+=info['lost_work_cost'];useful+=info['useful_revisit'];total+=info['revisit_occurred'];invalid+=not info['action_valid']
    if term or trunc:break
   rows.append({'episode':ep+1,'seed':seed,'policy':name,'return':ret,'mean_active_memory':np.mean(mem),'peak_active_memory':max(mem,default=0),'violation_steps':sum(1 for m in mem if m>a.memory_budget),'violation_magnitude':viol,'restoration_recovery_cost':restore,'lost_work_cost':lost,'useful_revisit_success_rate':useful/max(1,total),'invalid_action_rate':invalid/max(1,len(mem))})
 fields=rows[0].keys()
 with (out/'evaluation_episodes.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 metrics=[k for k in fields if k not in ('episode','seed','policy')]
 with (out/'evaluation_summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['policy']+[v for m in metrics for v in (m+'_mean',m+'_std')]);w.writeheader()
  for name in policies:
   subset=[x for x in rows if x['policy']==name];w.writerow({'policy':name,**{z:x for m in metrics for z,x in ((m+'_mean',np.mean([r[m] for r in subset])),(m+'_std',np.std([r[m] for r in subset])))}})
 print((out/'evaluation_summary.csv').read_text())
if __name__=='__main__':main()
