import argparse,csv,random
from pathlib import Path
import numpy as np,torch
from environment import TabManagementEnv
from agent import DQNAgent

def main():
 p=argparse.ArgumentParser();p.add_argument('--episodes',type=int,default=250);p.add_argument('--max-steps',type=int,default=120);p.add_argument('--seed',type=int,default=3090);p.add_argument('--memory-budget',type=float,default=8.);p.add_argument('--n-tabs',type=int,default=12);p.add_argument('--checkpoint',default='models/tabasco_dqn.pt');p.add_argument('--results-dir',default='results');p.add_argument('--learning-rate',type=float,default=.001);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--gamma',type=float,default=.97);p.add_argument('--replay-capacity',type=int,default=50000);p.add_argument('--target-update',type=int,default=500);a=p.parse_args()
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(1)
 env=TabManagementEnv(a.n_tabs,a.memory_budget,a.max_steps,a.seed);agent=DQNAgent(env.observation_space_shape[0],env.action_space_n,a.seed,a.learning_rate,a.gamma,a.batch_size,a.replay_capacity,a.target_update)
 out=Path(a.results_dir);out.mkdir(parents=True,exist_ok=True);rows=[]
 for ep in range(a.episodes):
  obs=env.reset(a.seed+ep+1);ret=0.;losses=[];mem=[];viol=0;restore=0;lost=0
  for _ in range(a.max_steps):
   action=agent.act(obs);n,r,term,trunc,info=env.step(action);agent.remember(obs,action,r,n,term or trunc);loss=agent.update()
   if loss is not None:losses.append(loss)
   obs=n;ret+=r;mem.append(info['active_memory']);viol+=info['memory_violation'];restore+=info['restoration_cost'];lost+=info['lost_work_cost']
   if term or trunc:break
  rows.append({'episode':ep+1,'return':ret,'mean_loss':float(np.mean(losses)) if losses else 0.,'mean_active_memory':float(np.mean(mem)),'peak_active_memory':max(mem,default=0),'memory_violations':viol,'restoration_cost':restore,'lost_work_cost':lost,'epsilon':agent.epsilon})
  if (ep+1)%max(1,a.episodes//10)==0:print(f"episode {ep+1}/{a.episodes} return={ret:.2f} loss={rows[-1]['mean_loss']:.4f} epsilon={agent.epsilon:.3f}")
 with (out/'training.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 agent.save(a.checkpoint,vars(a));print(f"Saved checkpoint: {a.checkpoint}");print(f"Mean return last 20: {np.mean([r['return'] for r in rows[-20:]]):.3f}")
if __name__=='__main__':main()
