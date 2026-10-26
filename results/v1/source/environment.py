"""Synthetic, fixed-size browser-tab management MDP for TABasco."""
from __future__ import annotations
import numpy as np

KEEP, SUSPEND, CLOSE = 0, 1, 2
OP_NAMES = ("KEEP", "SUSPEND", "CLOSE")

class TabManagementEnv:
    """One tab action per step. Observation is (N, 8) flattened plus 5 globals.

    Slot columns: occupied, active, normalized size, recency, frequency,
    current-task match, estimated revisit probability, unsaved-work risk.
    Globals: active memory/budget, current task/4, remaining horizon, step,
    occupied fraction. Active memory alone is budget constrained; suspended
    tabs retain metadata with a small fixed 0.02 memory fraction each.
    Actions are tab_index*3 + {KEEP,SUSPEND,CLOSE}; invalid/empty actions
    cost 0.12 and do not alter the slot. Every action advances time.
    """
    def __init__(self, n_tabs=12, memory_budget=8.0, max_steps=120, seed=None, coefficients=None):
        self.n_tabs=int(n_tabs); self.memory_budget=float(memory_budget); self.max_steps=int(max_steps)
        if self.n_tabs < 1 or self.memory_budget <= 0 or self.max_steps < 1: raise ValueError("n_tabs, budget, and max_steps must be positive")
        self.observation_space_shape=(self.n_tabs*8+5,); self.action_space_n=self.n_tabs*3
        self.c={"memory":1.0,"restore":0.8,"lost":1.2,"disruption":0.45,"action":0.035,"invalid":0.12}
        if coefficients:self.c.update(coefficients)
        self.rng=np.random.default_rng(seed); self._seed=seed
        self.reset(seed)
    def reset(self, seed=None):
        if seed is not None: self.rng=np.random.default_rng(seed)
        self.t=0; self.task=int(self.rng.integers(4)); self.tabs=[None]*self.n_tabs; self.closed=[]
        self.memory_violations=0
        initial_count=int(self.rng.integers(1,self.n_tabs+1))
        for i in range(initial_count): self._new_tab(i)
        self._update_revisit_probabilities()
        return self._obs()
    def _new_tab(self,i,task=None):
        task=int(self.rng.integers(4)) if task is None else task
        self.tabs[i]={"active":True,"size":float(self.rng.uniform(.45,1.35)),"last":self.t,"freq":1,
          "task":task,"p":float(self.rng.uniform(.12,.78)),"risk":float(self.rng.uniform(0,1)),"age":0,"status":"active"}
    def _active_memory(self): return sum(x["size"] for x in self.tabs if x and x["active"])
    def _update_revisit_probabilities(self):
        for x in self.tabs:
            if x:
                age=max(0,self.t-x["last"]); affinity=1.0 if x["task"]==self.task else .28
                x["p"]=float(np.clip((.12+.68*np.exp(-age/22)+.08*np.log1p(x["freq"])) * affinity, .03,.92))
    def _obs(self):
        rows=[]
        for x in self.tabs:
            if x is None: rows.extend([0.0]*8)
            else: rows.extend([1.,float(x["active"]),x["size"]/1.35,min(1,(self.t-x["last"])/100),min(1,x["freq"]/20),float(x["task"]==self.task),x["p"],x["risk"]])
        rows += [self._active_memory()/self.memory_budget,self.task/3,max(0,(self.max_steps-self.t)/self.max_steps),self.t/self.max_steps,sum(x is not None for x in self.tabs)/self.n_tabs]
        return np.asarray(rows,dtype=np.float32)
    def decode_action(self,action):
        a=int(action)
        if a<0 or a>=self.action_space_n: return None,None
        return a//3,a%3
    def eligible_actions(self):
        return [i*3+op for i,x in enumerate(self.tabs) if x for op in (KEEP,SUSPEND,CLOSE)]
    def step(self,action):
        i,op=self.decode_action(action); components={"memory_penalty":0.,"restoration_cost":0.,"lost_work_cost":0.,"task_disruption_cost":0.,"action_cost":0.}
        valid=i is not None and self.tabs[i] is not None; event=None; restored=False; recovered=False
        if not valid: components["action_cost"]+=self.c["invalid"]
        else:
            x=self.tabs[i]
            if op==KEEP:
                if not x["active"]:
                    cost=.12 + x["size"]*.18; components["restoration_cost"]+=cost; event="restore_suspended"; restored=True
                    x["active"]=True;x["status"]="active"
                else: components["action_cost"]+=self.c["action"]*.25
            elif op==SUSPEND:
                if not x["active"]: components["action_cost"]+=self.c["action"]
                else:
                    x["active"]=False;x["status"]="suspended";event="suspend"
                    components["task_disruption_cost"]+=self.c["disruption"]*x["p"]*(1.0 if x["task"]==self.task else .35)
            else:
                components["task_disruption_cost"]+=self.c["disruption"]*x["p"]*(1.0 if x["task"]==self.task else .35)
                components["lost_work_cost"]+=self.c["lost"]*x["risk"]*.65
                if not x["active"]: components["restoration_cost"]+=.18*x["size"];event="close_suspended"
                else:event="close_active"
                self.closed.append({"size":x["size"],"risk":x["risk"],"task":x["task"],"p":x["p"],"closed_at":self.t});self.tabs[i]=None
            components["action_cost"]+=self.c["action"]
        # Time evolution: task switches, revisit events, arrivals and expiration.
        self.t+=1
        if self.rng.random()<.11: self.task=int(self.rng.integers(4))
        self._update_revisit_probabilities()
        revisit_event=None; useful=0
        candidates=[(j,x) for j,x in enumerate(self.tabs) if x and self.rng.random()<x["p"]]
        if candidates:
            j,x=max(candidates,key=lambda pair:pair[1]["p"]); useful=int(x["task"]==self.task)
            if x["active"]: revisit_event="active_revisit"
            elif x:
                if x["status"]=="suspended": components["restoration_cost"]+=.22*x["size"];x["active"]=True;x["status"]="active";revisit_event="suspended_revisit";restored=True
                else: components["restoration_cost"]+=.55*x["size"];components["lost_work_cost"]+=.45*x["risk"];x["active"]=True;x["status"]="active";revisit_event="closed_recovery";recovered=True
            x["last"]=self.t;x["freq"]+=1;x["risk"]*=.75
        # Closed tabs leave a short-lived history record and may be revisited.
        self.closed=[g for g in self.closed if self.t-g["closed_at"]<30]
        if self.closed and self.rng.random()<max(g["p"] for g in self.closed)*.35:
            g=max(self.closed,key=lambda q:q["p"]);self.closed.remove(g);empty=[j for j,x in enumerate(self.tabs) if x is None]
            if empty:
                components["restoration_cost"]+=.55*g["size"];components["lost_work_cost"]+=.45*g["risk"]
                self._new_tab(int(self.rng.choice(empty)),g["task"]);event="closed_tab_recovered";recovered=True;useful=int(g["task"]==self.task)
        for x in self.tabs:
            if x:x["age"]+=1
        # Stochastic arrivals occupy empty slots; old tabs expire with small hazard.
        if self.rng.random()<.16:
            empty=[j for j,x in enumerate(self.tabs) if x is None]
            if empty:self._new_tab(int(self.rng.choice(empty)))
        for j,x in enumerate(self.tabs):
            if x and x["age"]>24 and self.rng.random()<.012:self.tabs[j]=None
        # Measure budget after the full transition, including arrivals/restores.
        excess=max(0.,self._active_memory()-self.memory_budget); components["memory_penalty"]=self.c["memory"]*excess
        if excess>0:self.memory_violations+=1
        reward=-sum(components.values())
        terminated=self.t>=self.max_steps or all(x is None for x in self.tabs)
        info={**components,"reward_components":dict(components),"action_valid":bool(valid),"action":int(action),"action_name":OP_NAMES[op] if op is not None else "INVALID","tab_index":i,"event":event,"revisit_event":revisit_event,"revisit_occurred":bool(revisit_event or recovered),"restored":restored,"recovered":recovered,"useful_revisit":useful,"active_memory":self._active_memory(),"memory_budget":self.memory_budget,"memory_violation":excess>0,"memory_excess":excess,"occupied_tabs":sum(x is not None for x in self.tabs),"reward":reward}
        return self._obs(),float(reward),bool(terminated),False,info
