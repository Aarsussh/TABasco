"""Compact experience-replay DQN implemented specifically for TABasco."""
import random
from collections import deque
from pathlib import Path
import numpy as np
import torch
from torch import nn

class QNetwork(nn.Module):
    def __init__(self,obs_dim,n_actions):
        super().__init__(); self.layers=nn.Sequential(nn.Linear(obs_dim,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,n_actions))
    def forward(self,x): return self.layers(x)

class DQNAgent:
    def __init__(self,obs_dim,n_actions,seed=0,learning_rate=1e-3,gamma=.97,batch_size=64,replay_capacity=50000,target_update=500,epsilon_start=1.,epsilon_end=.05,epsilon_decay=20000):
        self.obs_dim=obs_dim;self.n_actions=n_actions;self.gamma=gamma;self.batch_size=batch_size;self.target_update=target_update
        self.epsilon_start=epsilon_start;self.epsilon_end=epsilon_end;self.epsilon_decay=epsilon_decay;self.steps=0;self.updates=0
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        self.device=torch.device("cpu");self.online=QNetwork(obs_dim,n_actions).to(self.device);self.target=QNetwork(obs_dim,n_actions).to(self.device);self.target.load_state_dict(self.online.state_dict());self.target.eval()
        self.optim=torch.optim.Adam(self.online.parameters(),lr=learning_rate);self.memory=deque(maxlen=replay_capacity);self.rng=random.Random(seed)
    @property
    def epsilon(self): return self.epsilon_end+(self.epsilon_start-self.epsilon_end)*np.exp(-self.steps/self.epsilon_decay)
    def act(self,obs,explore=True):
        self.steps+=1
        if explore and self.rng.random()<self.epsilon:return self.rng.randrange(self.n_actions)
        with torch.no_grad():return int(self.online(torch.as_tensor(np.asarray(obs,dtype=np.float32)).unsqueeze(0)).argmax(1).item())
    def remember(self,s,a,r,n,done): self.memory.append((np.asarray(s,dtype=np.float32),int(a),float(r),np.asarray(n,dtype=np.float32),float(done)))
    def update(self):
        if len(self.memory)<self.batch_size:return None
        batch=self.rng.sample(self.memory,self.batch_size);s,a,r,n,d=zip(*batch)
        s=torch.as_tensor(np.stack(s));a=torch.as_tensor(a,dtype=torch.long);r=torch.as_tensor(r,dtype=torch.float32);n=torch.as_tensor(np.stack(n));d=torch.as_tensor(d,dtype=torch.float32)
        q=self.online(s).gather(1,a[:,None]).squeeze(1)
        with torch.no_grad(): target=r+self.gamma*(1-d)*self.target(n).max(1).values
        loss=nn.functional.smooth_l1_loss(q,target);self.optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.online.parameters(),10.);self.optim.step()
        self.updates+=1
        if self.updates%self.target_update==0:self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())
    def save(self,path,metadata=None):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"state_dict":self.online.state_dict(),"obs_dim":self.obs_dim,"n_actions":self.n_actions,"metadata":metadata or {}},p)
    @classmethod
    def load(cls,path,seed=0,**kwargs):
        ckpt=torch.load(path,map_location="cpu",weights_only=False);agent=cls(ckpt["obs_dim"],ckpt["n_actions"],seed=seed,**kwargs);agent.online.load_state_dict(ckpt["state_dict"]);agent.target.load_state_dict(agent.online.state_dict());agent.metadata=ckpt.get("metadata",{});return agent
