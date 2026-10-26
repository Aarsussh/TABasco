import csv
import os
from pathlib import Path
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'tabasco-mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def read(path):
 try:
  with Path(path).open(newline='') as f:return list(csv.DictReader(f))
 except (FileNotFoundError, OSError):return []
def main():
 out=Path('results');out.mkdir(exist_ok=True);train=read(out/'training.csv');eps=read(out/'evaluation_episodes.csv')
 if train:
  x=np.array([int(r['episode']) for r in train]);y=np.array([float(r['return']) for r in train]);window=max(1,min(20,len(y)//10 or 1));roll=np.convolve(y,np.ones(window)/window,mode='valid')
  fig,ax=plt.subplots(figsize=(8,4.5));ax.plot(x,y,alpha=.25,label='Episode return');ax.plot(x[window-1:],roll,label=f'{window}-episode rolling mean',linewidth=2);ax.set(xlabel='Training episode',ylabel='Return',title='TABasco DQN training');ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'training_returns.png',dpi=180);plt.close(fig)
 if eps:
  names=list(dict.fromkeys(r['policy'] for r in eps));vals=[np.array([float(r['return']) for r in eps if r['policy']==n]) for n in names];fig,ax=plt.subplots(figsize=(7,4.5));ax.bar(names,[v.mean() for v in vals],yerr=[v.std() for v in vals],capsize=4);ax.set(ylabel='Episode return (mean ± SD)',title='Held-out policy comparison');ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(out/'return_comparison.png',dpi=180);plt.close(fig)
  fig,ax=plt.subplots(figsize=(7,4.5));ax.bar(names,[np.mean([float(r['mean_active_memory']) for r in eps if r['policy']==n]) for n in names]);ax.axhline(8,color='red',linestyle='--',label='Memory budget');ax.set(ylabel='Mean active memory (synthetic units)',title='Active memory by policy');ax.legend();ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(out/'memory_comparison.png',dpi=180);plt.close(fig)
 print('Plots saved where source CSV data was available.')
if __name__=='__main__':main()
