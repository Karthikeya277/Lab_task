# rl_swarm.py — Multi-agent RL scaffold (simple A2C-like) + Rule-based fallback

import random
from typing import List, Tuple
import numpy as np
from config import *
from simulator import FarmSimulator

rng = random.Random(SEED)
MOVES = [(0,0),(1,0),(-1,0),(0,1),(0,-1)]
def clip(v, lo, hi): return max(lo, min(hi, v))

class RuleBasedAgent:
    def __init__(self, idx: int): self.idx = idx
    def act(self, sim: FarmSimulator):
        a = sim.agents[self.idx]
        cell = sim.grid[a.x][a.y]
        if cell.pest > 0.5: return "apply_pesticide", (0,0)
        if cell.disease > 0.4: return "apply_fungicide", (0,0)
        if cell.moisture < 0.35: return "irrigate", (0,0)
        if cell.nutrient < 0.35: return "fertilize", (0,0)
        if cell.crop and cell.growth > 0.85 and cell.health() > 0.6: return "harvest", (0,0)
        dx, dy = rng.choice(MOVES)
        return "monitor", (dx, dy)

class SimpleA2CAgent:
    def __init__(self, idx: int, lr=0.05, gamma=0.99):
        self.idx = idx; self.lr=lr; self.gamma=gamma
        self.table = {}; self.prev=None
    def _state_hash(self, sim: FarmSimulator):
        a=sim.agents[self.idx]; c=sim.grid[a.x][a.y]
        def bucket(v): return int(clip(v,0,0.999)*4)
        return (bucket(c.moisture), bucket(c.nutrient), bucket(c.pest), bucket(c.disease),
                int(c.growth*4), 1 if c.crop else 0)
    def _policy(self, s):
        prefs=self.table.get(s); 
        if prefs is None: prefs={a:0.0 for a in ACTIONS}; self.table[s]=prefs
        logits=np.array([prefs[a] for a in ACTIONS], dtype=float); logits-=logits.max()
        return np.exp(logits)/np.exp(logits).sum()
    def act(self, sim: FarmSimulator):
        a=sim.agents[self.idx]
        if rng.random()<0.7: dx,dy=rng.choice(MOVES)
        else:
            cx,cy=sim.w//2, sim.h//2
            dx=1 if a.x<cx and rng.random()<0.5 else (-1 if a.x>cx and rng.random()<0.5 else 0)
            dy=1 if a.y<cy and rng.random()<0.5 else (-1 if a.y>cy and rng.random()<0.5 else 0)
        s=self._state_hash(sim); probs=self._policy(s)
        action=rng.choices(ACTIONS, weights=probs, k=1)[0]
        self.prev=(s,action); return action,(dx,dy)
    def learn(self, reward, done=False):
        if self.prev is None: return
        s,a=self.prev; prefs=self.table[s]
        for act in ACTIONS: prefs[act]*=(1 - self.lr*0.01)
        prefs[a]+=self.lr*reward
        self.prev=None if done else self.prev

def move_agent(sim: FarmSimulator, idx: int, dx: int, dy: int):
    a=sim.agents[idx]
    nx=clip(a.x+dx,0,sim.w-1); ny=clip(a.y+dy,0,sim.h-1)
    if (nx,ny)!=(a.x,a.y):
        a.battery=max(0.0, a.battery - BATTERY_DRAIN_PER_MOVE)
    a.x,a.y=nx,ny
