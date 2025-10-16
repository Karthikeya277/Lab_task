# simulator.py — Farm environment simulator

import math, random
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from config import *

rng = random.Random(SEED)

CROP_TYPES = ["wheat", "corn", "soy"]
CROP_GROWTH_TIME = {"wheat": 14, "corn": 18, "soy": 16}
CROP_VALUE = {"wheat": 1.0, "corn": 1.3, "soy": 1.1}

@dataclass
class Cell:
    crop: Optional[str] = None
    moisture: float = 0.6
    nutrient: float = 0.7
    pest: float = 0.0
    disease: float = 0.0
    growth: float = 0.0
    last_action: str = "idle"
    def health(self):
        good = 0.5*(self.moisture + self.nutrient)
        bad = 0.6*self.pest + 0.6*self.disease
        return max(0.0, min(1.0, good - bad))

@dataclass
class Weather:
    temp: float = TEMP_MEAN
    humidity: float = HUMID_MEAN
    rain: float = 0.0
    wind_dx: float = 0.0
    wind_dy: float = 0.0

@dataclass
class AgentState:
    x: int = 0
    y: int = 0
    battery: float = MAX_BATTERY
    carrying: Optional[str] = None
    last_action: str = "idle"
    reward: float = 0.0
    harvested_today: int = 0

class FarmSimulator:
    def __init__(self, w=GRID_W, h=GRID_H):
        self.w, self.h = w, h
        self.grid = [[Cell() for _ in range(h)] for __ in range(w)]
        self.weather = Weather()
        self.ticks = 0
        self.day = 0
        self.total_yield = 0.0
        self.total_water_used = 0.0
        self.total_chem_used = 0.0
        self.total_monitoring = 0
        self.biodiversity_score = 1.0
        self.llm_shaping = dict(LLM_SHAPING_DEFAULTS)

        for x in range(w):
            for y in range(h):
                if rng.random() < INITIAL_CROP_DENSITY:
                    self.grid[x][y].crop = rng.choice(CROP_TYPES)
                    self.grid[x][y].growth = rng.uniform(0.15, 0.4)
                    if rng.random() < 0.03:
                        self.grid[x][y].pest = rng.uniform(0.2, 0.6)
                    if rng.random() < 0.02:
                        self.grid[x][y].disease = rng.uniform(0.2, 0.5)

        self.agents: List[AgentState] = []
        for i in range(NUM_AGENTS):
            bx, by = BASE_LOCATIONS[i % len(BASE_LOCATIONS)]
            self.agents.append(AgentState(
                x=min(max(0, bx + rng.randint(-1,1)), self.w-1),
                y=min(max(0, by + rng.randint(-1,1)), self.h-1)
            ))

    def step_weather(self):
        if self.ticks % TICKS_PER_DAY == 0 and self.ticks > 0:
            self.day += 1
            self.weather.temp = float(np.clip(rng.gauss(TEMP_MEAN, TEMP_STD), 12, 44))
            self.weather.humidity = float(np.clip(rng.gauss(HUMID_MEAN, HUMID_STD), 0.05, 0.95))
            self.weather.rain = 1.0 if rng.random() < RAIN_CHANCE else 0.0
            self.weather.wind_dx = float(np.clip(rng.uniform(-1,1), -1, 1))
            self.weather.wind_dy = float(np.clip(rng.uniform(-1,1), -1, 1))

    def spread_process(self):
        new_pest = np.zeros((self.w, self.h))
        new_dis = np.zeros((self.w, self.h))
        for x in range(self.w):
            for y in range(self.h):
                c = self.grid[x][y]
                if c.pest > 0.05:
                    for nx, ny in self.neighbors(x, y):
                        biasx = 1 + WIND_VARIANCE*self.weather.wind_dx if nx > x else 1
                        biasy = 1 + WIND_VARIANCE*self.weather.wind_dy if ny > y else 1
                        p = PEST_SPREAD_RATE * c.pest * biasx * biasy
                        if rng.random() < p:
                            new_pest[nx, ny] = max(new_pest[nx, ny], 0.15*c.pest)
                if c.disease > 0.05:
                    for nx, ny in self.neighbors(x, y):
                        p = DISEASE_SPREAD_RATE * c.disease
                        if rng.random() < p:
                            new_dis[nx, ny] = max(new_dis[nx, ny], 0.12*c.disease)
        for x in range(self.w):
            for y in range(self.h):
                self.grid[x][y].pest = min(1.0, self.grid[x][y].pest + new_pest[x, y])
                self.grid[x][y].disease = min(1.0, self.grid[x][y].disease + new_dis[x, y])

    def growth_process(self):
        rain_bonus = 0.18 if self.weather.rain > 0 else 0.0
        for x in range(self.w):
            for y in range(self.h):
                c = self.grid[x][y]
                if not c.crop: continue
                c.moisture = max(0.0, min(1.0, c.moisture - MOISTURE_DECAY + rain_bonus*0.5*self.weather.humidity))
                c.nutrient = max(0.0, min(1.0, c.nutrient - NUTRIENT_DECAY))
                h = c.health()
                c.growth = max(0.0, min(1.0, c.growth + GROWTH_RATE * (0.5 + h)))
                if c.moisture > 0.7 and c.nutrient > 0.7:
                    c.pest = max(0.0, c.pest - 0.0008)
                    c.disease = max(0.0, c.disease - 0.0008)

        crop_counts = {t:0 for t in CROP_TYPES}
        total = 0
        for x in range(self.w):
            for y in range(self.h):
                if self.grid[x][y].crop:
                    crop_counts[self.grid[x][y].crop] += 1
                    total += 1
        if total > 0:
            evenness = math.exp(-sum([(c/total)*math.log((c/total)+1e-6) for c in crop_counts.values()]))
            evenness /= len(CROP_TYPES)
            self.biodiversity_score = 0.5*self.biodiversity_score + 0.5*max(0.2, min(1.0, evenness))

    def neighbors(self, x, y):
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                if dx==0 and dy==0: continue
                nx, ny = x+dx, y+dy
                if 0 <= nx < self.w and 0 <= ny < self.h:
                    yield nx, ny

    def apply_action(self, agent_idx: int, action: str):
        a = self.agents[agent_idx]
        c = self.grid[a.x][a.y]
        a.last_action = action
        c.last_action = action
        reward = 0.0
        cost = ACTION_COSTS.get(action, 0.0)

        if action == "irrigate":
            c.moisture = min(1.0, c.moisture + 0.35)
            self.total_water_used += 1.0
            reward += 0.05 * self.llm_shaping.get("irrigate_multiplier",1.0)
        elif action == "apply_pesticide":
            before = c.pest
            c.pest = max(0.0, c.pest - 0.4)
            delta = before - c.pest
            self.total_chem_used += 1.0
            reward += 0.08 * delta * self.llm_shaping.get("pesticide_multiplier",1.0)
        elif action == "apply_fungicide":
            before = c.disease
            c.disease = max(0.0, c.disease - 0.4)
            delta = before - c.disease
            self.total_chem_used += 1.0
            reward += 0.08 * delta * self.llm_shaping.get("fungicide_multiplier",1.0)
        elif action == "fertilize":
            c.nutrient = min(1.0, c.nutrient + 0.25)
            reward += 0.05 * self.llm_shaping.get("fertilize_multiplier",1.0)
        elif action == "monitor":
            self.total_monitoring += 1
            reward += 0.01 * self.llm_shaping.get("monitor_multiplier",1.0)
        elif action == "harvest":
            if c.crop and c.growth > 0.8 and c.health() > 0.5:
                yield_gain = CROP_VALUE[c.crop] * (0.4 + 0.6*c.health())
                self.total_yield += yield_gain
                a.harvested_today += 1
                c.crop = rng.choice(CROP_TYPES) if rng.random()<0.7 else None
                c.growth = 0.0
                c.pest *= 0.3
                c.disease *= 0.3
                reward += REWARD_YIELD * yield_gain
            else:
                reward -= 0.02

        reward -= REWARD_ACTION_COST_SCALE * cost
        a.reward += reward
        return reward

    def sustainability_index(self):
        water_penalty = 1.0 / (1.0 + 0.02*self.total_water_used)
        chem_penalty = 1.0 / (1.0 + 0.03*self.total_chem_used)
        sustain = (W_SUSTAIN_WATER * water_penalty +
                   W_SUSTAIN_CHEM * chem_penalty +
                   W_SUSTAIN_BIODIV * self.biodiversity_score)
        return max(0.0, min(1.0, sustain))

    def step(self):
        self.ticks += 1
        self.step_weather()
        self.spread_process()
        self.growth_process()
