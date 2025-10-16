# config.py — global configuration for Autonomous Agricultural Swarm

# Environment/grid
GRID_W = 32           # cols
GRID_H = 20           # rows
CELL_SIZE = 28        # pixels for Pygame drawing
MARGIN = 2            # cell margin for grid rendering
INITIAL_CROP_DENSITY = 0.85  # probability a cell has a crop at start

# Simulation timing
TICKS_PER_DAY = 120   # simulation steps = one "day"
RAIN_CHANCE = 0.12    # daily probability of rain event
WIND_VARIANCE = 0.4   # affects pest spread directionality

# Weather ranges
TEMP_MEAN = 29.0
TEMP_STD = 4.5
HUMID_MEAN = 0.55     # 0..1
HUMID_STD = 0.18

# Soil & crop parameters
MOISTURE_DECAY = 0.006      # per tick without irrigation/rain
NUTRIENT_DECAY = 0.0015     # per tick usage by crop
GROWTH_RATE = 0.002         # base growth per tick (scaled by health)
PEST_SPREAD_RATE = 0.0025   # chance to infect neighbor per tick
DISEASE_SPREAD_RATE = 0.0018

# Actions
ACTIONS = ["idle", "monitor", "irrigate", "apply_pesticide", "apply_fungicide", "fertilize", "harvest"]
ACTION_COSTS = {
    "idle": 0.0,
    "monitor": 0.001,
    "irrigate": 0.01,
    "apply_pesticide": 0.012,
    "apply_fungicide": 0.012,
    "fertilize": 0.009,
    "harvest": 0.003,
}

# Agents
NUM_AGENTS = 6
OBSERVATION_RADIUS = 2
MAX_BATTERY = 1.0
BATTERY_DRAIN_PER_MOVE = 0.0015
BATTERY_RECHARGE_PER_TICK = 0.002
BASE_LOCATIONS = [(1,1), (GRID_W-2, GRID_H-2)]

# Reward shaping (base)
REWARD_YIELD = 2.0
REWARD_HEALTH = 0.6
REWARD_SUSTAIN = 0.6
REWARD_ACTION_COST_SCALE = 1.0

# LLM shaping multipliers (dynamic; fed by llm_parser & GUI conditions)
LLM_SHAPING_DEFAULTS = {
    "irrigate_multiplier": 1.0,
    "pesticide_multiplier": 1.0,
    "fungicide_multiplier": 1.0,
    "monitor_multiplier": 1.0,
    "fertilize_multiplier": 1.0
}

# Visualization
WINDOW_W = 1280
WINDOW_H = 800
FPS = 60                # default FPS; can be changed live with [ and ]
PANEL_W = 360
FONT_NAME = "freesansbold.ttf"

# Sustainability metric weights
W_SUSTAIN_WATER = 0.35
W_SUSTAIN_CHEM = 0.35
W_SUSTAIN_BIODIV = 0.30

# Random seed (None = random)
SEED = 42
