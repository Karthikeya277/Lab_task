# main.py — Simulator + LLM parser + agents + 3D visualization

from config import *
from simulator import FarmSimulator
from rl_swarm import RuleBasedAgent, SimpleA2CAgent
from llm_parser import parse_report
from visualization_3d_1 import simulate_and_render_3d   # <-- 3D renderer only

def build_agents(mode="rule"):
    return [RuleBasedAgent(i) if mode=="rule" else SimpleA2CAgent(i) for i in range(NUM_AGENTS)]

def main():
    sim = FarmSimulator()
    report = (
        "Weather bulletin: heatwave expected; humidity moderate. "
        "Agronomy watch: aphid activity rising in northern parcels. "
        "Blight risk near low-drainage zones. Increase monitoring."
    )
    llm = parse_report(report)
    sim.llm_shaping.update(llm["multipliers"])  # base (GUI may adjust further)
    agents = build_agents(mode="rule")  # or "a2c"

    # run live 3D visualization
    simulate_and_render_3d(sim, agents, llm["summary"])

if __name__ == "__main__":
    main()
