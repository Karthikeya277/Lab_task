# visualization.py — Clean UI + Pro controls (legend + buttons moved bottom-left)
# - Legend sits at bottom-left of the grid
# - Weather toggles (buttons + R/S/W/D hotkeys) placed beside the legend (2×2)
# - Unified HUD box for legend+controls to look polished
# - Live speed controls: +/− for move rate, [/] for FPS
# - Pause/Step (Space pause, N step)
# - Scrollable advisories; neat right panel; reward shaping tied to toggles
# - Snapshot (O) & Screenshot (P)

import pygame, random, json, time
from pygame import gfxdraw
from typing import List
from config import *
from simulator import FarmSimulator
from rl_swarm import move_agent

# --- defaults (modifiable live) ---
MOVE_EVERY_N_TICKS_DEFAULT = 8

def clamp(v, lo=0, hi=255): return max(lo, min(hi, v))

class ToggleButton:
    def __init__(self, rect, label, get_state, set_state):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.get_state = get_state
        self.set_state = set_state
    def draw(self, surface, font):
        active = self.get_state()
        bg = (210,235,220) if active else (240,244,250)
        border = (60,160,90) if active else (200,210,225)
        fg = (22,60,30) if active else (35,45,58)
        pygame.draw.rect(surface, bg, self.rect, border_radius=10)
        pygame.draw.rect(surface, border, self.rect, 1, border_radius=10)
        text = font.render(self.label + ("  ●" if active else "  ○"), True, fg)
        surface.blit(text, (self.rect.x + (self.rect.w-text.get_width())//2,
                            self.rect.y + (self.rect.h-text.get_height())//2))
    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.set_state(not self.get_state())

class FarmViz:
    def __init__(self, sim: FarmSimulator):
        pygame.init()
        self.sim = sim
        self.w = WINDOW_W; self.h = WINDOW_H
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Autonomous Agricultural Swarm")
        self.clock = pygame.time.Clock()
        self.font_xs = pygame.font.Font(FONT_NAME, 12)
        self.font_sm = pygame.font.Font(FONT_NAME, 14)
        self.font_md = pygame.font.Font(FONT_NAME, 18)
        self.font_lg = pygame.font.Font(FONT_NAME, 22)

        # Layout
        self.grid_w = self.w - PANEL_W - 12
        self.grid_h = self.h - 12
        self.grid_origin = (6,6)

        # Colors
        self.COL_BG = (238,243,248)
        self.COL_PANEL = (252,253,255)
        self.COL_TEXT = (30,40,52)
        self.COL_FRAME = (210,218,230)
        self.COL_RAIN = (120,160,255,100)

        # Precompute cell rects
        self.rects = [[None]*self.sim.h for _ in range(self.sim.w)]
        for x in range(self.sim.w):
            for y in range(self.sim.h):
                px = self.grid_origin[0] + x*(CELL_SIZE+MARGIN)
                py = self.grid_origin[1] + y*(CELL_SIZE+MARGIN)
                self.rects[x][y] = pygame.Rect(px, py, CELL_SIZE, CELL_SIZE)

        # Weather toggles
        self.conditions = {"rainy": False, "sunny": False, "wind_storm": False, "drought": False}

        # UI controls (buttons are positioned dynamically near legend)
        self.buttons: List[ToggleButton] = []
        self._init_weather_toggles()

        # Advisories scroll
        self.advisory_scroll = 0; self.advisory_line_h = 16

        # Sim control
        self.paused = False
        self.move_every = MOVE_EVERY_N_TICKS_DEFAULT
        self.current_fps = FPS

        # Reward shaping base (LLM multipliers at start); GUI conditions apply multiplicatively on top each tick
        self.base_shaping = dict(self.sim.llm_shaping)

        # Cached rects for HUD layout
        self.legend_rect_cache = None
        self.controls_rect_cache = None
        self.hud_rect_cache = None

    # -------- Weather Controls --------
    def _init_weather_toggles(self):
        # placeholder rects; real positions are computed each frame in position_hud()
        dummy = (0,0,120,32)
        def add(label, key):
            self.buttons.append(ToggleButton(
                dummy, label,
                get_state=lambda k=key: self.conditions[k],
                set_state=lambda v,k=key: self._toggle_condition(k, v)
            ))
        add("Rainy", "rainy")
        add("Sunny", "sunny")
        add("Wind Storm", "wind_storm")
        add("Drought", "drought")

    def _toggle_condition(self, key, val):
        self.conditions[key] = val
        # simple mutual exclusivity for rainy/sunny
        if key == "rainy" and val: self.conditions["sunny"] = False
        if key == "sunny" and val: self.conditions["rainy"] = False

    # --- Dynamic layout for bottom-left HUD (legend + buttons) ---
    def compute_legend_rect(self):
        # Fixed legend size; will sit at bottom-left inside the grid bounds
        legend_w, legend_h = 210, 96
        x = self.grid_origin[0] + 10
        y = self.grid_origin[1] + self.grid_h - legend_h - 10
        return pygame.Rect(x, y, legend_w, legend_h)

    def position_hud(self):
        """Place legend at bottom-left and buttons to its right in a 2×2 grid."""
        legend = self.compute_legend_rect()
        # Buttons layout to the right of legend
        bw, bh, gap = 120, 32, 10
        bx = legend.right + 12
        by = legend.top
        # Ensure buttons don't go beyond grid area; if tight, stack under legend
        max_buttons_w = 2*bw + gap
        right_space = self.grid_origin[0] + self.grid_w - bx - 10
        if right_space < max_buttons_w:
            # not enough horizontal space: put buttons below legend
            bx = legend.left
            by = legend.bottom + 10

        # Assign rects to the 4 buttons
        coords = [
            (bx, by),
            (bx + bw + gap, by),
            (bx, by + bh + gap),
            (bx + bw + gap, by + bh + gap)
        ]
        for btn, (x, y) in zip(self.buttons, coords):
            btn.rect.update(x, y, bw, bh)

        # Controls bounding box (around buttons grid)
        min_x = min(c[0] for c in coords)
        min_y = min(c[1] for c in coords)
        max_x = max(c[0] for c in coords) + bw
        max_y = max(c[1] for c in coords) + bh
        controls = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)

        # HUD rect is union of legend and controls with padding
        left = min(legend.left, controls.left) - 8
        top = min(legend.top, controls.top) - 8
        right = max(legend.right, controls.right) + 8
        bottom = max(legend.bottom, controls.bottom) + 8
        hud = pygame.Rect(left, top, right-left, bottom-top)

        self.legend_rect_cache = legend
        self.controls_rect_cache = controls
        self.hud_rect_cache = hud

    # --------- Condition → Weather + Reward Shaping ---------
    def _apply_weather_overrides(self):
        w = self.sim.weather; c = self.conditions
        # Reset shaping to base LLM multipliers
        shaping = dict(self.base_shaping)

        if c["rainy"]:
            w.rain = 1.0; w.humidity = max(w.humidity, 0.78)
            w.temp = min(max(w.temp, 20.0), 32.0)
            w.wind_dx = random.uniform(-0.3, 0.3); w.wind_dy = random.uniform(-0.3, 0.3)
            shaping["irrigate_multiplier"] *= 0.75
            shaping["monitor_multiplier"] *= 1.05
            shaping["fungicide_multiplier"] *= 1.05
        if c["sunny"]:
            w.rain = 0.0; w.humidity = min(w.humidity, 0.5)
            w.temp = max(w.temp, 31.0)
            w.wind_dx = random.uniform(-0.2, 0.2); w.wind_dy = random.uniform(-0.2, 0.2)
            shaping["irrigate_multiplier"] *= 1.10
            shaping["monitor_multiplier"] *= 1.05
        if c["wind_storm"]:
            w.wind_dx = random.uniform(-1.0, 1.0); w.wind_dy = random.uniform(-1.0, 1.0)
            if not c["rainy"]:
                w.rain = 0.0; w.humidity = max(0.3, min(0.7, w.humidity))
            shaping["monitor_multiplier"] *= 1.15
            shaping["pesticide_multiplier"] *= 1.05
        if c["drought"]:
            w.rain = 0.0; w.humidity = min(w.humidity, 0.35)
            w.temp = max(w.temp, 33.0)
            w.wind_dx = random.uniform(-0.3, 0.3); w.wind_dy = random.uniform(-0.3, 0.3)
            shaping["irrigate_multiplier"] *= 1.30
            shaping["monitor_multiplier"] *= 1.05

        self.sim.llm_shaping.update(shaping)

    def _active_condition_messages(self):
        msgs=[]
        if self.conditions["rainy"]: msgs.append("Rainy — reduce irrigation; check drainage.")
        if self.conditions["sunny"]: msgs.append("Sunny — watch moisture; irrigate if dry.")
        if self.conditions["wind_storm"]: msgs.append("Wind storm — inspect lodging/damage; map hotspots.")
        if self.conditions["drought"]: msgs.append("Drought — irrigate more; schedule water smartly.")
        return msgs

    # --------------- Drawing helpers ---------------
    def draw_cell(self, x, y):
        c = self.sim.grid[x][y]; r = self.rects[x][y]
        soil = (120, 85, 60); soil = (soil[0], int(soil[1]*(0.8+0.4*c.moisture)), soil[2])
        pygame.draw.rect(self.screen, soil, r, border_radius=4)
        if not c.crop:
            if c.moisture > 0.7: pygame.draw.rect(self.screen, (110,160,110), r, 1, border_radius=4)
            return
        base = {"wheat": (205,190,100), "corn": (60,180,70), "soy": (70,160,120)}[c.crop]
        h = c.health()
        plant = (int(base[0]*(0.5+0.5*h)), int(base[1]*(0.6+0.5*h)), int(base[2]*(0.5+0.6*h)))
        cx, cy = r.center; stem_h = int(r.h * (0.2 + 0.7*c.growth))
        pygame.draw.line(self.screen, (50,120,50), (cx, r.bottom-3), (cx, r.bottom-3-stem_h), 3)
        leaf_span = int(6 + 10*c.growth)
        pygame.draw.line(self.screen, plant, (cx, r.bottom-8-int(0.3*stem_h)), (cx-leaf_span, r.bottom-10-int(0.5*stem_h)), 2)
        pygame.draw.line(self.screen, plant, (cx, r.bottom-8-int(0.5*stem_h)), (cx+leaf_span, r.bottom-9-int(0.8*stem_h)), 2)
        if c.pest > 0.2:
            for i in range(2):
                px = r.x + 4 + int((r.w-8)*(i*0.3 + 0.2))
                py = r.y + r.h - 5 - int(stem_h*0.5) + i*3
                gfxdraw.filled_circle(self.screen, px, py, 2, (190,40,40))
        if c.disease > 0.2:
            pygame.draw.circle(self.screen, (120,70,150), (r.x+int(0.3*r.w), r.y+int(0.6*r.h)), 3, 0)

    def draw_agents(self):
        for i, a in enumerate(self.sim.agents):
            r = self.rects[a.x][a.y]; cx, cy = r.center; radius = 7
            col = (40,100,230) if i%2==0 else (230,120,40)
            gfxdraw.filled_circle(self.screen, cx, cy, radius, col)
            pygame.draw.circle(self.screen, (20,20,20), (cx, cy), radius+max(1, int(4*a.battery)), 1)
            self.screen.blit(self.font_xs.render(str(i), True, (255,255,255)), (cx-4, cy-7))

    def draw_weather_overlay(self):
        if self.sim.weather.rain > 0:
            surf = pygame.Surface((self.grid_w, self.grid_h), pygame.SRCALPHA)
            surf.fill(self.COL_RAIN); self.screen.blit(surf, self.grid_origin)

    def _draw_legend(self, rect: pygame.Rect):
        # Legend box
        pygame.draw.rect(self.screen, (246,249,253), rect, border_radius=8)
        pygame.draw.rect(self.screen, self.COL_FRAME, rect, 1, border_radius=8)
        x = rect.x + 10; y = rect.y + 8
        self.screen.blit(self.font_sm.render("Legend", True, (33,66,120)), (x, y))
        y += 22
        # pest
        gfxdraw.filled_circle(self.screen, x+8, y+6, 4, (190,40,40))
        self.screen.blit(self.font_xs.render("Pest hotspot", True, self.COL_TEXT), (x+20, y))
        y += 18
        pygame.draw.circle(self.screen, (120,70,150), (x+8, y+6), 4, 0)
        self.screen.blit(self.font_xs.render("Disease patch", True, self.COL_TEXT), (x+20, y))
        y += 18
        pygame.draw.circle(self.screen, (20,20,20), (x+8, y+6), 8, 1)
        self.screen.blit(self.font_xs.render("Robot (battery ring)", True, self.COL_TEXT), (x+20, y))

    def _draw_hud(self):
        """Draw unified HUD box at bottom-left (legend + buttons)."""
        legend = self.legend_rect_cache
        controls = self.controls_rect_cache
        hud = self.hud_rect_cache
        if not (legend and controls and hud): return

        # HUD background
        pygame.draw.rect(self.screen, (245,248,252), hud, border_radius=10)
        pygame.draw.rect(self.screen, self.COL_FRAME, hud, 1, border_radius=10)

        # Legend
        self._draw_legend(legend)

        # Buttons (with tiny section header)
        header_y = controls.y - 22
        self.screen.blit(self.font_xs.render("Weather Controls", True, (70,90,110)),
                         (controls.x, max(hud.y+6, header_y)))
        for btn in self.buttons:
            btn.draw(self.screen, self.font_sm)

    def _section_title(self, text, y):
        self.screen.blit(self.font_md.render(text, True, (33,66,120)), (self.w-PANEL_W+16, y))
        pygame.draw.line(self.screen, self.COL_FRAME, (self.w-PANEL_W+14, y+22), (self.w-18, y+22), 1)
        return y + 30

    def _kv(self, k, v, x, y):
        self.screen.blit(self.font_sm.render(str(k)+":", True, self.COL_TEXT), (self.w-PANEL_W+x, y))
        self.screen.blit(self.font_sm.render(str(v), True, self.COL_TEXT), (self.w-PANEL_W+170, y))

    def _wrap(self, text, width, font):
        words, lines, cur = text.split(), [], ""
        for w in words:
            if font.size(cur + (" " if cur else "") + w)[0] <= width:
                cur = (cur + " " + w) if cur else w
            else:
                lines.append(cur); cur = w
        if cur: lines.append(cur)
        return lines

    def panel(self, llm_summary: list):
        panel = pygame.Rect(self.w-PANEL_W, 0, PANEL_W, self.h)
        pygame.draw.rect(self.screen, self.COL_PANEL, panel)
        pygame.draw.line(self.screen, self.COL_FRAME, (self.w-PANEL_W,0), (self.w-PANEL_W,self.h), 2)

        title = self.font_lg.render("Farm Dashboard", True, self.COL_TEXT)
        self.screen.blit(title, (self.w-PANEL_W+16, 12))

        # Weather readouts
        y = 54
        y = self._section_title("Weather", y)
        self._kv("Day", str(self.sim.day), 16, y); y+=20
        self._kv("Temp (°C)", f"{self.sim.weather.temp:.1f}", 16, y); y+=20
        self._kv("Humidity", f"{self.sim.weather.humidity:.2f}", 16, y); y+=20
        self._kv("Rain", "Yes" if self.sim.weather.rain>0 else "No", 16, y); y+=12

        # Speed
        y = self._section_title("Speed", y)
        self._kv("Move every", f"{self.move_every} tick(s)", 16, y); y+=20
        self._kv("FPS (visual)", f"{self.current_fps}", 16, y); y+=12
        self._kv("Paused", "Yes" if self.paused else "No", 16, y); y+=12
        hint = "Hotkeys: R Rainy, S Sunny, W Wind, D Drought | Space Pause, N Step | +/- Move | [/] FPS | P PNG | O JSON"
        self.screen.blit(self.font_xs.render(hint, True, (70,90,110)), (self.w-PANEL_W+16, y+6))
        y += 28

        # Metrics
        y = self._section_title("Metrics", y)
        sustain = self.sim.sustainability_index()
        self._kv("Yield (Σ)", f"{self.sim.total_yield:.2f}", 16, y); y+=20
        self._kv("Sustainability", f"{sustain:.2f}", 16, y); y+=20
        self._kv("Water Used", f"{self.sim.total_water_used:.0f}", 16, y); y+=20
        self._kv("Chemicals", f"{self.sim.total_chem_used:.0f}", 16, y); y+=8

        # Advisories (scrollable)
        y = self._section_title("Advisories", y+6)
        clip = pygame.Rect(self.w-PANEL_W+14, y, PANEL_W-28, 210)
        pygame.draw.rect(self.screen, (246,249,253), clip, border_radius=8)
        pygame.draw.rect(self.screen, self.COL_FRAME, clip, 1, border_radius=8)
        inner_x, inner_y = clip.x+10, clip.y+10 + self.advisory_scroll
        lines = []
        for s in llm_summary:
            lines.extend(self._wrap(s, clip.w-20, self.font_sm)); lines.append("")
        for line in lines:
            if clip.top <= inner_y <= clip.bottom-14:
                self.screen.blit(self.font_sm.render(line, True, (40,80,120)), (inner_x, inner_y))
            inner_y += self.advisory_line_h

        # Agents
        y = self._section_title("Agents", clip.bottom + 14)
        for i, a in enumerate(self.sim.agents[:6]):
            line = f"#{i} ({a.x},{a.y})  {a.last_action:>12}  r:{a.reward:.2f}  bat:{a.battery:.2f}"
            self.screen.blit(self.font_xs.render(line, True, self.COL_TEXT), (self.w-PANEL_W+16, y)); y += 18

    def render(self, llm_summary: list):
        self.screen.fill(self.COL_BG)
        pygame.draw.rect(self.screen, (220,228,238), (*self.grid_origin, self.grid_w, self.grid_h), 2, border_radius=10)

        # Grid
        for x in range(self.sim.w):
            for y in range(self.sim.h):
                self.draw_cell(x, y)
        self.draw_agents()
        self.draw_weather_overlay()

        # Bottom-left HUD (legend + buttons)
        self._draw_hud()

        # Right panel
        self.panel(llm_summary)

        pygame.display.flip()
        self.clock.tick(self.current_fps)

    def dynamic_summary(self, base_summary: List[str]) -> List[str]:
        msgs = list(base_summary); msgs.extend(self._active_condition_messages())
        seen, out = set(), []
        for m in msgs:
            if m not in seen: out.append(m); seen.add(m)
        return out

def simulate_and_render(sim: FarmSimulator, agents, base_llm_summary):
    viz = FarmViz(sim)
    running = True
    while running:
        # Update HUD layout first so event hitboxes are correct
        viz.position_hud()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.MOUSEWHEEL:
                viz.advisory_scroll += event.y * 12
                viz.advisory_scroll = max(-600, min(0, viz.advisory_scroll))
            elif event.type == pygame.KEYDOWN:
                k = event.key
                # Weather hotkeys
                if k == pygame.K_r: viz._toggle_condition("rainy", not viz.conditions["rainy"])
                elif k == pygame.K_s: viz._toggle_condition("sunny", not viz.conditions["sunny"])
                elif k == pygame.K_w: viz._toggle_condition("wind_storm", not viz.conditions["wind_storm"])
                elif k == pygame.K_d: viz._toggle_condition("drought", not viz.conditions["drought"])
                # Pause/Step
                elif k == pygame.K_SPACE: viz.paused = not viz.paused
                elif k == pygame.K_n and viz.paused:
                    viz._apply_weather_overrides(); sim.step()
                # Move speed (+ slower, - faster)
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    viz.move_every = min(60, viz.move_every + 1)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    viz.move_every = max(1, viz.move_every - 1)
                # FPS speed
                elif k == pygame.K_LEFTBRACKET:
                    viz.current_fps = max(1, viz.current_fps - 5)
                elif k == pygame.K_RIGHTBRACKET:
                    viz.current_fps = min(120, viz.current_fps + 5)
                # Screenshot & Snapshot
                elif k == pygame.K_p:
                    fname = f"screenshot_{int(time.time())}.png"
                    pygame.image.save(viz.screen, fname)
                    print(f"Saved screenshot: {fname}")
                elif k == pygame.K_o:
                    snap = {
                        "ticks": sim.ticks,
                        "day": sim.day,
                        "yield_sum": sim.total_yield,
                        "sustainability": sim.sustainability_index(),
                        "water_used": sim.total_water_used,
                        "chem_used": sim.total_chem_used,
                        "conditions": dict(viz.conditions),
                        "llm_shaping": dict(sim.llm_shaping),
                    }
                    fname = f"snapshot_{int(time.time())}.json"
                    with open(fname, "w") as f: json.dump(snap, f, indent=2)
                    print(f"Saved snapshot: {fname}")

            # Button clicks (legend-adjacent)
            for b in viz.buttons: b.handle(event)

        if viz.paused:
            summary = viz.dynamic_summary(base_llm_summary)
            viz.render(summary)
            continue

        # Apply condition effects each frame
        viz._apply_weather_overrides()

        # Agent loop (throttled by viz.move_every)
        if sim.ticks % viz.move_every == 0:
            for i, agent in enumerate(agents):
                action, (dx, dy) = agent.act(sim)
                move_agent(sim, i, dx, dy)
                sim.apply_action(i, action)
                a = sim.agents[i]
                if (a.x, a.y) in BASE_LOCATIONS and action == "idle":
                    a.battery = min(MAX_BATTERY, a.battery + BATTERY_RECHARGE_PER_TICK)

        # Environment step & render
        sim.step()
        summary = viz.dynamic_summary(base_llm_summary)
        viz.render(summary)

    pygame.quit()
