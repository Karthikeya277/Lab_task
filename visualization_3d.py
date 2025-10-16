# visualization_3d.py — 3D Farm Visualization with OpenGL
# Requirements: pip install PyOpenGL PyOpenGL_accelerate pygame numpy
# - Full 3D rendering of farm grid with height-based crops
# - Rotating camera view with mouse/keyboard controls
# - Weather effects in 3D space
# - Interactive controls and side panel
# - All features from 2D version preserved

import pygame
import random
import json
import time
import math
import numpy as np
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from typing import List
from config import *
from simulator import FarmSimulator
from rl_swarm import move_agent

# --- Defaults ---
MOVE_EVERY_N_TICKS_DEFAULT = 8

class Camera3D:
    def __init__(self):
        self.distance = 60.0
        self.angle_h = 45.0  # horizontal angle
        self.angle_v = 30.0  # vertical angle
        self.target = [0, 0, 0]
        self.mouse_sensitivity = 0.3
        self.zoom_speed = 2.0
        
    def apply(self):
        glLoadIdentity()
        # Calculate camera position
        rad_h = math.radians(self.angle_h)
        rad_v = math.radians(self.angle_v)
        
        x = self.distance * math.cos(rad_v) * math.cos(rad_h)
        y = self.distance * math.sin(rad_v)
        z = self.distance * math.cos(rad_v) * math.sin(rad_h)
        
        gluLookAt(x, y, z, 
                  self.target[0], self.target[1], self.target[2],
                  0, 1, 0)
    
    def rotate(self, dx, dy):
        self.angle_h += dx * self.mouse_sensitivity
        self.angle_v = max(-89, min(89, self.angle_v + dy * self.mouse_sensitivity))
    
    def zoom(self, delta):
        self.distance = max(20, min(150, self.distance - delta * self.zoom_speed))

class ToggleButton:
    def __init__(self, rect, label, get_state, set_state):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.get_state = get_state
        self.set_state = set_state
        
    def draw(self, surface, font):
        active = self.get_state()
        bg = (210, 235, 220) if active else (240, 244, 250)
        border = (60, 160, 90) if active else (200, 210, 225)
        fg = (22, 60, 30) if active else (35, 45, 58)
        pygame.draw.rect(surface, bg, self.rect, border_radius=10)
        pygame.draw.rect(surface, border, self.rect, 1, border_radius=10)
        text = font.render(self.label + ("  ●" if active else "  ○"), True, fg)
        surface.blit(text, (self.rect.x + (self.rect.w - text.get_width()) // 2,
                            self.rect.y + (self.rect.h - text.get_height()) // 2))
    
    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.set_state(not self.get_state())

class FarmViz3D:
    def __init__(self, sim: FarmSimulator):
        pygame.init()
        self.sim = sim
        self.w = WINDOW_W
        self.h = WINDOW_H
        
        # Create OpenGL window
        self.screen = pygame.display.set_mode((self.w, self.h), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Autonomous Agricultural Swarm - 3D View")
        
        # Setup OpenGL
        self.setup_opengl()
        
        # Camera
        self.camera = Camera3D()
        self.mouse_dragging = False
        self.last_mouse_pos = (0, 0)
        
        # Fonts and UI overlay surface
        self.font_xs = pygame.font.Font(FONT_NAME, 12)
        self.font_sm = pygame.font.Font(FONT_NAME, 14)
        self.font_md = pygame.font.Font(FONT_NAME, 18)
        self.font_lg = pygame.font.Font(FONT_NAME, 22)
        
        # Create 2D overlay surface for UI
        self.overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        
        # Colors
        self.COL_BG = (238, 243, 248)
        self.COL_PANEL = (252, 253, 255)
        self.COL_TEXT = (30, 40, 52)
        self.COL_FRAME = (210, 218, 230)
        
        # Weather conditions
        self.conditions = {"rainy": False, "sunny": False, "wind_storm": False, "drought": False}
        
        # UI controls
        self.buttons: List[ToggleButton] = []
        self._init_weather_toggles()
        
        # Advisories scroll
        self.advisory_scroll = 0
        self.advisory_line_h = 16
        
        # Sim control
        self.paused = False
        self.move_every = MOVE_EVERY_N_TICKS_DEFAULT
        self.current_fps = FPS
        
        # Reward shaping
        self.base_shaping = dict(self.sim.llm_shaping)
        
        self.clock = pygame.time.Clock()
    
    def setup_opengl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Lighting setup
        glLightfv(GL_LIGHT0, GL_POSITION, (20, 40, 20, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.7, 1))
        
        # Projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.w / self.h), 0.1, 500.0)
        glMatrixMode(GL_MODELVIEW)
        
        glClearColor(0.93, 0.95, 0.97, 1)
    
    def _init_weather_toggles(self):
        # Position buttons in bottom-left area
        x_start = 20
        y_start = self.h - 150
        btn_w, btn_h = 120, 32
        gap = 10
        
        positions = [
            (x_start, y_start),
            (x_start + btn_w + gap, y_start),
            (x_start, y_start + btn_h + gap),
            (x_start + btn_w + gap, y_start + btn_h + gap)
        ]
        
        labels = [("Rainy", "rainy"), ("Sunny", "sunny"), 
                  ("Wind Storm", "wind_storm"), ("Drought", "drought")]
        
        for (x, y), (label, key) in zip(positions, labels):
            self.buttons.append(ToggleButton(
                (x, y, btn_w, btn_h), label,
                get_state=lambda k=key: self.conditions[k],
                set_state=lambda v, k=key: self._toggle_condition(k, v)
            ))
    
    def _toggle_condition(self, key, val):
        self.conditions[key] = val
        if key == "rainy" and val:
            self.conditions["sunny"] = False
        if key == "sunny" and val:
            self.conditions["rainy"] = False
    
    def _apply_weather_overrides(self):
        w = self.sim.weather
        c = self.conditions
        shaping = dict(self.base_shaping)
        
        if c["rainy"]:
            w.rain = 1.0
            w.humidity = max(w.humidity, 0.78)
            w.temp = min(max(w.temp, 20.0), 32.0)
            w.wind_dx = random.uniform(-0.3, 0.3)
            w.wind_dy = random.uniform(-0.3, 0.3)
            shaping["irrigate_multiplier"] *= 0.75
            shaping["monitor_multiplier"] *= 1.05
            shaping["fungicide_multiplier"] *= 1.05
        if c["sunny"]:
            w.rain = 0.0
            w.humidity = min(w.humidity, 0.5)
            w.temp = max(w.temp, 31.0)
            shaping["irrigate_multiplier"] *= 1.10
        if c["wind_storm"]:
            w.wind_dx = random.uniform(-1.0, 1.0)
            w.wind_dy = random.uniform(-1.0, 1.0)
            shaping["monitor_multiplier"] *= 1.15
        if c["drought"]:
            w.rain = 0.0
            w.humidity = min(w.humidity, 0.35)
            w.temp = max(w.temp, 33.0)
            shaping["irrigate_multiplier"] *= 1.30
        
        self.sim.llm_shaping.update(shaping)
    
    def _active_condition_messages(self):
        msgs = []
        if self.conditions["rainy"]:
            msgs.append("Rainy — reduce irrigation; check drainage.")
        if self.conditions["sunny"]:
            msgs.append("Sunny — watch moisture; irrigate if dry.")
        if self.conditions["wind_storm"]:
            msgs.append("Wind storm — inspect lodging/damage.")
        if self.conditions["drought"]:
            msgs.append("Drought — irrigate more; schedule water smartly.")
        return msgs
    
    def draw_cube(self, size=1.0):
        """Draw a simple cube"""
        s = size / 2
        glBegin(GL_QUADS)
        # Front
        glNormal3f(0, 0, 1)
        glVertex3f(-s, -s, s)
        glVertex3f(s, -s, s)
        glVertex3f(s, s, s)
        glVertex3f(-s, s, s)
        # Back
        glNormal3f(0, 0, -1)
        glVertex3f(-s, -s, -s)
        glVertex3f(-s, s, -s)
        glVertex3f(s, s, -s)
        glVertex3f(s, -s, -s)
        # Top
        glNormal3f(0, 1, 0)
        glVertex3f(-s, s, -s)
        glVertex3f(-s, s, s)
        glVertex3f(s, s, s)
        glVertex3f(s, s, -s)
        # Bottom
        glNormal3f(0, -1, 0)
        glVertex3f(-s, -s, -s)
        glVertex3f(s, -s, -s)
        glVertex3f(s, -s, s)
        glVertex3f(-s, -s, s)
        # Right
        glNormal3f(1, 0, 0)
        glVertex3f(s, -s, -s)
        glVertex3f(s, s, -s)
        glVertex3f(s, s, s)
        glVertex3f(s, -s, s)
        # Left
        glNormal3f(-1, 0, 0)
        glVertex3f(-s, -s, -s)
        glVertex3f(-s, -s, s)
        glVertex3f(-s, s, s)
        glVertex3f(-s, s, -s)
        glEnd()
    
    def draw_sphere(self, radius=0.5, slices=12, stacks=12):
        """Draw a sphere using GLU"""
        quad = gluNewQuadric()
        gluSphere(quad, radius, slices, stacks)
        gluDeleteQuadric(quad)
    
    def draw_cylinder(self, base=0.1, top=0.1, height=1.0, slices=12):
        """Draw a cylinder"""
        quad = gluNewQuadric()
        gluCylinder(quad, base, top, height, slices, 4)
        gluDeleteQuadric(quad)
    
    def draw_torus(self, inner_radius, outer_radius, sides=16, rings=16):
        """Draw a torus (donut shape) without GLUT"""
        for i in range(rings):
            glBegin(GL_QUAD_STRIP)
            for j in range(sides + 1):
                for k in range(2):
                    s = (i + k) % rings + 0.5
                    t = j % sides
                    angle1 = s * 2.0 * math.pi / rings
                    angle2 = t * 2.0 * math.pi / sides
                    x = (outer_radius + inner_radius * math.cos(angle2)) * math.cos(angle1)
                    y = (outer_radius + inner_radius * math.cos(angle2)) * math.sin(angle1)
                    z = inner_radius * math.sin(angle2)
                    nx = math.cos(angle2) * math.cos(angle1)
                    ny = math.cos(angle2) * math.sin(angle1)
                    nz = math.sin(angle2)
                    glNormal3f(nx, ny, nz)
                    glVertex3f(x, y, z)
            glEnd()
    
    def draw_ground(self):
        """Draw ground plane"""
        size = self.sim.w * 1.5
        glColor3f(0.42, 0.35, 0.28)
        glBegin(GL_QUADS)
        glNormal3f(0, 1, 0)
        glVertex3f(-size, -0.1, -size)
        glVertex3f(size, -0.1, -size)
        glVertex3f(size, -0.1, size)
        glVertex3f(-size, -0.1, size)
        glEnd()
    
    def draw_cell_3d(self, x, y):
        """Draw a single farm cell in 3D"""
        c = self.sim.grid[x][y]
        
        # Convert grid coords to world coords
        wx = (x - self.sim.w / 2) * 2.0
        wz = (y - self.sim.h / 2) * 2.0
        
        glPushMatrix()
        glTranslatef(wx, 0, wz)
        
        # Soil block
        soil_brightness = 0.8 + 0.4 * c.moisture
        glColor3f(0.47 * soil_brightness, 0.33 * soil_brightness, 0.24 * soil_brightness)
        glPushMatrix()
        glTranslatef(0, -0.05, 0)
        glScalef(1.8, 0.1, 1.8)
        self.draw_cube()
        glPopMatrix()
        
        # Draw crop if exists
        if c.crop:
            h = c.health()
            growth = c.growth
            
            # Stem
            glColor3f(0.2, 0.47, 0.2)
            glPushMatrix()
            stem_height = 0.5 + growth * 2.5
            glRotatef(-90, 1, 0, 0)
            self.draw_cylinder(0.05, 0.03, stem_height, 8)
            glPopMatrix()
            
            # Crop head/leaves
            crop_colors = {
                "wheat": (0.80, 0.75, 0.39),
                "corn": (0.24, 0.71, 0.27),
                "soy": (0.27, 0.63, 0.47)
            }
            base_color = crop_colors.get(c.crop, (0.31, 0.69, 0.31))
            glColor3f(base_color[0] * (0.6 + 0.4 * h),
                     base_color[1] * (0.6 + 0.4 * h),
                     base_color[2] * (0.6 + 0.4 * h))
            
            # Draw leaves as spheres
            for i in range(3):
                angle = (i / 3) * 2 * math.pi
                offset = 0.15 + growth * 0.2
                glPushMatrix()
                glTranslatef(math.cos(angle) * offset, 
                           stem_height * (0.5 + i * 0.15),
                           math.sin(angle) * offset)
                self.draw_sphere(0.15 + growth * 0.15, 10, 10)
                glPopMatrix()
            
            # Pest indicator
            if c.pest > 0.2:
                glColor3f(0.75, 0.16, 0.16)
                glPushMatrix()
                glTranslatef(0.2, stem_height * 0.7, 0)
                self.draw_sphere(0.08, 8, 8)
                glPopMatrix()
            
            # Disease indicator
            if c.disease > 0.2:
                glColor3f(0.47, 0.27, 0.59)
                glPushMatrix()
                glTranslatef(-0.2, stem_height * 0.6, 0)
                self.draw_sphere(0.1, 8, 8)
                glPopMatrix()
        
        glPopMatrix()
    
    def draw_agent_3d(self, agent, index):
        """Draw an agent (robot) in 3D"""
        wx = (agent.x - self.sim.w / 2) * 2.0
        wz = (agent.y - self.sim.h / 2) * 2.0
        
        glPushMatrix()
        glTranslatef(wx, 0.8, wz)
        
        # Robot body
        col = (0.16, 0.39, 0.90) if index % 2 == 0 else (0.90, 0.47, 0.16)
        glColor3f(*col)
        self.draw_sphere(0.4, 16, 16)
        
        # Battery ring using custom torus
        battery_color = (0, agent.battery, 0)
        glColor3f(*battery_color)
        glPushMatrix()
        glRotatef(90, 1, 0, 0)
        self.draw_torus(0.08, 0.5, sides=16, rings=16)
        glPopMatrix()
        
        # ID marker (small cube on top)
        glColor3f(1, 1, 1)
        glPushMatrix()
        glTranslatef(0, 0.6, 0)
        glScalef(0.2, 0.2, 0.2)
        self.draw_cube()
        glPopMatrix()
        
        glPopMatrix()
    
    def draw_rain_effect(self):
        """Draw rain particles if raining"""
        if self.sim.weather.rain > 0:
            glDisable(GL_LIGHTING)
            glColor4f(0.47, 0.63, 1.0, 0.3)
            glBegin(GL_LINES)
            for _ in range(100):
                x = random.uniform(-self.sim.w, self.sim.w)
                z = random.uniform(-self.sim.h, self.sim.h)
                y_top = random.uniform(5, 15)
                glVertex3f(x, y_top, z)
                glVertex3f(x, y_top - 2, z)
            glEnd()
            glEnable(GL_LIGHTING)
    
    def render_3d_scene(self):
        """Render the 3D OpenGL scene"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Apply camera
        self.camera.apply()
        
        # Draw ground
        self.draw_ground()
        
        # Draw grid cells
        for x in range(self.sim.w):
            for y in range(self.sim.h):
                self.draw_cell_3d(x, y)
        
        # Draw agents
        for i, agent in enumerate(self.sim.agents):
            self.draw_agent_3d(agent, i)
        
        # Draw weather effects
        self.draw_rain_effect()
    
    def render_2d_overlay(self, llm_summary: list):
        """Render 2D UI overlay"""
        self.overlay.fill((0, 0, 0, 0))
        
        # Draw weather control buttons
        for btn in self.buttons:
            btn.draw(self.overlay, self.font_sm)
        
        # Draw control hints
        hint_y = self.h - 200
        hints = [
            "Camera: Drag mouse to rotate | Scroll to zoom",
            "Hotkeys: R Rainy | S Sunny | W Wind | D Drought",
            "Space Pause | N Step | +/- Speed | P Screenshot"
        ]
        for i, hint in enumerate(hints):
            text = self.font_xs.render(hint, True, (70, 90, 110))
            self.overlay.blit(text, (20, hint_y + i * 16))
        
        # Draw side panel
        self.draw_side_panel(llm_summary)
        
        # Convert overlay to OpenGL texture and draw
        texture_data = pygame.image.tostring(self.overlay, "RGBA", True)
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.w, 0, self.h, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        
        glRasterPos2i(0, 0)
        glDrawPixels(self.w, self.h, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
    
    def draw_side_panel(self, llm_summary: list):
        """Draw the side information panel"""
        panel = pygame.Rect(self.w - PANEL_W, 0, PANEL_W, self.h)
        pygame.draw.rect(self.overlay, (*self.COL_PANEL, 240), panel)
        pygame.draw.line(self.overlay, self.COL_FRAME, 
                        (self.w - PANEL_W, 0), (self.w - PANEL_W, self.h), 2)
        
        title = self.font_lg.render("Farm Dashboard", True, self.COL_TEXT)
        self.overlay.blit(title, (self.w - PANEL_W + 16, 12))
        
        y = 50
        
        # Weather
        y = self._section_title("Weather", y)
        self._kv("Day", str(self.sim.day), 16, y); y += 20
        self._kv("Temp (°C)", f"{self.sim.weather.temp:.1f}", 16, y); y += 20
        self._kv("Humidity", f"{self.sim.weather.humidity:.2f}", 16, y); y += 20
        self._kv("Rain", "Yes" if self.sim.weather.rain > 0 else "No", 16, y); y += 20
        
        # Control
        y = self._section_title("Control", y)
        self._kv("Move every", f"{self.move_every} tick(s)", 16, y); y += 20
        self._kv("FPS", f"{self.current_fps}", 16, y); y += 20
        self._kv("Paused", "Yes" if self.paused else "No", 16, y); y += 20
        
        # Metrics
        y = self._section_title("Metrics", y)
        sustain = self.sim.sustainability_index()
        self._kv("Yield (Σ)", f"{self.sim.total_yield:.2f}", 16, y); y += 20
        self._kv("Sustainability", f"{sustain:.2f}", 16, y); y += 20
        self._kv("Water Used", f"{self.sim.total_water_used:.0f}", 16, y); y += 20
        self._kv("Chemicals", f"{self.sim.total_chem_used:.0f}", 16, y); y += 20
        
        # Agents
        y = self._section_title("Agents", y)
        for i, a in enumerate(self.sim.agents[:6]):
            line = f"#{i} ({a.x},{a.y}) {a.last_action[:8]} b:{a.battery:.2f}"
            text = self.font_xs.render(line, True, self.COL_TEXT)
            self.overlay.blit(text, (self.w - PANEL_W + 16, y))
            y += 18
    
    def _section_title(self, text, y):
        title = self.font_md.render(text, True, (33, 66, 120))
        self.overlay.blit(title, (self.w - PANEL_W + 16, y))
        pygame.draw.line(self.overlay, self.COL_FRAME,
                        (self.w - PANEL_W + 14, y + 22),
                        (self.w - 18, y + 22), 1)
        return y + 30
    
    def _kv(self, k, v, x, y):
        key_text = self.font_sm.render(str(k) + ":", True, self.COL_TEXT)
        val_text = self.font_sm.render(str(v), True, self.COL_TEXT)
        self.overlay.blit(key_text, (self.w - PANEL_W + x, y))
        self.overlay.blit(val_text, (self.w - PANEL_W + 170, y))
    
    def render(self, llm_summary: list):
        """Main render function"""
        self.render_3d_scene()
        self.render_2d_overlay(llm_summary)
        pygame.display.flip()
        self.clock.tick(self.current_fps)
    
    def dynamic_summary(self, base_summary: List[str]) -> List[str]:
        msgs = list(base_summary)
        msgs.extend(self._active_condition_messages())
        seen, out = set(), []
        for m in msgs:
            if m not in seen:
                out.append(m)
                seen.add(m)
        return out

def simulate_and_render_3d(sim: FarmSimulator, agents, base_llm_summary):
    """Main simulation loop with 3D rendering"""
    viz = FarmViz3D(sim)
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Check UI buttons first
                    handled = False
                    for btn in viz.buttons:
                        btn.handle(event)
                        if btn.rect.collidepoint(event.pos):
                            handled = True
                    if not handled:
                        viz.mouse_dragging = True
                        viz.last_mouse_pos = event.pos
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    viz.mouse_dragging = False
            
            elif event.type == pygame.MOUSEMOTION:
                if viz.mouse_dragging:
                    dx = event.pos[0] - viz.last_mouse_pos[0]
                    dy = event.pos[1] - viz.last_mouse_pos[1]
                    viz.camera.rotate(dx, dy)
                    viz.last_mouse_pos = event.pos
            
            elif event.type == pygame.MOUSEWHEEL:
                viz.camera.zoom(event.y)
            
            elif event.type == pygame.KEYDOWN:
                k = event.key
                # Weather hotkeys
                if k == pygame.K_r:
                    viz._toggle_condition("rainy", not viz.conditions["rainy"])
                elif k == pygame.K_s:
                    viz._toggle_condition("sunny", not viz.conditions["sunny"])
                elif k == pygame.K_w:
                    viz._toggle_condition("wind_storm", not viz.conditions["wind_storm"])
                elif k == pygame.K_d:
                    viz._toggle_condition("drought", not viz.conditions["drought"])
                # Pause/Step
                elif k == pygame.K_SPACE:
                    viz.paused = not viz.paused
                elif k == pygame.K_n and viz.paused:
                    viz._apply_weather_overrides()
                    sim.step()
                # Speed controls
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    viz.move_every = min(60, viz.move_every + 1)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    viz.move_every = max(1, viz.move_every - 1)
                elif k == pygame.K_LEFTBRACKET:
                    viz.current_fps = max(1, viz.current_fps - 5)
                elif k == pygame.K_RIGHTBRACKET:
                    viz.current_fps = min(120, viz.current_fps + 5)
                # Screenshot
                elif k == pygame.K_p:
                    fname = f"screenshot_3d_{int(time.time())}.png"
                    pygame.image.save(viz.screen, fname)
                    print(f"Saved screenshot: {fname}")
                # Snapshot
                elif k == pygame.K_o:
                    snap = {
                        "ticks": sim.ticks,
                        "day": sim.day,
                        "yield_sum": sim.total_yield,
                        "sustainability": sim.sustainability_index(),
                        "water_used": sim.total_water_used,
                        "chem_used": sim.total_chem_used,
                        "conditions": dict(viz.conditions),
                        "camera": {
                            "distance": viz.camera.distance,
                            "angle_h": viz.camera.angle_h,
                            "angle_v": viz.camera.angle_v
                        }
                    }
                    fname = f"snapshot_3d_{int(time.time())}.json"
                    with open(fname, "w") as f:
                        json.dump(snap, f, indent=2)
                    print(f"Saved snapshot: {fname}")
        
        if viz.paused:
            summary = viz.dynamic_summary(base_llm_summary)
            viz.render(summary)
            continue
        
        # Apply weather effects
        viz._apply_weather_overrides()
        
        # Agent loop (throttled by move_every)
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
