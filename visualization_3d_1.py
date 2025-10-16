# visualization_3d_enhanced.py — Enhanced 3D Farm Visualization
# Major improvements:
# - Grid lines for better cell definition
# - Agent action trails/indicators
# - Clearer crop visualization with height variation
# - Better color contrast and lighting
# - Agent target indicators
# - Action labels floating above agents
# - Improved camera positioning

import pygame
import random
import json
import time
import math
import numpy as np
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from typing import List, Tuple
from collections import deque
from config import *
from simulator import FarmSimulator
from rl_swarm import move_agent

MOVE_EVERY_N_TICKS_DEFAULT = 8

class Camera3D:
    def __init__(self):
        self.distance = 45.0  # Closer view
        self.angle_h = 45.0
        self.angle_v = 35.0  # Better angle
        self.target = [0, 0, 0]
        self.mouse_sensitivity = 0.3
        self.zoom_speed = 2.0
        
    def apply(self):
        glLoadIdentity()
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

class AgentTrail:
    """Track agent movement trail"""
    def __init__(self, max_length=15):
        self.positions = deque(maxlen=max_length)
        self.actions = deque(maxlen=max_length)
    
    def add(self, x, y, action):
        self.positions.append((x, y))
        self.actions.append(action)

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
        
        self.screen = pygame.display.set_mode((self.w, self.h), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Autonomous Agricultural Swarm - 3D View")
        
        self.setup_opengl()
        
        self.camera = Camera3D()
        self.mouse_dragging = False
        self.last_mouse_pos = (0, 0)
        
        # Agent trails
        self.agent_trails = [AgentTrail() for _ in sim.agents]
        
        # Visual options
        self.show_grid = True
        self.show_trails = True
        self.show_action_labels = True
        
        # Fonts
        self.font_xs = pygame.font.Font(FONT_NAME, 11)
        self.font_sm = pygame.font.Font(FONT_NAME, 13)
        self.font_md = pygame.font.Font(FONT_NAME, 16)
        self.font_lg = pygame.font.Font(FONT_NAME, 20)
        
        self.overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        
        self.COL_BG = (238, 243, 248)
        self.COL_PANEL = (252, 253, 255)
        self.COL_TEXT = (30, 40, 52)
        self.COL_FRAME = (210, 218, 230)
        
        self.conditions = {"rainy": False, "sunny": False, "wind_storm": False, "drought": False}
        
        self.buttons: List[ToggleButton] = []
        self._init_weather_toggles()
        
        self.paused = False
        self.move_every = MOVE_EVERY_N_TICKS_DEFAULT
        self.current_fps = FPS
        
        self.base_shaping = dict(self.sim.llm_shaping)
        
        self.clock = pygame.time.Clock()
    
    def setup_opengl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Main light (sun)
        glLightfv(GL_LIGHT0, GL_POSITION, (30, 50, 30, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.4, 0.4, 0.4, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 0.95, 0.85, 1))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (0.5, 0.5, 0.5, 1))
        
        # Fill light
        glLightfv(GL_LIGHT1, GL_POSITION, (-20, 30, -20, 1))
        glLightfv(GL_LIGHT1, GL_AMBIENT, (0.2, 0.2, 0.25, 1))
        glLightfv(GL_LIGHT1, GL_DIFFUSE, (0.3, 0.3, 0.4, 1))
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(50, (self.w / self.h), 0.1, 500.0)
        glMatrixMode(GL_MODELVIEW)
        
        # Sky blue background
        glClearColor(0.68, 0.85, 0.95, 1)
    
    def _init_weather_toggles(self):
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
    
    def draw_sphere(self, radius=0.5, slices=16, stacks=16):
        quad = gluNewQuadric()
        gluSphere(quad, radius, slices, stacks)
        gluDeleteQuadric(quad)
    
    def draw_cylinder(self, base=0.1, top=0.1, height=1.0, slices=16):
        quad = gluNewQuadric()
        gluCylinder(quad, base, top, height, slices, 4)
        gluDeleteQuadric(quad)
    
    def draw_ground(self):
        """Enhanced ground with grid pattern"""
        size = self.sim.w * 1.2
        
        # Base ground
        glColor3f(0.35, 0.28, 0.22)
        glBegin(GL_QUADS)
        glNormal3f(0, 1, 0)
        glVertex3f(-size, -0.15, -size)
        glVertex3f(size, -0.15, -size)
        glVertex3f(size, -0.15, size)
        glVertex3f(-size, -0.15, size)
        glEnd()
    
    def draw_grid_lines(self):
        """Draw clear grid lines to separate cells"""
        if not self.show_grid:
            return
            
        glDisable(GL_LIGHTING)
        glLineWidth(1.5)
        glColor4f(0.25, 0.20, 0.16, 0.6)
        
        # Vertical lines
        for x in range(self.sim.w + 1):
            wx = (x - self.sim.w / 2) * 2.0 - 0.9
            glBegin(GL_LINES)
            glVertex3f(wx, 0.01, -(self.sim.h / 2) * 2.0 - 0.9)
            glVertex3f(wx, 0.01, (self.sim.h / 2) * 2.0 + 0.9)
            glEnd()
        
        # Horizontal lines
        for y in range(self.sim.h + 1):
            wz = (y - self.sim.h / 2) * 2.0 - 0.9
            glBegin(GL_LINES)
            glVertex3f(-(self.sim.w / 2) * 2.0 - 0.9, 0.01, wz)
            glVertex3f((self.sim.w / 2) * 2.0 + 0.9, 0.01, wz)
            glEnd()
        
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)
    
    def draw_cell_3d(self, x, y):
        """Enhanced cell visualization"""
        c = self.sim.grid[x][y]
        
        wx = (x - self.sim.w / 2) * 2.0
        wz = (y - self.sim.h / 2) * 2.0
        
        glPushMatrix()
        glTranslatef(wx, 0, wz)
        
        # Soil with moisture indication
        moisture_factor = 0.5 + 0.5 * c.moisture
        soil_r = 0.52 * moisture_factor
        soil_g = 0.40 * moisture_factor
        soil_b = 0.28 * moisture_factor
        glColor3f(soil_r, soil_g, soil_b)
        
        glPushMatrix()
        glTranslatef(0, -0.05, 0)
        glScalef(1.85, 0.12, 1.85)
        self.draw_cube()
        glPopMatrix()
        
        # Draw crop
        if c.crop:
            h = c.health()
            growth = c.growth
            
            # Stem - thicker and more visible
            glColor3f(0.15, 0.55, 0.15)
            glPushMatrix()
            stem_height = 1.0 + growth * 3.0
            glRotatef(-90, 1, 0, 0)
            self.draw_cylinder(0.08, 0.04, stem_height, 12)
            glPopMatrix()
            
            # Crop type specific appearance
            if c.crop == "wheat":
                self._draw_wheat(stem_height, h, growth)
            elif c.crop == "corn":
                self._draw_corn(stem_height, h, growth)
            elif c.crop == "soy":
                self._draw_soy(stem_height, h, growth)
            
            # Health indicator bar
            self._draw_health_bar(wx, wz, stem_height + 0.8, h)
            
            # Pest/disease indicators (larger and clearer)
            if c.pest > 0.2:
                glColor3f(0.9, 0.2, 0.2)
                glPushMatrix()
                glTranslatef(0.3, stem_height * 0.7, 0)
                self.draw_sphere(0.12, 12, 12)
                glPopMatrix()
            
            if c.disease > 0.2:
                glColor3f(0.6, 0.2, 0.8)
                glPushMatrix()
                glTranslatef(-0.3, stem_height * 0.6, 0)
                self.draw_sphere(0.12, 12, 12)
                glPopMatrix()
        
        glPopMatrix()
    
    def _draw_wheat(self, stem_height, health, growth):
        """Draw wheat-specific appearance"""
        glColor3f(0.88 * (0.5 + 0.5 * health), 
                 0.82 * (0.5 + 0.5 * health), 
                 0.42 * (0.5 + 0.5 * health))
        
        # Wheat head
        glPushMatrix()
        glTranslatef(0, stem_height, 0)
        glScalef(0.15, 0.4 + growth * 0.3, 0.15)
        self.draw_cube()
        glPopMatrix()
    
    def _draw_corn(self, stem_height, health, growth):
        """Draw corn-specific appearance"""
        base_col = (0.3, 0.75, 0.3)
        glColor3f(base_col[0] * (0.5 + 0.5 * health),
                 base_col[1] * (0.5 + 0.5 * health),
                 base_col[2] * (0.5 + 0.5 * health))
        
        # Corn leaves
        for i in range(4):
            angle = (i / 4) * 2 * math.pi
            glPushMatrix()
            glTranslatef(math.cos(angle) * 0.25, 
                        stem_height * (0.4 + i * 0.15),
                        math.sin(angle) * 0.25)
            glRotatef(45, math.cos(angle), 0, math.sin(angle))
            glScalef(0.4, 0.15, 0.1)
            self.draw_cube()
            glPopMatrix()
        
        # Corn ear
        glColor3f(0.9, 0.85, 0.3)
        glPushMatrix()
        glTranslatef(0.2, stem_height * 0.7, 0)
        glRotatef(45, 0, 0, 1)
        glScalef(0.15, 0.35, 0.15)
        self.draw_cube()
        glPopMatrix()
    
    def _draw_soy(self, stem_height, health, growth):
        """Draw soy-specific appearance"""
        base_col = (0.3, 0.68, 0.52)
        glColor3f(base_col[0] * (0.5 + 0.5 * health),
                 base_col[1] * (0.5 + 0.5 * health),
                 base_col[2] * (0.5 + 0.5 * health))
        
        # Soy leaves clusters
        for i in range(5):
            angle = (i / 5) * 2 * math.pi
            offset = 0.18 + growth * 0.15
            glPushMatrix()
            glTranslatef(math.cos(angle) * offset, 
                        stem_height * (0.3 + i * 0.12),
                        math.sin(angle) * offset)
            self.draw_sphere(0.18, 12, 12)
            glPopMatrix()
    
    def _draw_health_bar(self, wx, wz, height, health):
        """Draw floating health bar above crop"""
        glDisable(GL_LIGHTING)
        
        # Background bar
        glColor3f(0.3, 0.3, 0.3)
        glPushMatrix()
        glTranslatef(0, height, 0)
        glRotatef(self.camera.angle_h, 0, 1, 0)
        glScalef(0.6, 0.08, 0.02)
        self.draw_cube()
        glPopMatrix()
        
        # Health bar (red to green)
        if health < 0.5:
            glColor3f(1.0, health * 2, 0)
        else:
            glColor3f(2.0 * (1.0 - health), 1.0, 0)
        
        glPushMatrix()
        glTranslatef(-0.3 * (1 - health), height, 0)
        glRotatef(self.camera.angle_h, 0, 1, 0)
        glScalef(0.6 * health, 0.08, 0.03)
        self.draw_cube()
        glPopMatrix()
        
        glEnable(GL_LIGHTING)
    
    def draw_agent_3d(self, agent, index):
        """Enhanced agent visualization"""
        wx = (agent.x - self.sim.w / 2) * 2.0
        wz = (agent.y - self.sim.h / 2) * 2.0
        
        glPushMatrix()
        glTranslatef(wx, 1.2, wz)
        
        # Robot body - larger and clearer
        if index % 3 == 0:
            col = (0.2, 0.5, 1.0)
        elif index % 3 == 1:
            col = (1.0, 0.6, 0.2)
        else:
            col = (0.3, 0.9, 0.4)
        
        glColor3f(*col)
        self.draw_sphere(0.5, 20, 20)
        
        # Battery indicator ring
        battery_pct = agent.battery
        if battery_pct > 0.6:
            batt_col = (0.2, 0.9, 0.2)
        elif battery_pct > 0.3:
            batt_col = (0.9, 0.9, 0.2)
        else:
            batt_col = (0.9, 0.2, 0.2)
        
        glDisable(GL_LIGHTING)
        glColor3f(*batt_col)
        glLineWidth(4)
        glBegin(GL_LINE_LOOP)
        for i in range(32):
            if i / 32.0 <= battery_pct:
                angle = i * 2 * math.pi / 32
                x = 0.6 * math.cos(angle)
                z = 0.6 * math.sin(angle)
                glVertex3f(x, 0, z)
        glEnd()
        glLineWidth(1)
        glEnable(GL_LIGHTING)
        
        # Agent ID marker
        glColor3f(1, 1, 1)
        glPushMatrix()
        glTranslatef(0, 0.7, 0)
        glScalef(0.25, 0.25, 0.25)
        self.draw_cube()
        glPopMatrix()
        
        # Action indicator - beam showing current action
        self._draw_action_indicator(agent, wx, wz)
        
        glPopMatrix()
    
    def _draw_action_indicator(self, agent, wx, wz):
        """Draw visual indicator of agent's current action"""
        action = agent.last_action
        
        glDisable(GL_LIGHTING)
        glLineWidth(3)
        
        if action == "irrigate":
            # Blue beam downward
            glColor4f(0.2, 0.5, 1.0, 0.7)
            glBegin(GL_LINES)
            glVertex3f(0, 0, 0)
            glVertex3f(0, -1.0, 0)
            glEnd()
        elif action == "fertilize":
            # Green spray
            glColor4f(0.3, 0.9, 0.3, 0.6)
            for i in range(8):
                angle = i * 2 * math.pi / 8
                glBegin(GL_LINES)
                glVertex3f(0, 0, 0)
                glVertex3f(math.cos(angle) * 0.5, -0.8, math.sin(angle) * 0.5)
                glEnd()
        elif action == "pesticide":
            # Red spray
            glColor4f(1.0, 0.3, 0.3, 0.6)
            for i in range(8):
                angle = i * 2 * math.pi / 8
                glBegin(GL_LINES)
                glVertex3f(0, 0, 0)
                glVertex3f(math.cos(angle) * 0.5, -0.8, math.sin(angle) * 0.5)
                glEnd()
        elif action == "fungicide":
            # Purple spray
            glColor4f(0.8, 0.3, 0.8, 0.6)
            for i in range(8):
                angle = i * 2 * math.pi / 8
                glBegin(GL_LINES)
                glVertex3f(0, 0, 0)
                glVertex3f(math.cos(angle) * 0.5, -0.8, math.sin(angle) * 0.5)
                glEnd()
        elif action == "monitor":
            # Yellow scan beam
            glColor4f(1.0, 1.0, 0.3, 0.7)
            glBegin(GL_LINE_LOOP)
            for i in range(16):
                angle = i * 2 * math.pi / 16
                glVertex3f(math.cos(angle) * 0.6, -0.5, math.sin(angle) * 0.6)
            glEnd()
        
        glLineWidth(1)
        glEnable(GL_LIGHTING)
    
    def draw_agent_trails(self):
        """Draw movement trails for agents"""
        if not self.show_trails:
            return
        
        glDisable(GL_LIGHTING)
        glLineWidth(2)
        
        for i, trail in enumerate(self.agent_trails):
            if len(trail.positions) < 2:
                continue
            
            # Trail color matches agent
            if i % 3 == 0:
                col = (0.2, 0.5, 1.0)
            elif i % 3 == 1:
                col = (1.0, 0.6, 0.2)
            else:
                col = (0.3, 0.9, 0.4)
            
            glBegin(GL_LINE_STRIP)
            for j, (x, y) in enumerate(trail.positions):
                alpha = (j + 1) / len(trail.positions)
                glColor4f(col[0], col[1], col[2], alpha * 0.5)
                wx = (x - self.sim.w / 2) * 2.0
                wz = (y - self.sim.h / 2) * 2.0
                glVertex3f(wx, 0.8, wz)
            glEnd()
        
        glLineWidth(1)
        glEnable(GL_LIGHTING)
    
    def draw_rain_effect(self):
        """Enhanced rain effect"""
        if self.sim.weather.rain > 0:
            glDisable(GL_LIGHTING)
            glColor4f(0.5, 0.7, 1.0, 0.4)
            glLineWidth(2)
            glBegin(GL_LINES)
            for _ in range(150):
                x = random.uniform(-self.sim.w * 1.5, self.sim.w * 1.5)
                z = random.uniform(-self.sim.h * 1.5, self.sim.h * 1.5)
                y_top = random.uniform(8, 20)
                glVertex3f(x, y_top, z)
                glVertex3f(x - 0.2, y_top - 2.5, z - 0.2)
            glEnd()
            glLineWidth(1)
            glEnable(GL_LIGHTING)
    
    def render_3d_scene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        self.camera.apply()
        
        self.draw_ground()
        self.draw_grid_lines()
        
        # Draw cells
        for x in range(self.sim.w):
            for y in range(self.sim.h):
                self.draw_cell_3d(x, y)
        
        # Draw agent trails
        self.draw_agent_trails()
        
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
        
        # Enhanced control hints with background
        hint_bg = pygame.Rect(15, self.h - 235, 500, 75)
        pygame.draw.rect(self.overlay, (255, 255, 255, 200), hint_bg, border_radius=8)
        pygame.draw.rect(self.overlay, (100, 120, 140), hint_bg, 2, border_radius=8)
        
        hint_y = self.h - 225
        title = self.font_sm.render("Controls", True, (40, 60, 80))
        self.overlay.blit(title, (25, hint_y))
        hint_y += 22
        
        hints = [
            "🖱️  Drag to rotate | Scroll to zoom",
            "⌨️  R/S/W/D Weather | Space Pause | +/- Speed"
        ]
        for i, hint in enumerate(hints):
            text = self.font_xs.render(hint, True, (60, 80, 100))
            self.overlay.blit(text, (25, hint_y + i * 16))
        
        # Draw legend in bottom left
        self._draw_legend()
        
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
    
    def _draw_legend(self):
        """Draw legend explaining visual elements"""
        legend_x = 20
        legend_y = 50
        
        legend_bg = pygame.Rect(legend_x - 5, legend_y - 5, 180, 200)
        pygame.draw.rect(self.overlay, (255, 255, 255, 220), legend_bg, border_radius=8)
        pygame.draw.rect(self.overlay, (100, 120, 140), legend_bg, 2, border_radius=8)
        
        title = self.font_md.render("Legend", True, (40, 60, 80))
        self.overlay.blit(title, (legend_x + 5, legend_y))
        
        items = [
            ("● Blue Robot", (51, 128, 255)),
            ("● Orange Robot", (255, 153, 51)),
            ("● Green Robot", (77, 230, 102)),
            ("🌾 Wheat (Golden)", (204, 191, 100)),
            ("🌽 Corn (Green)", (61, 181, 69)),
            ("🫘 Soy (Teal)", (69, 173, 133)),
            ("🔴 Pest", (230, 51, 51)),
            ("🟣 Disease", (153, 69, 204)),
        ]
        
        y = legend_y + 30
        for label, color in items:
            if "Robot" in label:
                pygame.draw.circle(self.overlay, color, (legend_x + 10, y + 7), 6)
                text = self.font_xs.render(label[2:], True, (50, 50, 50))
                self.overlay.blit(text, (legend_x + 22, y + 2))
            else:
                text = self.font_xs.render(label, True, (50, 50, 50))
                self.overlay.blit(text, (legend_x + 5, y + 2))
            y += 20
    
    def draw_side_panel(self, llm_summary: list):
        """Enhanced side information panel"""
        panel = pygame.Rect(self.w - PANEL_W, 0, PANEL_W, self.h)
        pygame.draw.rect(self.overlay, (*self.COL_PANEL, 245), panel)
        pygame.draw.line(self.overlay, self.COL_FRAME, 
                        (self.w - PANEL_W, 0), (self.w - PANEL_W, self.h), 3)
        
        title = self.font_lg.render("Farm Dashboard", True, (30, 50, 80))
        self.overlay.blit(title, (self.w - PANEL_W + 16, 12))
        
        y = 50
        
        # Weather section
        y = self._section_title("🌤️ Weather", y)
        self._kv("Day", str(self.sim.day), 16, y); y += 22
        self._kv("Temp (°C)", f"{self.sim.weather.temp:.1f}", 16, y); y += 22
        self._kv("Humidity", f"{self.sim.weather.humidity:.2f}", 16, y); y += 22
        self._kv("Rain", "Yes ☔" if self.sim.weather.rain > 0 else "No ☀️", 16, y); y += 22
        
        # Control section
        y = self._section_title("⚙️ Control", y)
        self._kv("Move every", f"{self.move_every} tick(s)", 16, y); y += 22
        self._kv("FPS", f"{self.current_fps}", 16, y); y += 22
        self._kv("Status", "⏸️ Paused" if self.paused else "▶️ Running", 16, y); y += 22
        
        # Metrics section
        y = self._section_title("📊 Metrics", y)
        sustain = self.sim.sustainability_index()
        self._kv("Yield (Σ)", f"{self.sim.total_yield:.2f}", 16, y); y += 22
        self._kv("Sustainability", f"{sustain:.2f}", 16, y); y += 22
        
        # Sustainability indicator bar
        bar_x = self.w - PANEL_W + 16
        bar_w = 200
        pygame.draw.rect(self.overlay, (220, 220, 220), 
                        (bar_x, y, bar_w, 12), border_radius=6)
        
        if sustain > 0.7:
            bar_color = (76, 175, 80)
        elif sustain > 0.4:
            bar_color = (255, 193, 7)
        else:
            bar_color = (244, 67, 54)
        
        pygame.draw.rect(self.overlay, bar_color, 
                        (bar_x, y, int(bar_w * sustain), 12), border_radius=6)
        y += 20
        
        self._kv("Water Used", f"{self.sim.total_water_used:.0f}", 16, y); y += 22
        self._kv("Chemicals", f"{self.sim.total_chem_used:.0f}", 16, y); y += 22
        
        # Agents section
        y = self._section_title("🤖 Agents", y)
        
        for i, a in enumerate(self.sim.agents[:8]):
            # Agent color indicator
            if i % 3 == 0:
                color = (51, 128, 255)
            elif i % 3 == 1:
                color = (255, 153, 51)
            else:
                color = (77, 230, 102)
            
            pygame.draw.circle(self.overlay, color, 
                             (self.w - PANEL_W + 20, y + 8), 5)
            
            action_icon = {
                "irrigate": "💧",
                "fertilize": "🌱",
                "pesticide": "🔴",
                "fungicide": "🟣",
                "monitor": "👁️",
                "idle": "⏸️"
            }.get(a.last_action[:8], "⚡")
            
            # Battery color
            if a.battery > 0.6:
                batt_color = (76, 175, 80)
            elif a.battery > 0.3:
                batt_color = (255, 193, 7)
            else:
                batt_color = (244, 67, 54)
            
            line = f"#{i} ({a.x},{a.y}) {action_icon}"
            text = self.font_xs.render(line, True, (50, 50, 50))
            self.overlay.blit(text, (self.w - PANEL_W + 32, y + 2))
            
            # Battery bar
            bar_x = self.w - PANEL_W + 32
            bar_y = y + 16
            pygame.draw.rect(self.overlay, (200, 200, 200),
                           (bar_x, bar_y, 80, 6), border_radius=3)
            pygame.draw.rect(self.overlay, batt_color,
                           (bar_x, bar_y, int(80 * a.battery), 6), border_radius=3)
            
            y += 28
        
        # Active conditions
        if any(self.conditions.values()):
            y += 10
            y = self._section_title("⚠️ Active Conditions", y)
            for msg in self._active_condition_messages():
                # Wrap text
                words = msg.split()
                line = ""
                for word in words:
                    test_line = line + word + " "
                    if self.font_xs.size(test_line)[0] < PANEL_W - 32:
                        line = test_line
                    else:
                        text = self.font_xs.render(line, True, (100, 60, 20))
                        self.overlay.blit(text, (self.w - PANEL_W + 16, y))
                        y += 16
                        line = word + " "
                if line:
                    text = self.font_xs.render(line, True, (100, 60, 20))
                    self.overlay.blit(text, (self.w - PANEL_W + 16, y))
                    y += 18
    
    def _section_title(self, text, y):
        title = self.font_md.render(text, True, (33, 66, 120))
        self.overlay.blit(title, (self.w - PANEL_W + 16, y))
        pygame.draw.line(self.overlay, (180, 190, 200),
                        (self.w - PANEL_W + 14, y + 24),
                        (self.w - 18, y + 24), 2)
        return y + 35
    
    def _kv(self, k, v, x, y):
        key_text = self.font_sm.render(str(k) + ":", True, (60, 60, 60))
        val_text = self.font_sm.render(str(v), True, (30, 30, 30))
        self.overlay.blit(key_text, (self.w - PANEL_W + x, y))
        self.overlay.blit(val_text, (self.w - PANEL_W + 125, y))
    
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
    """Main simulation loop with enhanced 3D rendering"""
    viz = FarmViz3D(sim)
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
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
                # Toggle features
                elif k == pygame.K_g:
                    viz.show_grid = not viz.show_grid
                elif k == pygame.K_t:
                    viz.show_trails = not viz.show_trails
                # Screenshot
                elif k == pygame.K_p:
                    fname = f"screenshot_3d_{int(time.time())}.png"
                    pygame.image.save(viz.screen, fname)
                    print(f"✅ Saved screenshot: {fname}")
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
                    print(f"✅ Saved snapshot: {fname}")
        
        if viz.paused:
            summary = viz.dynamic_summary(base_llm_summary)
            viz.render(summary)
            continue
        
        viz._apply_weather_overrides()
        
        # Agent loop with trail tracking
        if sim.ticks % viz.move_every == 0:
            for i, agent_obj in enumerate(agents):
                # Get action and movement from the agent object
                if hasattr(agent_obj, 'act'):
                    action, (dx, dy) = agent_obj.act(sim)
                else:
                    # Fallback for non-RL agents - use simple logic
                    action = "monitor"
                    dx, dy = 0, 0
                
                # Move agent
                move_agent(sim, i, dx, dy)
                
                # Apply action
                sim.apply_action(i, action)
                
                # Update trail
                agent_state = sim.agents[i]
                viz.agent_trails[i].add(agent_state.x, agent_state.y, action)
                
                # Recharge at base
                if (agent_state.x, agent_state.y) in BASE_LOCATIONS and action == "idle":
                    agent_state.battery = min(MAX_BATTERY, agent_state.battery + BATTERY_RECHARGE_PER_TICK)
        
        sim.step()
        summary = viz.dynamic_summary(base_llm_summary)
        viz.render(summary)
    
    pygame.quit()
    print("\n🎯 Simulation ended successfully!")
