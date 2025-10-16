#!/usr/bin/env python3
"""
dual_view.py — Single-screen synchronized 2D + 3D view with legend + HUD
- Left: 2D Pygame surface (grid heatmap + agents), sent as a texture.
- Right: True 3D OpenGL scene + semi-transparent sidebar HUD (legend & text).
- Top: global status bar (ticks, FPS, agents).
"""

import math, time
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from config import *  # GRID_W, GRID_H, FPS, BASE_LOCATIONS, etc.
from simulator import FarmSimulator
from rl_swarm import RuleBasedAgent, SimpleA2CAgent, move_agent
from llm_parser import parse_report

MOVE_EVERY_N_TICKS_DEFAULT = 8  # default tick interval between agent moves

# ---------------- Camera ----------------
class OrbitCamera:
    def __init__(self):
        self.yaw = -40.0
        self.pitch = 30.0
        self.distance = 40.0
        self.target = (GRID_W/2, 0, GRID_H/2)
        self._dragging = False
    def begin_drag(self, _pos): self._dragging = True
    def end_drag(self): self._dragging = False
    def handle_motion(self, _pos, buttons, rel):
        if self._dragging and buttons[2]:
            dx, dy = rel
            self.yaw += dx * 0.3
            self.pitch = max(-89, min(89, self.pitch - dy * 0.3))
    def zoom(self, delta):
        self.distance = max(5.0, min(120.0, self.distance * (0.9 if delta > 0 else 1.1)))
    def apply(self):
        x = self.target[0] + self.distance * math.cos(math.radians(self.pitch)) * math.cos(math.radians(self.yaw))
        z = self.target[2] + self.distance * math.cos(math.radians(self.pitch)) * math.sin(math.radians(self.yaw))
        y = self.target[1] + self.distance * math.sin(math.radians(self.pitch))
        gluLookAt(x, y, z, self.target[0], self.target[1], self.target[2], 0, 1, 0)

# ---------------- OpenGL helpers ----------------
def gl_init():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (20, 50, 10, 1))
    glLightfv(GL_LIGHT0, GL_AMBIENT,  (0.3, 0.3, 0.3, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  (0.8, 0.8, 0.8, 1))
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClearColor(0.06, 0.06, 0.07, 1.0)

def set_perspective(w, h):
    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    gluPerspective(60.0, max(1.0, w/float(max(1, h))), 0.1, 500.0)
    glMatrixMode(GL_MODELVIEW)

def set_ortho(w, h):
    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    glOrtho(0, w, h, 0, -1, 1)  # y-down to match pygame
    glMatrixMode(GL_MODELVIEW); glLoadIdentity()

def glut_like_cube():
    glBegin(GL_QUADS)
    # +Y
    glNormal3f(0,1,0)
    glVertex3f(-.5,.5,-.5); glVertex3f(.5,.5,-.5); glVertex3f(.5,.5,.5); glVertex3f(-.5,.5,.5)
    # -Y
    glNormal3f(0,-1,0)
    glVertex3f(-.5,-.5,.5); glVertex3f(.5,-.5,.5); glVertex3f(.5,-.5,-.5); glVertex3f(-.5,-.5,-.5)
    # +X
    glNormal3f(1,0,0)
    glVertex3f(.5,-.5,-.5); glVertex3f(.5,-.5,.5); glVertex3f(.5,.5,.5); glVertex3f(.5,.5,-.5)
    # -X
    glNormal3f(-1,0,0)
    glVertex3f(-.5,-.5,.5); glVertex3f(-.5,-.5,-.5); glVertex3f(-.5,.5,-.5); glVertex3f(-.5,.5,.5)
    # +Z
    glNormal3f(0,0,1)
    glVertex3f(-.5,-.5,.5); glVertex3f(.5,-.5,.5); glVertex3f(.5,.5,.5); glVertex3f(-.5,.5,.5)
    # -Z
    glNormal3f(0,0,-1)
    glVertex3f(.5,-.5,-.5); glVertex3f(-.5,-.5,-.5); glVertex3f(-.5,.5,-.5); glVertex3f(.5,.5,-.5)
    glEnd()

def draw_cube(x, y, z, s=1.0, color=(0.4,0.8,0.4,1)):
    glPushMatrix()
    glTranslatef(x, y, z)
    glScalef(s, s, s)
    glColor4f(*color)
    glut_like_cube()
    glPopMatrix()

def draw_ground(sim):
    glDisable(GL_LIGHTING)
    glColor4f(0.15,0.15,0.15,1)
    glBegin(GL_LINES)
    for x in range(sim.w+1):
        glVertex3f(x, 0, 0); glVertex3f(x, 0, sim.h)
    for z in range(sim.h+1):
        glVertex3f(0, 0, z); glVertex3f(sim.w, 0, z)
    glEnd()
    glEnable(GL_LIGHTING)

# ---------------- Color helpers ----------------
def health_to_color_rgb(health: float):
    """Map 0..1 health to color: red → yellow → green."""
    h = max(0.0, min(1.0, health))
    if h < 0.5:
        t = h / 0.5
        r = int(180 + (220-180)*t)
        g = int( 60 + (200- 60)*t)
        b = int( 60 + ( 80- 60)*t)
    else:
        t = (h-0.5)/0.5
        r = int(220 + ( 60-220)*t)
        g = int(200 + (200-200)*t)
        b = int( 80 + ( 90- 80)*t)
    return (r,g,b)

def agent_color(i: int):
    palette = [
        (70,160,255), (255,120,70), (120,220,120), (200,120,220),
        (255,200,80), (120,220,220), (230,120,120), (160,160,255),
    ]
    return palette[i % len(palette)]

# ---------------- 3D scene ----------------
def draw_scene_3d(sim):
    for y in range(sim.h):
        for x in range(sim.w):
            c = sim.grid[x][y]
            h = max(0.0, min(1.0, c.growth)) * 0.8
            if h <= 0.02:
                continue
            health = c.health()
            col = (0.2 + 0.8*health, 0.35 + 0.4*health, 0.2, 1)
            draw_cube(x+0.5, h/2.0, y+0.5, s=1.0, color=col)
    for i, a in enumerate(sim.agents):
        r,g,b = agent_color(i)
        draw_cube(a.x+0.5, 0.5, a.y+0.5, s=0.7, color=(r/255.0,g/255.0,b/255.0,1))

# ---------------- 2D scene ----------------
def draw_2d_surface(surface, sim, font, font_small):
    surface.fill((24, 26, 27))
    cell_w = surface.get_width() // sim.w
    cell_h = (surface.get_height()-32) // sim.h  # leave small footer
    cw = max(2, min(cell_w, cell_h))
    ox, oy = 8, 8

    # grid
    for y in range(sim.h):
        for x in range(sim.w):
            rx = ox + x*cw
            ry = oy + y*cw
            c = sim.grid[x][y]
            col = health_to_color_rgb(c.health())
            pygame.draw.rect(surface, col, (rx, ry, cw-1, cw-1))

    # bases
    for (bx, by) in BASE_LOCATIONS:
        rx = ox + bx*cw; ry = oy + by*cw
        pygame.draw.rect(surface, (90,90,90), (rx, ry, cw-1, cw-1), width=2)

    # agents
    for i, a in enumerate(sim.agents):
        rx = ox + a.x*cw; ry = oy + a.y*cw
        pygame.draw.rect(surface, agent_color(i), (rx+2, ry+2, cw-5, cw-5))

    # footer: small health gradient
    footer_h = 24
    gx, gy = 10, surface.get_height()-footer_h-6
    grad_w = max(80, surface.get_width()-20)
    grad = pygame.Surface((grad_w, 10), SRCALPHA)
    for ix in range(grad_w):
        col = health_to_color_rgb(ix/float(max(1, grad_w-1)))
        pygame.draw.line(grad, col, (ix, 0), (ix, 10))
    surface.blit(grad, (gx, gy))
    surface.blit(font_small.render("Crop health 0 → 1", True, (210,210,210)), (gx, gy+12))

# ---------------- HUD (legend + text) ----------------
def draw_hud_surface(surface, sim: FarmSimulator, font, font_small, llm_summary, llm_mult):
    # normalize inputs
    if isinstance(llm_summary, (list, tuple)):
        llm_text = " ".join(map(str, llm_summary))
    else:
        llm_text = str(llm_summary) if llm_summary is not None else ""
    try:
        llm_mult = dict(llm_mult)
    except Exception:
        llm_mult = {}

    W, H = surface.get_width(), surface.get_height()
    surface.fill((0,0,0,0))  # transparent
    panel = pygame.Surface((W, H), SRCALPHA)
    panel.fill((12,12,14,210))  # semi-transparent panel
    surface.blit(panel, (0,0))

    y = 12
    def title(t): 
        nonlocal y
        surface.blit(font.render(t, True, (240,240,240)), (14, y)); y += 26
    def kv(k,v):
        nonlocal y
        surface.blit(font_small.render(f"{k}: {v}", True, (210,210,210)), (18, y)); y += 20
    def sep(): 
        nonlocal y
        pygame.draw.line(surface, (70,70,80), (12, y+6), (W-12, y+6), 1); y += 16

    # Legend
    title("Legend")
    grad_w = W-28; grad_h = 14
    grad_surf = pygame.Surface((grad_w, grad_h), SRCALPHA)
    for ix in range(grad_w):
        col = health_to_color_rgb(ix/float(max(1, grad_w-1)))
        pygame.draw.line(grad_surf, col, (ix, 0), (ix, grad_h))
    surface.blit(grad_surf, (14, y))
    surface.blit(font_small.render("Crop Health 0 → 1", True, (200,200,200)), (14, y+grad_h+4))
    y += grad_h + 26
    surface.blit(font_small.render("Agents:", True, (200,200,200)), (14, y)); y += 18
    for i in range(min(NUM_AGENTS, 8)):
        c = agent_color(i)
        pygame.draw.rect(surface, c, (18, y+2, 14, 14))
        surface.blit(font_small.render(f"Agent {i}", True, (200,200,200)), (38, y))
        y += 18
    sep()

    # Weather (live)
    title("Weather")
    if hasattr(sim.weather, "temp"): kv("Temp", f"{sim.weather.temp:.1f} °C")
    if hasattr(sim.weather, "humidity"): kv("Humidity", f"{sim.weather.humidity:.2f}")
    if hasattr(sim.weather, "rain"): kv("Rain", f"{sim.weather.rain:.2f}")
    if hasattr(sim.weather, "wind_dx") and hasattr(sim.weather, "wind_dy"):
        kv("Wind", f"({sim.weather.wind_dx:.2f}, {sim.weather.wind_dy:.2f})")
    sep()

    # LLM Advisory
    title("LLM Advisory")
    wrap, line, maxw = [], "", W-28
    for w in llm_text.split():
        t = line + (" " if line else "") + w
        if font_small.size(t)[0] > maxw:
            wrap.append(line); line = w
        else:
            line = t
    if line: wrap.append(line)
    for s in wrap[:10]:
        surface.blit(font_small.render(s, True, (210,210,210)), (14, y)); y += 18
    if len(wrap) > 10:
        surface.blit(font_small.render("…", True, (210,210,210)), (14, y)); y += 18

    y += 4
    surface.blit(font_small.render("Multipliers:", True, (200,200,200)), (14, y)); y += 18
    for k,v in list(llm_mult.items())[:10]:
        try: vv = f"{float(v):.2f}"
        except Exception: vv = str(v)
        surface.blit(font_small.render(f"{k}: {vv}", True, (210,210,210)), (24, y)); y += 18
    sep()

    # Sustainability
    title("Sustainability")
    if hasattr(sim, "sustainability_index"):
        kv("Index", f"{sim.sustainability_index():.3f}")
    kv("Total Yield", f"{sim.total_yield:.1f}")
    kv("Water Used", f"{sim.total_water_used:.1f}")
    kv("Chem Used", f"{sim.total_chem_used:.1f}")
    if hasattr(sim, "biodiversity_score"):
        kv("Biodiversity", f"{sim.biodiversity_score:.2f}")
    sep()

    # Controls
    title("Controls")
    for line in [
        "Space: Pause/Resume   N: Step",
        "+ / -: Move speed     [ / ]: FPS",
        "R: Rain  S: Sunny  W: Wind  D: Drought",
        "Right-drag: Orbit     Wheel: Zoom",
        "Esc: Quit"
    ]:
        surface.blit(font_small.render(line, True, (210,210,210)), (14, y)); y += 18

# ---------------- Status Bar (top, across window) ----------------
def draw_status_bar(surface, font, ticks, fps, num_agents):
    W, H = surface.get_width(), surface.get_height()
    surface.fill((0,0,0,0))
    bar = pygame.Surface((W, H), SRCALPHA)
    bar.fill((15,15,18,215))
    surface.blit(bar, (0,0))
    text = f"Ticks: {ticks}    FPS cap: {fps}    Agents: {num_agents}"
    surface.blit(font.render(text, True, (235,235,235)), (12, 4))

# ---------------- Texture wrapper ----------------
class SurfaceTexture:
    def __init__(self, width, height):
        self.tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        self.w, self.h = width, height
    def update_from_surface(self, surf):
        data = pygame.image.tostring(surf, "RGBA", False)  # keep orientation
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.w, self.h, GL_RGBA, GL_UNSIGNED_BYTE, data)
    def draw_rect(self, x, y, w, h):
        glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, self.tex_id)
        vp_w, vp_h = glGetIntegerv(GL_VIEWPORT)[2], glGetIntegerv(GL_VIEWPORT)[3]
        set_ortho(vp_w, vp_h)
        glBegin(GL_QUADS)  # texcoords so text appears upright
        glTexCoord2f(0,0); glVertex2f(x,   y)      # top-left
        glTexCoord2f(1,0); glVertex2f(x+w, y)      # top-right
        glTexCoord2f(1,1); glVertex2f(x+w, y+h)    # bottom-right
        glTexCoord2f(0,1); glVertex2f(x,   y+h)    # bottom-left
        glEnd()
        glDisable(GL_TEXTURE_2D); glEnable(GL_LIGHTING); glEnable(GL_DEPTH_TEST)

# ---------------- Main ----------------
def build_controllers(mode="rule"):
    return [RuleBasedAgent(i) if mode=="rule" else SimpleA2CAgent(i) for i in range(NUM_AGENTS)]

def main():
    pygame.init()
    pygame.display.set_caption("Dual View — Synchronized 2D + 3D (Legend + HUD)")
    W, H = 1400, 820
    topbar_h = 28
    left_w = W // 2
    right_w = W - left_w
    hud_w = min(360, right_w)

    pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL)
    gl_init()

    surf2d = pygame.Surface((left_w, H-topbar_h), flags=SRCALPHA).convert_alpha()
    tex2d = SurfaceTexture(left_w, H-topbar_h)

    hud_surface = pygame.Surface((hud_w, H-topbar_h), flags=SRCALPHA).convert_alpha()
    hud_tex = SurfaceTexture(hud_w, H-topbar_h)

    bar_surface = pygame.Surface((W, topbar_h), flags=SRCALPHA).convert_alpha()
    bar_tex = SurfaceTexture(W, topbar_h)

    font = pygame.font.SysFont("consolas", 18)
    font_small = pygame.font.SysFont("consolas", 16)

    sim = FarmSimulator()  # creates sim.agents

    # LLM advisory (robust)
    report = ("Weather bulletin: heatwave expected; humidity moderate. "
              "Agronomy watch: aphid activity rising in northern parcels. "
              "Blight risk near low-drainage zones. Increase monitoring.")
    llm = parse_report(report) or {}
    raw_summary = llm.get("summary", "")
    llm_summary = " ".join(map(str, raw_summary)) if isinstance(raw_summary, (list,tuple)) else str(raw_summary or "")
    raw_mult = llm.get("multipliers", {})
    try: sim.llm_shaping.update(dict(raw_mult))
    except Exception: pass
    llm_mult = dict(sim.llm_shaping)

    controllers = build_controllers("rule")

    move_every = MOVE_EVERY_N_TICKS_DEFAULT
    fps = FPS
    paused = False
    cam = OrbitCamera()
    clock = pygame.time.Clock()

    running = True
    while running:
        # events
        for event in pygame.event.get():
            if event.type == QUIT: running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE: running = False
                elif event.key == K_SPACE: paused = not paused
                elif event.key == K_n: paused = True; sim.step()
                elif event.key in (K_PLUS, K_EQUALS): move_every = max(1, move_every-1)
                elif event.key == K_MINUS: move_every += 1
                elif event.key == K_LEFTBRACKET: fps = max(5, fps-5)
                elif event.key == K_RIGHTBRACKET: fps = min(240, fps+5)
                # weather presets
                elif event.key == K_r:
                    sim.weather.rain = 1.0 if getattr(sim.weather, "rain", 0.0) == 0.0 else 0.0
                elif event.key == K_s:
                    sim.weather.rain = 0.0
                    if hasattr(sim.weather, "temp"): sim.weather.temp = TEMP_MEAN + 4.0
                    if hasattr(sim.weather, "humidity"): sim.weather.humidity = max(0.0, HUMID_MEAN - 0.1)
                elif event.key == K_w:
                    sim.weather.wind_dx = 0.8 if getattr(sim.weather, "wind_dx", 0.0) == 0.0 else 0.0
                    sim.weather.wind_dy = 0.2 if getattr(sim.weather, "wind_dy", 0.0) == 0.0 else 0.0
                elif event.key == K_d:
                    sim.weather.rain = 0.0
                    if hasattr(sim.weather, "humidity"): sim.weather.humidity = max(0.0, HUMID_MEAN - 0.25)
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 3: cam.begin_drag(pygame.mouse.get_pos())
                elif event.button == 4: cam.zoom(+1)
                elif event.button == 5: cam.zoom(-1)
            elif event.type == MOUSEBUTTONUP:
                if event.button == 3: cam.end_drag()
            elif event.type == MOUSEMOTION:
                cam.handle_motion(pygame.mouse.get_pos(), pygame.mouse.get_pressed(), event.rel)

        # update
        if not paused and sim.ticks % move_every == 0:
            for i, ctrl in enumerate(controllers):
                action, (dx, dy) = ctrl.act(sim)
                move_agent(sim, i, dx, dy)
                sim.apply_action(i, action)
                a = sim.agents[i]
                if (a.x, a.y) in BASE_LOCATIONS and action == "idle":
                    a.battery = min(MAX_BATTERY, a.battery + BATTERY_RECHARGE_PER_TICK)

        if not paused:
            sim.step()

        # render: clear once
        glViewport(0, 0, W, H)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Top bar (across window)
        draw_status_bar(bar_surface, font_small, sim.ticks, fps, len(sim.agents))
        bar_tex.update_from_surface(bar_surface)
        glViewport(0, 0, W, topbar_h)
        bar_tex.draw_rect(0, 0, W, topbar_h)

        # Left: 2D → texture (left half, below status bar)
        draw_2d_surface(surf2d, sim, font, font_small)
        tex2d.update_from_surface(surf2d)
        glViewport(0, 0, left_w, H)  # set left viewport
        tex2d.draw_rect(0, topbar_h, left_w, H-topbar_h)

        # Right: 3D world (right half, full height)
        glViewport(left_w, 0, right_w, H)
        set_perspective(right_w, H)
        glLoadIdentity()
        cam.apply()
        draw_ground(sim)
        draw_scene_3d(sim)

        # Right overlay: HUD panel (right side, below status bar)
        draw_hud_surface(hud_surface, sim, font, font_small, llm_summary, llm_mult)
        hud_tex.update_from_surface(hud_surface)
        glViewport(left_w, 0, right_w, H)
        hud_x = right_w - hud_w
        hud_tex.draw_rect(hud_x, topbar_h, hud_w, H-topbar_h)

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()

if __name__ == "__main__":
    main()
