"""
Dynamic Pathfinding Agent
AI 2002 - Artificial Intelligence Assignment 2
Implements: Greedy Best-First Search & A* Search
Heuristics: Manhattan Distance & Euclidean Distance
GUI: Pygame
"""

import pygame
import heapq
import math
import random
import time
from collections import defaultdict

# ─────────────────────────────────────────────
#  CONSTANTS & COLORS
# ─────────────────────────────────────────────
SCREEN_W, SCREEN_H = 1100, 700
GRID_OFFSET_X, GRID_OFFSET_Y = 10, 60
GRID_W, GRID_H = 720, 630
PANEL_X = 740
PANEL_W = 350

# Color palette
BG          = (15,  17,  26)
PANEL_BG    = (22,  26,  40)
GRID_LINE   = (35,  40,  60)
EMPTY       = (28,  32,  48)
WALL        = (70,  75, 100)
START       = (50, 220, 120)
GOAL        = (220,  70,  90)
FRONTIER    = (240, 200,  50)   # Yellow
VISITED     = (70, 130, 200)    # Blue
PATH        = (80, 220,  80)    # Green
AGENT       = (255, 140,   0)
TEXT_MAIN   = (220, 225, 240)
TEXT_DIM    = (110, 120, 150)
ACCENT      = ( 90, 160, 255)
BTN_ACTIVE  = ( 60, 130, 220)
BTN_HOVER   = ( 80, 150, 240)
BTN_NORMAL  = ( 38,  44,  65)
BTN_BORDER  = ( 60,  70, 100)
SUCCESS     = ( 50, 200, 120)
WARNING     = (240, 160,  50)
DANGER      = (220,  70,  70)

# ─────────────────────────────────────────────
#  HEURISTICS
# ─────────────────────────────────────────────
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# ─────────────────────────────────────────────
#  SEARCH ALGORITHMS
# ─────────────────────────────────────────────
def greedy_bfs(grid, start, goal, rows, cols, heuristic_fn):
    """Greedy Best-First Search - f(n) = h(n)"""
    open_heap = []
    counter = 0
    heapq.heappush(open_heap, (heuristic_fn(start, goal), counter, start))
    came_from = {start: None}
    visited = {start}
    frontier_set = {start}
    expanded_order = []

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        frontier_set.discard(current)

        if current == goal:
            return reconstruct_path(came_from, goal), visited, expanded_order

        expanded_order.append(current)

        for neighbor in get_neighbors(current, grid, rows, cols):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                h = heuristic_fn(neighbor, goal)
                counter += 1
                heapq.heappush(open_heap, (h, counter, neighbor))
                frontier_set.add(neighbor)

    return None, visited, expanded_order


def astar(grid, start, goal, rows, cols, heuristic_fn):
    """A* Search - f(n) = g(n) + h(n) with Expanded List"""
    open_heap = []
    counter = 0
    heapq.heappush(open_heap, (heuristic_fn(start, goal), counter, start))
    came_from = {start: None}
    g_score = defaultdict(lambda: float('inf'))
    g_score[start] = 0
    closed = set()
    frontier_set = {start}
    expanded_order = []

    while open_heap:
        f, _, current = heapq.heappop(open_heap)
        frontier_set.discard(current)

        if current in closed:
            continue

        if current == goal:
            return reconstruct_path(came_from, goal), closed | {current}, expanded_order

        closed.add(current)
        expanded_order.append(current)

        for neighbor in get_neighbors(current, grid, rows, cols):
            if neighbor in closed:
                continue
            tentative_g = g_score[current] + 1

            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_val = tentative_g + heuristic_fn(neighbor, goal)
                counter += 1
                heapq.heappush(open_heap, (f_val, counter, neighbor))
                frontier_set.add(neighbor)

    return None, closed, expanded_order


def get_neighbors(pos, grid, rows, cols):
    r, c = pos
    neighbors = []
    # 4-directional movement
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 1:
            neighbors.append((nr, nc))
    return neighbors


def reconstruct_path(came_from, goal):
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path


# ─────────────────────────────────────────────
#  BUTTON CLASS
# ─────────────────────────────────────────────
class Button:
    def __init__(self, x, y, w, h, label, color=BTN_NORMAL, active_color=BTN_ACTIVE):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.color = color
        self.active_color = active_color
        self.active = False
        self.hovered = False

    def draw(self, surf, font):
        col = self.active_color if self.active else (BTN_HOVER if self.hovered else self.color)
        pygame.draw.rect(surf, col, self.rect, border_radius=6)
        pygame.draw.rect(surf, BTN_BORDER, self.rect, 1, border_radius=6)
        txt = font.render(self.label, True, TEXT_MAIN)
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def check_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)

    def is_clicked(self, pos, event):
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(pos)


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────
class PathfindingApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Dynamic Pathfinding Agent — AI 2002")
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_lg  = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_md  = pygame.font.SysFont("Consolas", 14)
        self.font_sm  = pygame.font.SysFont("Consolas", 12)
        self.font_xl  = pygame.font.SysFont("Consolas", 22, bold=True)

        # Grid config
        self.rows = 20
        self.cols = 24
        self.cell_w = GRID_W // self.cols
        self.cell_h = GRID_H // self.rows

        # State
        self.grid = [[0]*self.cols for _ in range(self.rows)]
        self.start = (2, 2)
        self.goal  = (self.rows-3, self.cols-3)
        self.path  = []
        self.visited_nodes = set()
        self.frontier_nodes = set()
        self.agent_pos = None
        self.agent_step = 0
        self.running_animation = False
        self.dynamic_mode = False
        self.animation_speed = 3   # steps per frame

        # Metrics
        self.nodes_expanded = 0
        self.path_cost = 0
        self.exec_time = 0.0
        self.status_msg = "Ready. Set start/goal, place walls, then Run."
        self.status_color = TEXT_DIM
        self.replans = 0

        # Selections
        self.algorithm = "A*"       # "A*" or "GBFS"
        self.heuristic = "Manhattan" # "Manhattan" or "Euclidean"
        self.edit_mode = "wall"     # "wall", "start", "goal"

        # Animated search trace
        self.anim_visited = []
        self.anim_index = 0
        self.show_anim = False

        # Dynamic obstacle timer
        self.dyn_timer = 0
        self.dyn_interval = 30  # frames between spawns

        self._build_ui()

    # ── UI layout ─────────────────────────────
    def _build_ui(self):
        px = PANEL_X + 10
        bw, bh = 155, 32

        # Algorithm buttons
        self.btn_astar = Button(px, 70, bw, bh, "A* Search")
        self.btn_gbfs  = Button(px+bw+6, 70, bw, bh, "Greedy BFS")
        self.btn_astar.active = True

        # Heuristic buttons
        self.btn_manhattan = Button(px, 120, bw, bh, "Manhattan")
        self.btn_euclidean = Button(px+bw+6, 120, bw, bh, "Euclidean")
        self.btn_manhattan.active = True

        # Edit mode buttons
        self.btn_wall  = Button(px,       170, 96, bh, "Wall")
        self.btn_start = Button(px+102,   170, 96, bh, "Start")
        self.btn_goal  = Button(px+204,   170, 96, bh, "Goal")
        self.btn_wall.active = True

        # Action buttons
        self.btn_run     = Button(px,       220, bw, bh, "▶  Run",    BTN_NORMAL, (40,160,80))
        self.btn_step    = Button(px+bw+6,  220, bw, bh, "⏩  Animate",BTN_NORMAL, (40,120,180))
        self.btn_clear   = Button(px,       260, bw, bh, "🗑  Clear",  BTN_NORMAL, (140,50,50))
        self.btn_random  = Button(px+bw+6,  260, bw, bh, "🎲  Random", BTN_NORMAL, (100,60,160))
        self.btn_dynamic = Button(px,       300, bw, bh, "⚡ Dynamic", BTN_NORMAL, (160,100,20))
        self.btn_reset   = Button(px+bw+6,  300, bw, bh, "↺  Reset",  BTN_NORMAL, BTN_ACTIVE)

        # Grid size slider simulation (buttons)
        self.btn_rows_up   = Button(px+120, 350, 40, 26, "+", BTN_NORMAL, BTN_ACTIVE)
        self.btn_rows_down = Button(px+164, 350, 40, 26, "-", BTN_NORMAL, (140,50,50))
        self.btn_cols_up   = Button(px+120, 382, 40, 26, "+", BTN_NORMAL, BTN_ACTIVE)
        self.btn_cols_down = Button(px+164, 382, 40, 26, "-", BTN_NORMAL, (140,50,50))

        self.all_buttons = [
            self.btn_astar, self.btn_gbfs,
            self.btn_manhattan, self.btn_euclidean,
            self.btn_wall, self.btn_start, self.btn_goal,
            self.btn_run, self.btn_step, self.btn_clear, self.btn_random,
            self.btn_dynamic, self.btn_reset,
            self.btn_rows_up, self.btn_rows_down,
            self.btn_cols_up, self.btn_cols_down,
        ]

    # ── Grid helpers ──────────────────────────
    def cell_at(self, mx, my):
        gx = mx - GRID_OFFSET_X
        gy = my - GRID_OFFSET_Y
        if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
            c = gx // self.cell_w
            r = gy // self.cell_h
            if 0 <= r < self.rows and 0 <= c < self.cols:
                return r, c
        return None

    def cell_rect(self, r, c):
        x = GRID_OFFSET_X + c * self.cell_w
        y = GRID_OFFSET_Y + r * self.cell_h
        return pygame.Rect(x, y, self.cell_w-1, self.cell_h-1)

    def resize_grid(self, new_rows, new_cols):
        new_rows = max(5, min(40, new_rows))
        new_cols = max(5, min(40, new_cols))
        self.rows = new_rows
        self.cols = new_cols
        self.cell_w = GRID_W // self.cols
        self.cell_h = GRID_H // self.rows
        self.grid = [[0]*self.cols for _ in range(self.rows)]
        self.start = (2, 2)
        self.goal  = (self.rows-3, self.cols-3)
        self._clear_results()

    def _clear_results(self):
        self.path = []
        self.visited_nodes = set()
        self.frontier_nodes = set()
        self.agent_pos = None
        self.agent_step = 0
        self.running_animation = False
        self.show_anim = False
        self.anim_index = 0
        self.anim_visited = []
        self.nodes_expanded = 0
        self.path_cost = 0
        self.exec_time = 0.0
        self.replans = 0
        self.status_msg = "Cleared."
        self.status_color = TEXT_DIM

    def generate_random(self):
        self._clear_results()
        density = 0.28
        self.grid = [[0]*self.cols for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in (self.start, self.goal):
                    if random.random() < density:
                        self.grid[r][c] = 1
        self.status_msg = f"Random maze generated ({int(density*100)}% walls)."
        self.status_color = ACCENT

    # ── Search ────────────────────────────────
    def run_search(self, animate=False):
        hfn = manhattan if self.heuristic == "Manhattan" else euclidean
        t0 = time.time()
        if self.algorithm == "A*":
            path, visited, expanded = astar(self.grid, self.start, self.goal, self.rows, self.cols, hfn)
        else:
            path, visited, expanded = greedy_bfs(self.grid, self.start, self.goal, self.rows, self.cols, hfn)
        t1 = time.time()

        self.exec_time = (t1 - t0) * 1000
        self.nodes_expanded = len(visited)
        self.visited_nodes = visited
        self.anim_visited = expanded

        if path:
            self.path = path
            self.path_cost = len(path) - 1
            self.status_msg = f"Path found! Cost={self.path_cost}, Nodes={self.nodes_expanded}"
            self.status_color = SUCCESS
            if animate:
                self.agent_pos = self.start
                self.agent_step = 0
                self.running_animation = True
        else:
            self.path = []
            self.status_msg = "No path found!"
            self.status_color = DANGER

    def replan_from(self, pos):
        """Re-plan from current agent position when blocked."""
        hfn = manhattan if self.heuristic == "Manhattan" else euclidean
        if self.algorithm == "A*":
            path, visited, _ = astar(self.grid, pos, self.goal, self.rows, self.cols, hfn)
        else:
            path, visited, _ = greedy_bfs(self.grid, pos, self.goal, self.rows, self.cols, hfn)
        if path:
            self.path = path
            self.agent_step = 0
            self.replans += 1
            self.path_cost = len(path) - 1
            self.visited_nodes = visited
            self.status_msg = f"Re-planned! Replans={self.replans}, Cost={self.path_cost}"
            self.status_color = WARNING
        else:
            self.running_animation = False
            self.status_msg = "Agent is trapped! No path available."
            self.status_color = DANGER

    # ── Main loop ─────────────────────────────
    def run(self):
        dragging = False
        drag_value = 1

        while True:
            mx, my = pygame.mouse.get_pos()
            for btn in self.all_buttons:
                btn.check_hover((mx, my))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

                # Mouse drag on grid
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    cell = self.cell_at(mx, my)
                    if cell:
                        r, c = cell
                        if self.edit_mode == "wall":
                            dragging = True
                            drag_value = 0 if self.grid[r][c] == 1 else 1
                            if cell not in (self.start, self.goal):
                                self.grid[r][c] = drag_value
                        elif self.edit_mode == "start":
                            if cell != self.goal:
                                self.start = cell
                                self._clear_results()
                        elif self.edit_mode == "goal":
                            if cell != self.start:
                                self.goal = cell
                                self._clear_results()

                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    dragging = False

                if event.type == pygame.MOUSEMOTION and dragging and self.edit_mode == "wall":
                    cell = self.cell_at(mx, my)
                    if cell and cell not in (self.start, self.goal):
                        self.grid[cell[0]][cell[1]] = drag_value

                # Buttons
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.btn_astar.is_clicked((mx,my), event):
                        self.algorithm = "A*"
                        self.btn_astar.active = True
                        self.btn_gbfs.active = False
                    elif self.btn_gbfs.is_clicked((mx,my), event):
                        self.algorithm = "GBFS"
                        self.btn_gbfs.active = True
                        self.btn_astar.active = False

                    elif self.btn_manhattan.is_clicked((mx,my), event):
                        self.heuristic = "Manhattan"
                        self.btn_manhattan.active = True
                        self.btn_euclidean.active = False
                    elif self.btn_euclidean.is_clicked((mx,my), event):
                        self.heuristic = "Euclidean"
                        self.btn_euclidean.active = True
                        self.btn_manhattan.active = False

                    elif self.btn_wall.is_clicked((mx,my), event):
                        self.edit_mode = "wall"
                        self.btn_wall.active = True
                        self.btn_start.active = False
                        self.btn_goal.active = False
                    elif self.btn_start.is_clicked((mx,my), event):
                        self.edit_mode = "start"
                        self.btn_start.active = True
                        self.btn_wall.active = False
                        self.btn_goal.active = False
                    elif self.btn_goal.is_clicked((mx,my), event):
                        self.edit_mode = "goal"
                        self.btn_goal.active = True
                        self.btn_wall.active = False
                        self.btn_start.active = False

                    elif self.btn_run.is_clicked((mx,my), event):
                        self._clear_results()
                        self.run_search(animate=False)
                    elif self.btn_step.is_clicked((mx,my), event):
                        self._clear_results()
                        self.run_search(animate=True)
                    elif self.btn_clear.is_clicked((mx,my), event):
                        self.grid = [[0]*self.cols for _ in range(self.rows)]
                        self._clear_results()
                    elif self.btn_random.is_clicked((mx,my), event):
                        self.generate_random()
                    elif self.btn_dynamic.is_clicked((mx,my), event):
                        self.dynamic_mode = not self.dynamic_mode
                        self.btn_dynamic.active = self.dynamic_mode
                        self.status_msg = f"Dynamic obstacles: {'ON' if self.dynamic_mode else 'OFF'}"
                        self.status_color = WARNING if self.dynamic_mode else TEXT_DIM
                    elif self.btn_reset.is_clicked((mx,my), event):
                        self._clear_results()
                        self.status_msg = "Results cleared."

                    elif self.btn_rows_up.is_clicked((mx,my), event):
                        self.resize_grid(self.rows+2, self.cols)
                    elif self.btn_rows_down.is_clicked((mx,my), event):
                        self.resize_grid(self.rows-2, self.cols)
                    elif self.btn_cols_up.is_clicked((mx,my), event):
                        self.resize_grid(self.rows, self.cols+2)
                    elif self.btn_cols_down.is_clicked((mx,my), event):
                        self.resize_grid(self.rows, self.cols-2)

            # ── Animation tick ─────────────────
            if self.running_animation and self.path:
                for _ in range(self.animation_speed):
                    if self.agent_step < len(self.path) - 1:
                        self.agent_step += 1
                        self.agent_pos = self.path[self.agent_step]
                    else:
                        self.running_animation = False
                        self.status_msg = f"✓ Goal reached! Cost={self.path_cost}, Nodes={self.nodes_expanded}, Time={self.exec_time:.2f}ms"
                        self.status_color = SUCCESS
                        break

            # ── Dynamic obstacles ──────────────
            if self.dynamic_mode and self.running_animation and self.agent_pos:
                self.dyn_timer += 1
                if self.dyn_timer >= self.dyn_interval:
                    self.dyn_timer = 0
                    # Spawn 1–2 random walls not on start/goal/agent
                    for _ in range(random.randint(1,2)):
                        tr = random.randint(0, self.rows-1)
                        tc = random.randint(0, self.cols-1)
                        pos = (tr, tc)
                        if pos not in (self.start, self.goal, self.agent_pos):
                            self.grid[tr][tc] = 1
                            # Check if new wall blocks current path
                            if self.path and pos in self.path:
                                self.replan_from(self.agent_pos)

            self._draw()
            self.clock.tick(60)

    # ── Drawing ───────────────────────────────
    def _draw(self):
        self.screen.fill(BG)
        self._draw_header()
        self._draw_grid()
        self._draw_panel()
        pygame.display.flip()

    def _draw_header(self):
        title = self.font_xl.render("Dynamic Pathfinding Agent", True, ACCENT)
        self.screen.blit(title, (GRID_OFFSET_X, 14))
        sub = self.font_sm.render("AI 2002 — Informed Search  |  Drag to place walls  |  Click Start/Goal to reposition", True, TEXT_DIM)
        self.screen.blit(sub, (GRID_OFFSET_X, 38))

    def _draw_grid(self):
        # Background
        pygame.draw.rect(self.screen, (20,23,35), 
                         (GRID_OFFSET_X-2, GRID_OFFSET_Y-2, GRID_W+4, GRID_H+4), border_radius=4)

        for r in range(self.rows):
            for c in range(self.cols):
                rect = self.cell_rect(r, c)
                pos = (r, c)

                if self.grid[r][c] == 1:
                    color = WALL
                elif pos == self.start:
                    color = START
                elif pos == self.goal:
                    color = GOAL
                elif self.running_animation and pos == self.agent_pos:
                    color = AGENT
                elif self.running_animation and self.path and pos in self.path[:self.agent_step+1]:
                    color = PATH
                elif not self.running_animation and self.path and pos in self.path:
                    color = PATH
                elif pos in self.frontier_nodes:
                    color = FRONTIER
                elif pos in self.visited_nodes:
                    color = VISITED
                else:
                    color = EMPTY

                pygame.draw.rect(self.screen, color, rect, border_radius=2)

        # Grid lines
        for r in range(self.rows+1):
            y = GRID_OFFSET_Y + r*self.cell_h
            pygame.draw.line(self.screen, GRID_LINE, (GRID_OFFSET_X, y), (GRID_OFFSET_X+GRID_W, y))
        for c in range(self.cols+1):
            x = GRID_OFFSET_X + c*self.cell_w
            pygame.draw.line(self.screen, GRID_LINE, (x, GRID_OFFSET_Y), (x, GRID_OFFSET_Y+GRID_H))

        # Start / Goal labels
        sr = self.cell_rect(*self.start)
        gr = self.cell_rect(*self.goal)
        st = self.font_sm.render("S", True, (0,0,0))
        gt = self.font_sm.render("G", True, (0,0,0))
        self.screen.blit(st, st.get_rect(center=sr.center))
        self.screen.blit(gt, gt.get_rect(center=gr.center))

    def _draw_panel(self):
        # Panel background
        panel_rect = pygame.Rect(PANEL_X, 0, PANEL_W, SCREEN_H)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect)
        pygame.draw.line(self.screen, BTN_BORDER, (PANEL_X, 0), (PANEL_X, SCREEN_H), 1)

        px = PANEL_X + 10

        # Section: Algorithm
        self._section_label("ALGORITHM", px, 50)
        self.btn_astar.draw(self.screen, self.font_md)
        self.btn_gbfs.draw(self.screen, self.font_md)

        # Section: Heuristic
        self._section_label("HEURISTIC", px, 102)
        self.btn_manhattan.draw(self.screen, self.font_md)
        self.btn_euclidean.draw(self.screen, self.font_md)

        # Section: Edit Mode
        self._section_label("EDIT MODE", px, 152)
        self.btn_wall.draw(self.screen, self.font_md)
        self.btn_start.draw(self.screen, self.font_md)
        self.btn_goal.draw(self.screen, self.font_md)

        # Section: Actions
        self._section_label("ACTIONS", px, 202)
        self.btn_run.draw(self.screen, self.font_md)
        self.btn_step.draw(self.screen, self.font_md)
        self.btn_clear.draw(self.screen, self.font_md)
        self.btn_random.draw(self.screen, self.font_md)
        self.btn_dynamic.draw(self.screen, self.font_md)
        self.btn_reset.draw(self.screen, self.font_md)

        # Section: Grid Size
        self._section_label("GRID SIZE", px, 334)
        self.screen.blit(self.font_md.render(f"Rows: {self.rows}", True, TEXT_MAIN), (px, 353))
        self.btn_rows_up.draw(self.screen, self.font_md)
        self.btn_rows_down.draw(self.screen, self.font_md)
        self.screen.blit(self.font_md.render(f"Cols: {self.cols}", True, TEXT_MAIN), (px, 385))
        self.btn_cols_up.draw(self.screen, self.font_md)
        self.btn_cols_down.draw(self.screen, self.font_md)

        # Section: Metrics
        self._section_label("METRICS", px, 424)
        metrics = [
            ("Nodes Expanded", str(self.nodes_expanded)),
            ("Path Cost",      str(self.path_cost)),
            ("Exec Time",      f"{self.exec_time:.2f} ms"),
            ("Re-plans",       str(self.replans)),
        ]
        for i, (k, v) in enumerate(metrics):
            y = 444 + i*24
            self.screen.blit(self.font_sm.render(k, True, TEXT_DIM),  (px, y))
            vtxt = self.font_sm.render(v, True, ACCENT)
            self.screen.blit(vtxt, (px+PANEL_W-30-vtxt.get_width(), y))

        # Status message
        self._section_label("STATUS", px, 544)
        # Word-wrap the status
        words = self.status_msg.split()
        line, lines = "", []
        for w in words:
            test = line + w + " "
            if self.font_sm.size(test)[0] > PANEL_W - 24:
                lines.append(line)
                line = w + " "
            else:
                line = test
        lines.append(line)
        for i, ln in enumerate(lines[:3]):
            self.screen.blit(self.font_sm.render(ln.strip(), True, self.status_color), (px, 564 + i*18))

        # Legend
        self._draw_legend(px, 636)

    def _section_label(self, text, x, y):
        lbl = self.font_sm.render(text, True, TEXT_DIM)
        self.screen.blit(lbl, (x, y))
        pygame.draw.line(self.screen, BTN_BORDER, (x, y+14), (x+PANEL_W-22, y+14), 1)

    def _draw_legend(self, px, y):
        items = [
            (START,    "Start"),
            (GOAL,     "Goal"),
            (PATH,     "Path"),
            (VISITED,  "Visited"),
            (FRONTIER, "Frontier"),
            (WALL,     "Wall"),
            (AGENT,    "Agent"),
        ]
        self.screen.blit(self.font_sm.render("LEGEND", True, TEXT_DIM), (px, y-16))
        pygame.draw.line(self.screen, BTN_BORDER, (px, y-2), (px+PANEL_W-22, y-2), 1)
        cols = 2
        for i, (color, label) in enumerate(items):
            cx = px + (i % cols) * 160
            cy = y + (i // cols) * 18
            pygame.draw.rect(self.screen, color, (cx, cy+2, 12, 12), border_radius=2)
            self.screen.blit(self.font_sm.render(label, True, TEXT_MAIN), (cx+16, cy))


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = PathfindingApp()
    app.run()
