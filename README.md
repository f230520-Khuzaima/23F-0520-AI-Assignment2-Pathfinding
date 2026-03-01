# Dynamic Pathfinding Agent
### AI 2002 – Artificial Intelligence | Assignment 2 | Question 6

A real-time, interactive grid-based pathfinding visualizer implementing **Greedy Best-First Search** and **A\*** with dynamic obstacle re-planning.

---

## Features

| Feature | Description |
|---|---|
| **Algorithms** | A\* Search, Greedy Best-First Search (GBFS) |
| **Heuristics** | Manhattan Distance, Euclidean Distance |
| **Grid Sizing** | Adjustable rows/columns (5–40) |
| **Interactive Editor** | Click/drag to place walls; reposition Start & Goal |
| **Random Maze** | One-click random maze generation (~28% wall density) |
| **Animation Mode** | Watch the agent navigate step-by-step |
| **Dynamic Obstacles** | Random walls spawn mid-navigation, agent re-plans in real time |
| **Metrics Dashboard** | Nodes expanded, path cost, execution time, re-plan count |
| **Color-coded Visualization** | Frontier (yellow), Visited (blue), Path (green), Agent (orange) |

---

## Installation

```bash
pip install pygame
```

Python 3.8+ required.

---

## How to Run

```bash
python main.py
```

---

## Controls

| Control | Action |
|---|---|
| **Wall** mode + drag | Place/erase walls on the grid |
| **Start** mode + click | Move the start node |
| **Goal** mode + click | Move the goal node |
| **▶ Run** | Instantly compute and show path |
| **⏩ Animate** | Animate the agent traversing the path |
| **⚡ Dynamic** | Toggle dynamic obstacle mode (spawns walls during animation) |
| **🎲 Random** | Generate a random maze |
| **🗑 Clear** | Remove all walls |
| **↺ Reset** | Clear search results only |
| **+/- Rows/Cols** | Resize the grid |

---

## Algorithm Descriptions

### A\* Search
Uses `f(n) = g(n) + h(n)` where `g(n)` is the actual cost from start and `h(n)` is the heuristic estimate to the goal. Guaranteed to find the **optimal path** when the heuristic is admissible.

### Greedy Best-First Search
Uses `f(n) = h(n)` only. Faster but **not optimal** — may find a suboptimal path by greedily following the heuristic.

### Manhattan Distance
`h = |dx| + |dy|` — Admissible for 4-directional movement.

### Euclidean Distance  
`h = sqrt(dx² + dy²)` — Admissible for any movement type; typically tighter estimate.

---

## Dynamic Re-planning

When **Dynamic Mode** is enabled:
1. New wall obstacles spawn randomly every ~30 frames while the agent moves.
2. If a spawned wall lands **on the current path**, the agent immediately re-plans from its current position.
3. If the wall is **off the path**, no recalculation is needed (efficient).
4. The **Re-plans** counter tracks how many times the agent had to recalculate.

---

## Visualization Colors

| Color | Meaning |
|---|---|
| 🟢 Green | Calculated optimal path |
| 🔵 Blue | Visited/expanded nodes |
| 🟡 Yellow | Frontier nodes (in priority queue) |
| 🟠 Orange | Agent current position |
| 🔴 Red/Pink | Goal node |
| 🟩 Green (dark) | Start node |
| ⬛ Dark gray | Wall/obstacle |

---

## Dependencies

```
pygame>=2.0.0
```

All other modules (`heapq`, `math`, `random`, `time`, `collections`) are from the Python standard library.

---

## File Structure

```
pathfinding_agent/
├── main.py       # Full application (single-file)
└── README.md     # This file
```
