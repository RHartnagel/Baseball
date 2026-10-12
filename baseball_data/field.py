"""Draw a baseball field in Statcast-style feet (home plate at origin, +y toward CF, +x toward RF)."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Wedge

BASE = 90 / np.sqrt(2)  # a base's x and y offset from home plate, ft
MOUND_Y = 60.5
WALL_FT = 380  # generic fence radius; real parks vary


def plot_field(ax=None, wall=WALL_FT):
    """Draw grass, infield dirt, bases, foul lines and a generic fence. Returns the axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    ax.add_patch(Wedge((0, 0), wall, 45, 135, facecolor="#7fb36b", edgecolor="none", zorder=0))
    ax.add_patch(Circle((0, MOUND_Y), 95, facecolor="#d9b98a", edgecolor="none", zorder=1))
    ax.add_patch(Polygon([(0, 0), (BASE, BASE), (0, 2 * BASE), (-BASE, BASE)], closed=True,
                         facecolor="#7fb36b", edgecolor="white", linewidth=1.5, zorder=2))
    ax.add_patch(Circle((0, MOUND_Y), 9, facecolor="#d9b98a", edgecolor="none", zorder=3))
    for x, y in ((BASE, BASE), (0, 2 * BASE), (-BASE, BASE)):
        ax.add_patch(Polygon([(x - 4, y), (x, y + 4), (x + 4, y), (x, y - 4)], facecolor="white", zorder=4))
    ax.add_patch(Polygon([(-4, 0), (4, 0), (4, -3), (0, -6), (-4, -3)], facecolor="white", zorder=4))
    for sign in (-1, 1):
        ax.plot([0, sign * wall / np.sqrt(2)], [0, wall / np.sqrt(2)], color="white", linewidth=1.5, zorder=2)
    t = np.radians(np.linspace(45, 135, 100))
    ax.plot(wall * np.cos(t), wall * np.sin(t), color="#444", linewidth=2, zorder=2)
    ax.set_xlim(-wall * 0.78, wall * 0.78)
    ax.set_ylim(-25, wall + 10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
