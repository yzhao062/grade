"""Generate the README figures for GRADE.

Three figures, each tied to a result in the paper:

  grade_overview.png  the two-layer graph: execution edges (read from the trace)
                      plus dependency edges (recovered and graded by source).
                      The structure is built with the real ``grade.build_graph``
                      API; the per-edge source labels illustrate how an adapter
                      grades a dependency edge (observed / declared / inferred).
  localization.png    step-level fault localization on Who&When: execution-graph
                      structure beats an early-fault position prior on every
                      metric (numbers match the localization table).
  transfer.png        leave-one-corpus-out transfer: size-normalized dependency
                      clears chance on every held-out corpus, while run size
                      inverts on two (numbers match the transfer table).

Needs matplotlib and networkx:  pip install matplotlib networkx
Run from anywhere:              python assets/make_figures.py
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.path import Path

# Make ``grade`` importable from a fresh checkout without an install.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from grade import build_graph  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Helvetica"]

# One palette across all three figures.
MINT, CORAL, GRAY = "#BFDFD2", "#ED8D5A", "#C9C9C9"
TEAL, LAVENDER = "#4098AC", "#B08FD0"
MINT_EDGE = "#8FB7A6"
NEAR_BLACK, SUBTITLE = "#1a1a1a", "#666666"


def save(fig, name):
    out = os.path.join(HERE, name)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Figure 1: the two-layer graph (hero)
# --------------------------------------------------------------------------- #
def make_overview():
    # A small multi-agent run. ``kind`` is decision/tool_call; ``deps`` lists the
    # prior steps each step actually relied on (the observed-dependency case).
    steps = [
        {"idx": 0, "agent": "planner", "kind": "decision"},
        {"idx": 1, "agent": "planner", "kind": "tool_call", "deps": [0]},
        {"idx": 2, "agent": "coder", "kind": "tool_call", "deps": [1]},
        {"idx": 3, "agent": "coder", "kind": "decision", "deps": [1, 2]},
        {"idx": 4, "agent": "tester", "kind": "tool_call", "deps": [2, 3]},
        {"idx": 5, "agent": "tester", "kind": "decision", "deps": [3, 4]},
    ]
    # Built by the actual library: step nodes, execution edges, dependency edges.
    G = build_graph(steps, dependency="explicit", shared_resource=False)
    handoffs = [(u, v) for u, v, d in G.edges(data=True) if d["etype"] == "handoff_to"]
    n = len(steps)
    xpos = {f"step::{s['idx']}": i for i, s in enumerate(steps)}

    # Illustrative grading of each dependency edge by how it is known. An adapter
    # assigns these; the representation carries the grade so weak edges are visible.
    DEPS = [
        (1, 0, "observed"), (2, 1, "observed"), (3, 1, "declared"),
        (3, 2, "observed"), (4, 2, "inferred"), (4, 3, "observed"),
        (5, 3, "declared"), (5, 4, "observed"),
    ]
    SRC_STYLE = {
        "observed": dict(color=CORAL, ls="-", lw=1.7),
        "declared": dict(color="#E0A87E", ls=(0, (5, 2)), lw=1.5),
        "inferred": dict(color="#9A9A9A", ls=(0, (1, 2)), lw=1.5),
    }
    AGENT_FILL = {"planner": "#CFE6EC", "coder": "#D6EBE0", "tester": "#E7DCF4"}

    fig, ax = plt.subplots(figsize=(7.4, 3.25), dpi=200)

    # Dependency layer: arcs bulging up, styled by source grade.
    for j, i, src in DEPS:
        xa, xb = xpos[f"step::{j}"], xpos[f"step::{i}"]
        h = 0.42 * abs(xb - xa) + 0.30
        verts = [(xa, 0), (xa, h), (xb, h), (xb, 0)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        ax.add_patch(mpatches.PathPatch(Path(verts, codes), fill=False,
                                        zorder=2, **SRC_STYLE[src]))

    # Execution layer: straight handoff arrows along the baseline.
    for u, v in handoffs:
        ax.annotate("", xy=(xpos[v], 0), xytext=(xpos[u], 0),
                    arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2.0,
                                    shrinkA=13, shrinkB=13), zorder=3)

    # Step nodes, filled by emitting agent.
    for s in steps:
        x = xpos[f"step::{s['idx']}"]
        ax.scatter(x, 0, s=560, color=AGENT_FILL[s["agent"]],
                   edgecolors=NEAR_BLACK, linewidth=1.1, zorder=4)
        ax.text(x, 0, f"s{s['idx']}", ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=NEAR_BLACK, zorder=5)
        ax.text(x, -0.52, s["agent"], ha="center", va="center",
                fontsize=6.6, color=SUBTITLE, zorder=5)

    # Layer labels in the left margin, stacked so they never overrun the nodes.
    ax.text(-1.6, 1.05, "dependency layer", fontsize=7.8, color="#9A4A33",
            style="italic", ha="left", va="center")
    ax.text(-1.6, 0.60, "recovered,\ngraded by source", fontsize=6.0,
            color=SUBTITLE, ha="left", va="center", linespacing=1.25)
    ax.text(-1.6, -0.95, "execution layer", fontsize=7.8, color=TEAL,
            style="italic", ha="left", va="center")
    ax.text(-1.6, -1.34, "read from\nthe trace", fontsize=6.0,
            color=SUBTITLE, ha="left", va="center", linespacing=1.25)

    ax.set_xlim(-1.75, n - 0.05)
    ax.set_ylim(-1.7, 2.55)
    ax.axis("off")

    ax.set_title("One run, one typed graph, two edge layers",
                 fontsize=11.5, fontweight="bold", pad=14, loc="left", x=0.0)

    layer_handles = [
        Line2D([0], [0], color=TEAL, lw=2.0, label="execution edge (handoff, from trace)"),
        Line2D([0], [0], color=CORAL, lw=1.7, label="dependency: observed (logged access)"),
        Line2D([0], [0], color="#E0A87E", lw=1.5, ls=(0, (5, 2)), label="dependency: declared"),
        Line2D([0], [0], color="#9A9A9A", lw=1.5, ls=(0, (1, 2)), label="dependency: inferred"),
    ]
    ax.legend(handles=layer_handles, loc="upper right", frameon=False,
              fontsize=6.8, labelspacing=0.4, handlelength=1.8,
              handletextpad=0.6, borderpad=0.2, ncol=1)
    save(fig, "grade_overview.png")


# --------------------------------------------------------------------------- #
# Figure 2: step-level fault localization (Who&When)
# --------------------------------------------------------------------------- #
def make_localization():
    METRICS = ["top-1", "top-3", "MRR"]
    RANDOM = [0.119, 0.346, 0.324]      # chance floor
    POSITION = [0.159, 0.516, 0.407]    # early-fault position prior
    STRUCTURE = [0.211, 0.614, 0.454]   # execution-graph structure

    fig, ax = plt.subplots(figsize=(5.4, 2.9), dpi=200)
    x = np.arange(len(METRICS))
    w = 0.26
    series = [
        (RANDOM, GRAY, "#AFAFAF", "random floor", False),
        (POSITION, MINT, MINT_EDGE, "position prior", False),
        (STRUCTURE, CORAL, NEAR_BLACK, "execution structure", True),
    ]
    for i, (vals, color, edge, label, bold) in enumerate(series):
        off = (i - 1) * w
        ax.bar(x + off, vals, w, color=color, edgecolor=edge,
               linewidth=0.9 if bold else 0.6, zorder=3, label=label)
        for xi, v in zip(x, vals):
            ax.text(xi + off, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=6.2, color=NEAR_BLACK,
                    fontweight="bold" if bold else "normal", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, fontsize=9)
    ax.set_ylim(0, 0.74)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6"], fontsize=7.5)
    ax.set_ylabel("score (higher is better)", fontsize=8.5, labelpad=2)
    ax.tick_params(axis="both", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#999999")
    ax.spines["bottom"].set_color("#999999")
    ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title("Execution structure localizes the faulting step",
                 fontsize=10.5, fontweight="bold", pad=16, loc="left", x=0.0)
    ax.text(0.0, 1.045, "Who&When multi-agent failures: structure beats an early-fault prior",
            transform=ax.transAxes, fontsize=7.0, color=SUBTITLE, ha="left")
    ax.legend(loc="upper left", frameon=False, fontsize=7.2, labelspacing=0.3,
              handlelength=1.2, handletextpad=0.5, borderpad=0.2)
    save(fig, "localization.png")


# --------------------------------------------------------------------------- #
# Figure 3: leave-one-corpus-out transfer
# --------------------------------------------------------------------------- #
def make_transfer():
    # label, flat (run size), dep (size-normalized dependency-only)
    DATA = [
        ("tau-bench\n(DB)", 0.468, 0.551),
        ("SWE-agent\n(coding)", 0.624, 0.646),
        ("SWE-Gym\n(coding)", 0.350, 0.584),
        ("OpenHands\n(coding)", 0.699, 0.595),
        ("tau2-bench\n(DB)", 0.725, 0.644),
        ("web\n(navigation)", 0.766, 0.662),
    ]
    CHANCE = 0.5
    BROWN = "#9A6B55"
    XLO, XHI = 0.30, 0.82

    fig, ax = plt.subplots(figsize=(5.6, 3.5), dpi=200)
    n = len(DATA)
    ys = [n - 1 - i for i in range(n)]

    ax.axvspan(XLO, CHANCE, color="#F2EAE4", zorder=0)
    ax.axvline(CHANCE, color="#BBBBBB", linewidth=1.0, linestyle=(0, (3, 3)), zorder=1.5)
    ax.text(CHANCE, n + 0.55, "chance", ha="center", va="bottom", fontsize=6.8, color="#777777")

    for i, (label, flat, dep) in enumerate(DATA):
        y = ys[i]
        below = flat < CHANCE
        ax.plot([flat, dep], [y, y], color="#C7C7C7", linewidth=1.4, zorder=2,
                solid_capstyle="round")
        ax.scatter(flat, y, s=80, color=MINT, zorder=3,
                   edgecolors=BROWN if below else MINT_EDGE,
                   linewidth=1.7 if below else 0.6)
        ax.scatter(dep, y, s=120, color=CORAL, edgecolors=NEAR_BLACK, linewidth=0.8, zorder=4)
        ax.text(flat, y - 0.24, f"{flat:.3f}", ha="center", va="top",
                fontsize=6.2, color=NEAR_BLACK, zorder=5)
        ax.text(dep, y + 0.24, f"{dep:.3f}", ha="center", va="bottom",
                fontsize=6.4, color=NEAR_BLACK, fontweight="bold", zorder=5)

    ax.text(0.40, 1.5, "run size below chance:\ntau-bench, SWE-Gym", ha="center",
            va="center", fontsize=6.4, color=BROWN, style="italic", linespacing=1.3, zorder=5)

    ax.set_xlim(XLO, XHI)
    ax.set_ylim(-0.7, n + 1.0)
    ax.set_xticks([0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
    ax.set_xticklabels([f"{t:.1f}" for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)], fontsize=7.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([d[0] for d in DATA], fontsize=7.5)
    ax.set_xlabel("Leave-one-corpus-out transfer ROC-AUC", fontsize=8.5)
    ax.tick_params(axis="both", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#999999")
    ax.spines["bottom"].set_color("#999999")
    ax.xaxis.grid(True, color="#E6E6E6", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title("Dependency transfers across classes; run size inverts on two",
                 fontsize=10.0, fontweight="bold", pad=18, loc="left", x=0.0)
    ax.text(0.0, 1.045, "Size-normalized dependency vs run size, fit on five corpora, scored on the held-out one",
            transform=ax.transAxes, fontsize=6.6, color=SUBTITLE, ha="left")

    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=MINT,
               markeredgecolor=MINT_EDGE, markersize=8, label="run size (flat)"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=CORAL,
               markeredgecolor=NEAR_BLACK, markersize=9, label="dependency"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.99),
              frameon=False, fontsize=7.2, labelspacing=0.35, handletextpad=0.5,
              borderpad=0.3)
    save(fig, "transfer.png")


if __name__ == "__main__":
    make_overview()
    make_localization()
    make_transfer()
