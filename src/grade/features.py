"""Layered structural features for an agent decision graph.

Three nested feature groups, matching the two-layer view of the representation:

  - ``flat``: size and counts only (no structure).
  - ``exec``: + execution-graph topology read from the agent / handoff structure
    (who acts, how concentrated, how control moves and returns). Deliberately uses
    agent-level structure, not the step chain, because the step handoff chain is
    just the run order and is collinear with ``n_steps``.
  - ``dep``: + dependency-layer structure from the ``depends_on`` DAG (how deep the
    chains run, the largest audit surface, the largest blast hub).

A detector built on ``flat + exec + dep`` that beats one on ``flat`` or
``flat + exec`` shows the typed structure carries signal beyond run size and
isolates which layer contributes. The groups are nested so adjacent ablations
compare strict feature prefixes.

Honest caveat about the dependency group: it carries independent signal only where
the dependency edges are OBSERVED and sparse (tool I/O, file reads). Where they are
INFERRED under a full-context assumption (every step depends on every prior step),
``dep_depth`` and ``n_dep_edges`` are functions of ``n_steps`` and add nothing the
flat group does not already have. The failure-detection study tests the observed
case on two corpora; it does not test the inferred case at run level, because the
available inferred-dependency corpus (Who&When) is failure-only.
"""
from __future__ import annotations

from collections import Counter


def _gini(values) -> float:
    """Gini concentration of a list of non-negative counts; 0 for empty/uniform/singleton."""
    vals = [v for v in values if v is not None]
    if len(vals) <= 1:
        return 0.0
    total = sum(vals)
    if total == 0:
        return 0.0
    n = len(vals)
    diffs = sum(abs(a - b) for a in vals for b in vals)
    return diffs / (2.0 * n * total)


def layered_features(G) -> dict:
    """Return named structural features grouped into ``flat`` / ``exec`` / ``dep``.

    The graph is a typed ``MultiDiGraph`` from :func:`grade.build_graph`:
    agent nodes emit step nodes, step nodes form a handoff chain, and ``depends_on``
    edges (observed or inferred) form the dependency layer.
    """
    try:
        import networkx as nx
    except Exception:  # pragma: no cover - exercised only where the extra is absent
        raise ImportError("grade requires networkx: pip install networkx")
    from grade import dependency_dag  # lazy import avoids any package cycle

    step_nodes = [(d["idx"], d["agent"]) for _, d in G.nodes(data=True)
                  if d.get("ntype") in ("decision", "tool_call")]
    step_nodes.sort(key=lambda t: t[0])
    agent_seq = [a for _, a in step_nodes]
    n_steps = len(agent_seq)
    n_tool = sum(1 for _, d in G.nodes(data=True) if d.get("ntype") == "tool_call")
    n_agents = sum(1 for _, d in G.nodes(data=True) if d.get("ntype") == "agent")

    flat = {
        "n_steps": n_steps,
        "n_tool_calls": n_tool,
        "n_decisions": n_steps - n_tool,
        "n_agents": n_agents,
    }

    # execution topology: agent activity concentration and how control moves / returns
    counts = Counter(agent_seq)
    changes = sum(1 for i in range(1, n_steps) if agent_seq[i] != agent_seq[i - 1])
    transitions, returns, seen = set(), 0, set()
    for i, a in enumerate(agent_seq):
        if i and a != agent_seq[i - 1]:
            transitions.add((agent_seq[i - 1], a))
            if a in seen:
                returns += 1
        seen.add(a)
    exec_ = {
        "max_agent_outdeg": max(counts.values()) if counts else 0,
        "agent_gini": _gini(list(counts.values())),
        "n_agent_transitions": len(transitions),
        "agent_recurrence": (returns / changes) if changes else 0.0,
    }

    # dependency layer: shape of the depends_on DAG (step -> what it relied on)
    dep = dependency_dag(G)
    if dep.number_of_nodes():
        indeg = [d for _, d in dep.in_degree()]
        outdeg = [d for _, d in dep.out_degree()]
        dep_depth = nx.dag_longest_path_length(dep) if nx.is_directed_acyclic_graph(dep) else 0
    else:
        indeg, outdeg, dep_depth = [0], [0], 0
    # depends_on points dependent -> dependency: a step's out-degree is its audit
    # surface (what it rests on), a step's in-degree is its blast (who rests on it).
    dep_ = {
        "n_dep_edges": dep.number_of_edges(),
        "dep_depth": dep_depth,
        "max_blast_indeg": max(indeg),
        "max_audit_outdeg": max(outdeg),
    }

    # Size-normalized dependency block: the raw counts above are collinear with
    # n_steps (n_dep_edges has |r| up to 0.96 with n_steps on web), so they re-encode
    # run size, which the flat group already carries. Dividing by n_steps changes the
    # estimand from absolute dependency volume/depth to dependency STRUCTURE conditional
    # on run size, which is the quantity the "signal beyond size" claim is about. These
    # are distinct features (events per step, relative chain length, hub shares), kept
    # separate from the raw dep_ group rather than overwriting it.
    ns = max(n_steps, 1)
    # Split the size-normalized dependency block into DENSITY and SHAPE so the
    # ablation can ask whether dependency STRUCTURE adds anything beyond the bare
    # revisit rate. dep_dens is how often steps re-touch state (repeated-resource
    # edges per step), the quantity a composition/density baseline already knows.
    # dep_shape is the topology of those edges: how deep the chains run and how
    # concentrated the hubs are. If dep_shape adds signal over flat+dep_dens, the
    # gain is graph structure, not just "failed runs revisit more".
    dep_dens = {
        "dep_edges_per_step": dep.number_of_edges() / ns,
    }
    dep_shape = {
        "rel_dep_depth": dep_depth / ns,
        "max_blast_share": max(indeg) / ns,
        "max_audit_share": max(outdeg) / ns,
    }

    return {"flat": flat, "exec": exec_, "dep": dep_, "dep_dens": dep_dens, "dep_shape": dep_shape}


def feature_vector(G, *, layer: str = "full"):
    """Flatten :func:`layered_features` into an ordered (names, values) pair.

    ``layer`` selects the column set. Nested size groups: ``"flat"`` (size and
    counts), ``"exec"`` (flat + execution topology), ``"full"`` (flat + exec +
    raw dependency counts). Size-normalized variants: ``"flatdens"`` (flat + bare
    revisit density), ``"flatdep"`` (flat + size-normalized dependency density and
    shape; the primary "signal beyond run size" set used in the paper), and
    ``"depnorm"`` (size-normalized dependency only, for dependency-only transfer).
    Names are returned so callers can keep a stable column order across runs.
    """
    groups = layered_features(G)
    order = {
        "flat": ["flat"],                                   # size and counts only
        "exec": ["flat", "exec"],                           # + execution topology (secondary)
        "full": ["flat", "exec", "dep"],                    # + raw dependency counts (legacy nested view)
        "flatdens": ["flat", "dep_dens"],                   # + bare revisit density (control baseline)
        "flatdep": ["flat", "dep_dens", "dep_shape"],       # + density + shape (size-normalized dependency, PRIMARY)
        "depnorm": ["dep_dens", "dep_shape"],               # size-normalized dependency only, no flat (Table 3 dep-only)
    }
    if layer not in order:
        raise ValueError(f"layer must be one of {sorted(order)}, got {layer!r}")
    names, values = [], []
    for g in order[layer]:
        for k, v in groups[g].items():
            names.append(k)
            values.append(float(v))
    return names, values
