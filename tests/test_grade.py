"""Hermetic tests for the grade core.

These exercise graph construction and the layered features on a small synthetic
run. They need only networkx (the core dependency); no network access, no
dataset download, and no API keys. Run with ``pytest``.
"""
import networkx as nx
import pytest

from grade import (
    NODE_TYPES,
    EXECUTION_EDGES,
    DEPENDENCY_EDGES,
    build_graph,
    characterize,
    dependency_dag,
    downstream_reach,
    layered_features,
    feature_vector,
)

# A tiny three-step run with one observed dependency chain: step 2 -> 1 -> 0.
# Two agents ("a", "b") so the execution layer has a handoff between agents.
STEPS = [
    {"idx": 0, "agent": "a", "kind": "tool_call"},
    {"idx": 1, "agent": "a", "kind": "decision", "deps": [0]},
    {"idx": 2, "agent": "b", "kind": "tool_call", "deps": [1]},
]


def test_constants_are_typed():
    assert NODE_TYPES == ("agent", "decision", "tool_call", "dependency_resource")
    assert "emits" in EXECUTION_EDGES and "handoff_to" in EXECUTION_EDGES
    assert "depends_on" in DEPENDENCY_EDGES


def test_node_types_are_within_the_declared_set():
    G = build_graph(STEPS, dependency="explicit", shared_resource=False)
    ntypes = {d["ntype"] for _, d in G.nodes(data=True)}
    assert ntypes <= set(NODE_TYPES)
    n_agent = sum(1 for _, d in G.nodes(data=True) if d["ntype"] == "agent")
    n_step = sum(1 for _, d in G.nodes(data=True) if d["ntype"] in ("decision", "tool_call"))
    assert n_agent == 2  # agents "a" and "b"
    assert n_step == 3


def test_execution_and_dependency_layers_are_separate():
    G = build_graph(STEPS, dependency="explicit", shared_resource=False)
    etypes = [d["etype"] for _, _, d in G.edges(data=True)]
    # execution layer: one emits per step + handoffs between consecutive steps
    assert etypes.count("emits") == 3
    assert etypes.count("handoff_to") == 2
    # dependency layer: exactly the two observed depends_on edges
    assert etypes.count("depends_on") == 2


def test_explicit_keeps_only_named_prior_deps():
    G = build_graph(STEPS, dependency="explicit", shared_resource=False)
    dep = dependency_dag(G)
    assert nx.is_directed_acyclic_graph(dep)
    assert dep.number_of_edges() == 2
    # depends_on points dependent -> dependency
    assert dep.has_edge("step::2", "step::1")
    assert dep.has_edge("step::1", "step::0")


def test_full_context_infers_every_prior_edge():
    G = build_graph(STEPS, dependency="full_context", shared_resource=False)
    dep = dependency_dag(G)
    # step 1 depends on {0}; step 2 depends on {0, 1} -> 3 inferred edges
    assert dep.number_of_edges() == 3


def test_characterize_counts():
    G = build_graph(STEPS, dependency="explicit", shared_resource=False)
    c = characterize(G)
    assert c["n_steps"] == 3
    assert c["n_tool_calls"] == 2
    assert c["n_agents"] == 2
    assert c["n_dep_edges"] == 2
    assert c["dep_depth"] == 2  # longest depends_on path 2 -> 1 -> 0 has length 2


def test_downstream_reach_is_the_transitive_blast_radius():
    G = build_graph(STEPS, dependency="explicit", shared_resource=False)
    assert downstream_reach(G, 0) == 2  # steps 1 and 2 transitively rest on step 0
    assert downstream_reach(G, 2) == 0  # nothing rests on the last step


def test_feature_layers_are_nested_with_stable_order():
    G = build_graph(STEPS, dependency="explicit", shared_resource=False)
    names_flat, vals_flat = feature_vector(G, layer="flat")
    names_full, vals_full = feature_vector(G, layer="full")
    assert len(names_flat) == len(vals_flat) == 4
    assert len(names_full) == len(vals_full) == 12
    assert names_full[: len(names_flat)] == names_flat  # nested prefix order
    groups = layered_features(G)
    assert set(groups) == {"flat", "exec", "dep", "dep_dens", "dep_shape"}


def test_feature_vector_rejects_unknown_layer():
    G = build_graph(STEPS, dependency="explicit", shared_resource=False)
    with pytest.raises(ValueError):
        feature_vector(G, layer="nope")


def test_build_graph_rejects_duplicate_idx():
    bad = [
        {"idx": 0, "agent": "a", "kind": "decision"},
        {"idx": 0, "agent": "a", "kind": "decision"},
    ]
    with pytest.raises(ValueError):
        build_graph(bad)


def test_build_graph_rejects_non_int_idx():
    bad = [{"idx": "0", "agent": "a", "kind": "decision"}]
    with pytest.raises(ValueError):
        build_graph(bad)


def test_build_graph_rejects_invalid_dependency():
    with pytest.raises(ValueError):
        build_graph(STEPS, dependency="observed")  # not one of the three valid modes


def test_size_normalized_layers():
    G = build_graph(STEPS, dependency="explicit", shared_resource=False)
    for layer in ("flatdens", "flatdep", "depnorm"):
        names, values = feature_vector(G, layer=layer)
        assert len(names) == len(values) > 0
    # flatdep (the paper's primary set) = flat(4) + dep_dens(1) + dep_shape(3)
    names_fd, _ = feature_vector(G, layer="flatdep")
    assert len(names_fd) == 8
    # depnorm = dep_dens(1) + dep_shape(3)
    names_dn, _ = feature_vector(G, layer="depnorm")
    assert len(names_dn) == 4
