"""Keystone experiment: does the two-layer graph predict agent failure better than
its execution-only or flat (size) projections?

For each labeled corpus we build the typed decision graph, extract nested feature
sets with ``grade.feature_vector`` (flat -> +execution topology ->
+dependency layer), and measure how well a simple detector separates failed from
successful runs. If the full features beat the flat baseline the typed structure is
informative (not merely expressible); and where the dependency layer beats
execution-only, that layer earns its place.

Corpora (label 1 = a FAILED run):
  - tau-bench        : db_match,        dependency OBSERVED
  - SWE-agent        : issue resolved,  dependency OBSERVED  (heavily imbalanced)
  - SWE-Gym          : issue resolved,  dependency OBSERVED
  - OpenHands        : issue resolved,  dependency OBSERVED
  - AgentRewardBench : expert label,    dependency OBSERVED
  - tau2-bench       : task success,    dependency OBSERVED
  - Who&When         : is_correct,      dependency INFERRED  -- a failure-only corpus, so
                       run-level detection is inapplicable (reported as skipped)

Method: 5 seeds x 5-fold stratified CV, standardized logistic regression (liblinear;
the default lbfgs solver segfaults under this Windows conda BLAS stack), balanced
class weights, ROC-AUC. The flat / exec / full feature sets are nested, so EXEC and
FLAT are column prefixes of the FULL matrix and every fold split is shared across
layers. Adjacent-layer gains are reported as the mean per-seed gain with a seed-block
95% CI: the 5 seeds are the independent-ish unit, while the 5 folds within a seed
share runs and are not independent, so a fold-level test would overstate significance.

Run:  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python experiment/agent_failure_detection.py
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from grade import build_graph, feature_vector

import agent_graph_characterization as ww
import agent_graph_swe_agent as swe
import agent_graph_tau_bench as tau
import agent_graph_openhands as oh
import agent_graph_swegym as swegym
import agent_reward_bench as web
import agent_graph_scienceworld as sci
import agent_graph_tau2_bench as tau2

SEEDS = range(5)
_T975_4 = 2.776  # t(0.975, df=4): seed-block 95% CI half-width over 5 seeds


# --- corpus loaders: each returns (graphs, labels) with label 1 = failed ---------

def load_whoandwhen():
    ww._ensure_corpus()
    graphs, labels = [], []
    for t in ww.load_tasks():
        steps = ww.to_steps(t)
        if len(steps) < 2:
            continue
        ic = str(t.get("is_correct", "")).lower()
        if ic not in ("true", "false"):
            continue  # unlabeled run: cannot use for detection
        graphs.append(build_graph(steps, dependency="full_context"))
        labels.append(1 if ic == "false" else 0)
    return graphs, labels


def load_tau():
    paths = tau._ensure_files()
    graphs, labels = [], []
    for rec in tau.load_runs(paths):
        msgs = rec.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 3:
            continue
        er = rec.get("eval_result") or {}
        if "db_match" not in er:
            continue  # no gold label for this run
        steps = tau.to_steps(msgs, rec.get("model_path", "model"))
        if len(steps) < 2:
            continue
        graphs.append(build_graph(steps, dependency="explicit", shared_resource=False))
        labels.append(0 if bool(er["db_match"]) else 1)
    return graphs, labels


def load_swe():
    graphs, labels = [], []
    for rec in swe.load_runs(swe.N_RUNS):
        steps = swe.to_steps(rec["traj"])
        if len(steps) < 2:
            continue
        graphs.append(build_graph(steps, dependency="explicit", shared_resource=False))
        labels.append(0 if rec["target"] else 1)  # target = issue resolved
    return graphs, labels


def load_openhands():
    graphs, labels = [], []
    for rec in oh.load_runs(oh.N_RUNS):
        steps = oh.to_steps(rec["traj"])
        if len(steps) < 2:
            continue
        graphs.append(build_graph(steps, dependency="explicit", shared_resource=False))
        labels.append(0 if rec["resolved"] else 1)  # resolved = issue solved
    return graphs, labels


def load_web():
    graphs, labels = [], []
    for rec in web.load_runs(web.N_RUNS):
        steps = web.to_steps(rec["steps"])
        if len(steps) < 2:
            continue
        graphs.append(build_graph(steps, dependency="explicit", shared_resource=False))
        labels.append(rec["label"])  # expert label, already 1 = failed
    return graphs, labels


def load_swegym():
    graphs, labels = [], []
    for rec in swegym.load_runs():  # already balanced + projected to steps
        steps = rec["steps"]
        if len(steps) < 2:
            continue
        graphs.append(build_graph(steps, dependency="explicit", shared_resource=False))
        labels.append(0 if rec["resolved"] else 1)  # resolved = issue solved
    return graphs, labels


def load_scienceworld():
    graphs, labels = [], []
    for rec in sci.load_runs():  # embodied; balanced within source; label 1 = failed
        steps = rec["steps"]
        if len(steps) < 2:
            continue
        graphs.append(build_graph(steps, dependency="explicit", shared_resource=False))
        labels.append(rec["label"])
    return graphs, labels


def load_tau2():
    graphs, labels = [], []
    for rec in tau2.load_runs():  # DB; single balanced model; label 1 = failed
        steps = rec["steps"]
        if len(steps) < 2:
            continue
        graphs.append(build_graph(steps, dependency="explicit", shared_resource=False))
        labels.append(rec["label"])
    return graphs, labels


# --- evaluation ------------------------------------------------------------------

_LAYERS = ("flat", "exec", "full", "flatdens", "flatdep")


def _layer_matrices(graphs):
    """One feature matrix per layer (columns differ); rows aligned across layers."""
    cols = {L: [] for L in _LAYERS}
    for G in graphs:
        for L in _LAYERS:
            cols[L].append(feature_vector(G, layer=L)[1])
    return {L: np.array(v, dtype=float) for L, v in cols.items()}


def _cv(X, y):
    """ROC-AUC per (seed, fold); the seed splits are shared across layers."""
    rows = []
    for s in SEEDS:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(solver="liblinear", class_weight="balanced"))
        rows.append(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc"))
    return np.array(rows)  # shape (n_seeds, n_folds)


def _gain(hi, lo):
    g = hi.mean(axis=1) - lo.mean(axis=1)  # per-seed mean gain (len n_seeds)
    half = _T975_4 * g.std(ddof=1) / np.sqrt(len(g))
    return g.mean(), half, int((g > 0).sum()), len(g)


def evaluate(name, graphs, labels, observed_dep):
    y = np.array(labels)
    n, nf = len(y), int(y.sum())
    tag = "observed dependency" if observed_dep else "INFERRED dependency"
    print(f"\n=== {name}: {n} runs, {nf} failed ({nf / n:.0%} base rate), {tag} ===")
    if nf < 5 or n - nf < 5:
        print("  one class too small for a stable AUC; run-level detection inapplicable")
        return
    M = {L: _cv(X, y) for L, X in _layer_matrices(graphs).items()}

    def auc(L):
        a = M[L].mean(axis=1)
        return a.mean(), _T975_4 * a.std(ddof=1) / np.sqrt(len(a))

    # PRIMARY contrast: does dependency STRUCTURE (size-normalized) beat run size alone?
    # This is the honest test. The exec block is agent/env protocol alternation on
    # single-agent corpora (near-constant across runs), so full - exec compares against a
    # degenerate baseline and is demoted to a secondary diagnostic below.
    fa, fh = auc("flat")
    na, nh = auc("flatdens")
    da, dh = auc("flatdep")
    m, half, pos, k = _gain(M["flatdep"], M["flat"])
    # CONTROL: is the dependency gain just revisit DENSITY, or does
    # graph SHAPE add on top? flatdens = flat + dep_edges_per_step (bare revisit rate);
    # flatdep adds rel_dep_depth + hub shares. shape|dens isolates the structure claim.
    ms, hs, ps, _ = _gain(M["flatdep"], M["flatdens"])
    print(f"  flat (size)            : ROC-AUC {fa:.3f} +/-{fh:.3f}")
    print(f"  flat + dep density     : ROC-AUC {na:.3f} +/-{nh:.3f}   (revisit rate only)")
    print(f"  flat + dep (norm)      : ROC-AUC {da:.3f} +/-{dh:.3f}   (density + shape)")
    print(f"  PRIMARY gain dep|size  : {m:+.3f}  95% CI [{m - half:+.3f}, {m + half:+.3f}]"
          f"  ({pos}/{k} seeds +)   <- does dependency beat size?")
    print(f"  CONTROL gain shape|dens: {ms:+.3f}  95% CI [{ms - hs:+.3f}, {ms + hs:+.3f}]"
          f"  ({ps}/{k} seeds +)   <- does graph shape add beyond bare revisit density?")
    ea, _ = auc("exec")
    ua, _ = auc("full")
    me, he, pe, _ = _gain(M["full"], M["exec"])
    print(f"  [secondary nested] flat+exec {ea:.3f}, full(raw dep) {ua:.3f}; "
          f"full-exec {me:+.3f} [{me - he:+.3f},{me + he:+.3f}] ({pe}/5)")


def main():
    print("Keystone: does dependency STRUCTURE predict agent failure beyond run size?")
    print("Primary contrast: flat (size/counts) vs flat + size-normalized dependency block.")
    print("label 1 = failed run; 5 seeds x 5-fold CV; seed-block 95% CI.")
    evaluate("tau-bench", *load_tau(), observed_dep=True)
    evaluate("SWE-agent", *load_swe(), observed_dep=True)
    evaluate("SWE-Gym", *load_swegym(), observed_dep=True)
    evaluate("OpenHands", *load_openhands(), observed_dep=True)
    evaluate("AgentRewardBench", *load_web(), observed_dep=True)
    evaluate("tau2-bench", *load_tau2(), observed_dep=True)
    evaluate("Who&When", *load_whoandwhen(), observed_dep=False)
    print("\nReading. The primary test is whether size-normalized dependency structure "
          "adds failure signal over run size alone (flat vs flat+dep). In these four "
          "observed-dependency corpora, it does on two and not on the other two, and the "
          "split lines up with how strong size already is: dependency structure helps on "
          "tau-bench and SWE-agent, where the flat baseline is weak (AUC ~0.58-0.63), and "
          "is null on OpenHands and AgentRewardBench, where run size alone already "
          "predicts failure (AUC ~0.70-0.77). We report this as an observed pattern in "
          "these corpora, not a universal threshold. The nested exec/full numbers are a "
          "secondary diagnostic only: on single-agent corpora the exec block encodes "
          "assistant/environment protocol alternation (near-constant across runs), not "
          "agent topology, so full-exec compares against a degenerate baseline and is not "
          "the headline. Limit: whether INFERRED dependency helps is untested at run "
          "level, since the one inferred corpus (Who&When) is failure-only.")


if __name__ == "__main__":
    main()
