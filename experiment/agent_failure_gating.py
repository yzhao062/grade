"""Gating experiment: does the attachment grade (observed vs inferred dependency)
change what the dependency layer is worth? Tests the paper's Section 4.3.4 gating
prescription -- admit dependency-shape features only where the saturation ratio
rho = n_dep / C(n_steps,2) is well below 1 (the non-inferred / unsaturated regime).

For each of the six observed-dependency corpora we build TWO dependency layers from
the SAME steps: the OBSERVED layer (logged reads/writes, dependency='explicit') and
the full-history INFERRED layer (dependency='full_context', the A_0 assumption that
every step depended on all earlier steps). We compare three feature sets under the
same 5-seed 5-fold within-corpus CV and pooled leave-one-corpus-out transfer:
  flat          : run size and counts (dependency-independent)
  flat+dep(obs) : flat + size-normalized OBSERVED dependency shape
  flat+dep(inf) : flat + size-normalized INFERRED full-history dependency shape
plus the per-run median rho for each layer.

Hypothesis (the degenerate regime): the inferred dependency shape collapses to
constants + run size (rel_dep_depth, blast/audit shares -> 1; edges/step -> (n-1)/2),
so flat+dep(inf) ~= flat and inverts in transfer just as run size does, while
flat+dep(obs) carries transferable signal. rho separates the regimes (obs ~0.01,
inf = 1.0): it is the gate the paper prescribes.

Run:  python experiment/agent_failure_gating.py
"""
import os
import sys
from math import comb

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from grade import build_graph, feature_vector, layered_features

import agent_graph_swe_agent as swe
import agent_graph_tau_bench as tau
import agent_graph_openhands as oh
import agent_graph_swegym as swegym
import agent_reward_bench as web
import agent_graph_tau2_bench as tau2

SEEDS = range(5)


def _rho(G):
    lf = layered_features(G)
    n = int(lf["flat"]["n_steps"])
    nd = int(lf["dep"]["n_dep_edges"])
    return (nd / comb(n, 2)) if n >= 2 else 0.0


def _corpus(name, pairs):
    flat, deponly, obs, inf, y, rho_obs, rho_inf = [], [], [], [], [], [], []
    for steps, label in pairs:
        if len(steps) < 2:
            continue
        g_obs = build_graph(steps, dependency="explicit", shared_resource=False)
        g_inf = build_graph(steps, dependency="full_context", shared_resource=False)
        flat.append(feature_vector(g_obs, layer="flat")[1])
        deponly.append(feature_vector(g_obs, layer="depnorm")[1])
        obs.append(feature_vector(g_obs, layer="flatdep")[1])
        inf.append(feature_vector(g_inf, layer="flatdep")[1])
        y.append(label)
        rho_obs.append(_rho(g_obs))
        rho_inf.append(_rho(g_inf))
    return {
        "name": name,
        "flat": np.array(flat, float),
        "deponly": np.array(deponly, float),
        "obs": np.array(obs, float),
        "inf": np.array(inf, float),
        "y": np.array(y),
        "rho_obs": float(np.median(rho_obs)) if rho_obs else float("nan"),
        "rho_inf": float(np.median(rho_inf)) if rho_inf else float("nan"),
    }


def _cv(X, y):
    rows = []
    for s in SEEDS:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(solver="liblinear", class_weight="balanced"))
        rows.append(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc"))
    return float(np.mean(rows))


def _transfer_scores(trX, trY, teX):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(solver="liblinear", class_weight="balanced"))
    clf.fit(trX, trY)
    return clf.decision_function(teX)


def _auc_ci_p(y, scores, n_boot=2000, seed=0):
    """Point AUC, a test-set bootstrap 95% CI, and a one-sided p-value for AUC > 0.5.

    The LOCO transfer fit is deterministic (liblinear on the pooled train set), so the
    uncertainty that matters is sampling of the held-out corpus: we resample the test
    runs with replacement for the CI. AUC > 0.5 is tested with a one-sided Mann-Whitney U
    on the decision scores, since U / (n_pos * n_neg) == AUC, so this is the exact rank
    test that 'clears chance' needs. (Several dep-only transfer
    values -- 0.551, 0.584, 0.595 -- hug the chance line and were reported point-only.)
    """
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    auc = float(roc_auc_score(y, scores))
    pos, neg = scores[y == 1], scores[y == 0]
    p = (float(mannwhitneyu(pos, neg, alternative="greater").pvalue)
         if len(pos) and len(neg) else float("nan"))
    rng = np.random.default_rng(seed)
    n, boots = len(y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() != yb.max():
            boots.append(roc_auc_score(yb, scores[idx]))
    lo, hi = ((float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
              if boots else (float("nan"), float("nan")))
    return auc, lo, hi, p


def load_all():
    C = {}
    print("loading tau-bench...", flush=True)
    paths = tau._ensure_files()
    C["tau-bench"] = _corpus("tau-bench", (
        (tau.to_steps(rec["messages"], rec.get("model_path", "model")),
         0 if bool((rec.get("eval_result") or {}).get("db_match")) else 1)
        for rec in tau.load_runs(paths)
        if isinstance(rec.get("messages"), list) and len(rec["messages"]) >= 3
        and "db_match" in (rec.get("eval_result") or {})))
    print("loading SWE-agent...", flush=True)
    C["SWE-agent"] = _corpus("SWE-agent", (
        (swe.to_steps(rec["traj"]), 0 if rec["target"] else 1)
        for rec in swe.load_runs(swe.N_RUNS)))
    print("loading SWE-Gym...", flush=True)
    C["SWE-Gym"] = _corpus("SWE-Gym", (
        (rec["steps"], 0 if rec["resolved"] else 1)
        for rec in swegym.load_runs()))
    print("loading OpenHands...", flush=True)
    C["OpenHands"] = _corpus("OpenHands", (
        (oh.to_steps(rec["traj"]), 0 if rec["resolved"] else 1)
        for rec in oh.load_runs(oh.N_RUNS)))
    print("loading tau2-bench...", flush=True)
    C["tau2-bench"] = _corpus("tau2-bench", (
        (rec["steps"], rec["label"]) for rec in tau2.load_runs()))
    print("loading web...", flush=True)
    C["web"] = _corpus("web", (
        (web.to_steps(rec["steps"]), rec["label"]) for rec in web.load_runs(web.N_RUNS)))
    return C


def main():
    C = load_all()
    order = ["tau-bench", "SWE-agent", "SWE-Gym", "OpenHands", "tau2-bench", "web"]

    print("\n=== WITHIN-CORPUS (5-seed 5-fold ROC-AUC) ===", flush=True)
    print("%-12s %6s %8s %10s %10s %8s %8s" % ("corpus", "flat", "dep-only", "+dep(obs)",
                                               "+dep(inf)", "rho_obs", "rho_inf"), flush=True)
    for nm in order:
        d = C[nm]
        print("%-12s %6.3f %8.3f %10.3f %10.3f %8.3f %8.3f"
              % (nm, _cv(d["flat"], d["y"]), _cv(d["deponly"], d["y"]), _cv(d["obs"], d["y"]),
                 _cv(d["inf"], d["y"]), d["rho_obs"], d["rho_inf"]), flush=True)

    print("\n=== POOLED LEAVE-ONE-CORPUS-OUT TRANSFER (ROC-AUC, test-set bootstrap "
          "95% CI, one-sided p vs 0.5; * = p<0.05) ===", flush=True)
    labels = {"flat": "flat", "deponly": "dep-only", "obs": "+dep(obs)", "inf": "+dep(inf)"}
    print("%-12s %-10s %7s   %-16s %9s" % ("held-out", "feat", "AUC", "95% CI", "p(>0.5)"),
          flush=True)
    for held in order:
        tr = [nm for nm in order if nm != held]
        for key in ("flat", "deponly", "obs", "inf"):
            trX = np.vstack([C[nm][key] for nm in tr])
            trY = np.concatenate([C[nm]["y"] for nm in tr])
            scores = _transfer_scores(trX, trY, C[held][key])
            auc, lo, hi, p = _auc_ci_p(C[held]["y"], scores)
            star = "*" if (p == p and p < 0.05) else " "
            print("%-12s %-10s %7.3f   [%5.3f, %5.3f] %7.3f %s"
                  % (held, labels[key], auc, lo, hi, p, star), flush=True)


if __name__ == "__main__":
    main()
