"""Diagnostic: why does the dependency layer help on tau-bench / SWE-agent but not on
OpenHands / AgentRewardBench? Is it a data limit, a feature-design (method) issue, or both?

For each observed-dependency corpus we load the same 12-feature matrix the keystone uses
(flat | exec | dep, 4 each) and ask three questions:

  1. DEGENERACY. Which features are (near-)constant within a corpus? The exec features are
     agent-level; on single-agent corpora (coding, web) agent_gini / n_agent_transitions /
     agent_recurrence collapse to 0 and max_agent_outdeg == n_steps, so "exec" adds nothing
     by construction. We report each feature's std and its |Pearson r| with n_steps.

  2. SIGNAL vs SIZE. For every feature, its single-feature ROC-AUC for the failure label,
     next to n_steps's own AUC. If the dep features only match the label as well as raw size
     does, and they are collinear with size, they carry no INDEPENDENT signal.

  3. THE DECISIVE TEST. The dep features are raw counts (n_dep_edges, dep_depth, ...) that
     scale with n_steps. We rebuild a SIZE-NORMALIZED dep block (each dep count / n_steps)
     and re-run the full-vs-exec ablation. If normalized deps recover a gain where raw deps
     did not, the null was a METHOD artifact (size-confounded features). If normalized deps
     still add nothing, the dependency STRUCTURE genuinely lacks failure signal on that
     corpus -- a DATA limit.

Run:  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python experiment/diagnose_layers.py
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import agent_failure_detection as K

FEATS = ["n_steps", "n_tool_calls", "n_decisions", "n_agents",          # flat 0-3
         "max_agent_outdeg", "agent_gini", "n_agent_transitions",       # exec 4-6
         "agent_recurrence",                                            # exec 7
         "n_dep_edges", "dep_depth", "max_blast_indeg", "max_audit_outdeg"]  # dep 8-11
SEEDS = range(5)
_T = 2.776  # t(.975, df=4)


def _matrix(graphs):
    return np.array([K.feature_vector(G, layer="full")[1] for G in graphs], dtype=float)


def _cv(X, y, cols):
    rows = []
    for s in SEEDS:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(solver="liblinear", class_weight="balanced"))
        rows.append(cross_val_score(clf, X[:, cols], y, cv=cv, scoring="roc_auc"))
    return np.array(rows)


def _single_auc(x, y):
    """Direction-free single-feature AUC; constant feature -> 0.5."""
    if np.std(x) == 0:
        return 0.5
    a = roc_auc_score(y, x)
    return max(a, 1 - a)


def diagnose(name, graphs, labels):
    y = np.array(labels)
    X = _matrix(graphs)
    n_steps = X[:, 0]
    print(f"\n{'='*78}\n{name}: {len(y)} runs, {int(y.sum())} failed ({y.mean():.0%})\n{'='*78}")

    print(f"  {'feature':18s} {'std':>9s} {'|r|n_steps':>10s} {'1-feat AUC':>11s}")
    for j, fname in enumerate(FEATS):
        x = X[:, j]
        r = 0.0 if np.std(x) == 0 else abs(np.corrcoef(x, n_steps)[0, 1])
        marker = "  <-flat" if j < 4 else ("  <-exec" if j < 8 else "  <-DEP")
        print(f"  {fname:18s} {np.std(x):9.3f} {r:10.2f} {_single_auc(x, y):11.3f}{marker}")

    # decisive test: raw dep block vs size-normalized dep block
    Xn = X.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        for j in (8, 9, 10, 11):
            Xn[:, j] = np.where(n_steps > 0, X[:, j] / n_steps, 0.0)

    def cvm(Xmat, cols):
        return _cv(Xmat, y, cols).mean(axis=1)

    def ci(g):
        return g.mean(), _T * g.std(ddof=1) / np.sqrt(len(g)), int((g > 0).sum())

    flat = cvm(X, [0, 1, 2, 3])
    fullraw = cvm(X, list(range(12)))
    fullnorm = cvm(Xn, list(range(12)))
    flatdepraw = cvm(X, [0, 1, 2, 3, 8, 9, 10, 11])
    flatdepnorm = cvm(Xn, [0, 1, 2, 3, 8, 9, 10, 11])
    execc = cvm(X, list(range(8)))

    print(f"  flat (size only)            : {flat.mean():.3f}")
    print(f"  flat + exec                 : {execc.mean():.3f}   (exec degenerate if single-agent)")
    g = ci(fullraw - execc)
    print(f"  full  (raw dep)             : {fullraw.mean():.3f}   full-exec {g[0]:+.3f} [{g[0]-g[1]:+.3f},{g[0]+g[1]:+.3f}] ({g[2]}/5)")
    g = ci(fullnorm - execc)
    print(f"  full  (norm dep)            : {fullnorm.mean():.3f}   full-exec {g[0]:+.3f} [{g[0]-g[1]:+.3f},{g[0]+g[1]:+.3f}] ({g[2]}/5)")
    g = ci(flatdepraw - flat)
    print(f"  flat + dep (raw, no exec)   : {flatdepraw.mean():.3f}   vs-flat  {g[0]:+.3f} [{g[0]-g[1]:+.3f},{g[0]+g[1]:+.3f}] ({g[2]}/5)")
    g = ci(flatdepnorm - flat)
    print(f"  flat + dep (norm, no exec)  : {flatdepnorm.mean():.3f}   vs-flat  {g[0]:+.3f} [{g[0]-g[1]:+.3f},{g[0]+g[1]:+.3f}] ({g[2]}/5)   <- does dependency STRUCTURE beat size?")


def main():
    print("Layer diagnostic: degeneracy, size-confounding, and a size-normalized dep test.")
    diagnose("tau-bench (multi-agent DB tool-use)", *K.load_tau())
    diagnose("SWE-agent (single-agent coding)", *K.load_swe())
    diagnose("OpenHands (single-agent coding)", *K.load_openhands())
    diagnose("AgentRewardBench (single-agent web)", *K.load_web())
    print("\nReadout: exec features near-constant on single-agent corpora confirms the exec "
          "layer is agent-level (degenerate for one agent). If NORM dep recovers a positive "
          "gain where RAW dep was null (OpenHands, web), the null was a size-confounded "
          "FEATURE artifact, not absent dependency signal; if NORM dep is still null, the "
          "dependency structure genuinely lacks run-failure signal on that corpus.")


if __name__ == "__main__":
    main()
