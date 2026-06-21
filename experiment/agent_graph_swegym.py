"""Second weak-coding corpus: SWE-Gym sampled OpenHands trajectories (observed per-file).

A second software-engineering corpus on the LOW-flat side of the boundary, to test whether
the SWE-agent result (dependency structure beats run size where size is a weak predictor)
replicates on independent weak-model coding data. SWE-Gym/OpenHands-Sampled-Trajectories is
a mixed rollout dump (gpt-4o + claude-3.5-sonnet under OpenHands on SWE-Gym instances) with
a per-run ``resolved`` label; pass rate is low (~8%), so we downsample the unresolved
majority to a balanced subset. The trajectory format is OpenHands, so the per-file
dependency parsing is reused verbatim from ``agent_graph_openhands``.

Caveats handled here: the split is ``train.raw`` (not ``train``); the trajectory column is
``messages``; tool-call ``arguments`` is a JSON string (decoded by the reused parser); rows
are projected to lightweight typed steps at load time so the full message lists (with tool
outputs) are not pinned in memory.

Run:  python experiment/agent_graph_swegym.py
"""
import json

from grade import build_graph, characterize
from agent_graph_openhands import to_steps  # identical OpenHands per-file dependency model

REPO = "SWE-Gym/OpenHands-Sampled-Trajectories"
N_PER_CLASS = 400  # balanced target: up to this many resolved and unresolved each


def _shards():
    from huggingface_hub import list_repo_files
    files = [f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet")]
    raw = [f for f in files if "raw" in f.lower()]  # the train.raw split
    return sorted(raw or files)


def _traj_col(names):
    for c in ("messages", "trajectory", "conversations"):
        if c in names:
            return c
    raise KeyError(f"no trajectory column among {names}")


def load_runs(n_per_class=N_PER_CLASS):
    """Balanced subset controlled for run_id: each run_id contributes the SAME number of
    resolved and unresolved runs, so the dependency features cannot separate model/scaffold
    instead of outcome. (The raw dump is ~8% pass and the unresolved majority is dominated by
    a single gpt-4o run_id, so a naive class balance confounds model with outcome.)
    """
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    by_run = {}  # run_id -> {0: [resolved...], 1: [unresolved...]}
    for shard in _shards():
        path = hf_hub_download(REPO, shard, repo_type="dataset")
        pf = pq.ParquetFile(path)
        names = pf.schema_arrow.names  # top-level columns (schema.names flattens nested leaves)
        tcol = _traj_col(names)
        cols = [c for c in (tcol, "resolved", "run_id") if c in names]
        for batch in pf.iter_batches(batch_size=64, columns=cols):
            d = batch.to_pydict()
            for i in range(len(d[tcol])):
                traj = d[tcol][i]
                if isinstance(traj, str):
                    traj = json.loads(traj)
                steps = to_steps(traj)
                if len(steps) < 2:
                    continue
                resolved = int(d["resolved"][i]) if "resolved" in d else 0
                rid = d["run_id"][i] if "run_id" in d else "all"
                slot = by_run.setdefault(rid, {0: [], 1: []})
                slot[resolved ^ 1].append({"resolved": resolved, "steps": steps})
        balanced = sum(min(len(v[0]), len(v[1])) for v in by_run.values())
        if balanced >= n_per_class:
            break
    # Use a SINGLE run_id (one scaffold: one model, one maxiter, one temperature). Pooling
    # multiple run_ids confounds outcome with scaffold: different maxiter caps give different
    # step-length distributions, which a single model exploits even with per-run_id class
    # balance, inflating flat/exec/dep alike. Restricting to the largest balanced run_id keeps
    # SWE-Gym comparable to the other single-source corpora.
    best = max(by_run.values(), key=lambda v: min(len(v[0]), len(v[1])))
    k = min(len(best[0]), len(best[1]))
    return best[0][:k] + best[1][:k]


def main():
    runs = load_runs()
    n_res = sum(r["resolved"] for r in runs)
    print(f"SWE-Gym: {len(runs)} runs (balanced), {n_res} resolved / {len(runs) - n_res} failed")
    depths = []
    for rec in runs[:200]:
        G = build_graph(rec["steps"], dependency="explicit", shared_resource=False)
        depths.append(characterize(G).get("dep_depth", 0))
    if depths:
        import statistics as st
        print(f"sample dep depth: median {st.median(depths)}, max {max(depths)} (n={len(depths)})")


if __name__ == "__main__":
    main()
