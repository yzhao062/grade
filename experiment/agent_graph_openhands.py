"""Strong-model coding corpus: OpenHands SWE trajectories (observed, multi-hop deps).

SWE-agent (nebius/SWE-agent-trajectories) is weak models (llama-70b/8b) that thrash;
this is the strong-model counterpart -- Qwen3-Coder-480B run under the OpenHands
scaffold on SWE-rebench issues, with a per-trajectory ``resolved`` label. It tests the
same depth question on a strong agent: do real edit dependencies stay shallow once the
model is competent, and does the dependency layer still carry run-level failure signal?

Corpus: nebius/SWE-rebench-openhands-trajectories. Each row has a ``trajectory`` (a list
of role-tagged messages; assistant turns carry ``tool_calls`` with a function ``name`` and
JSON ``arguments``), a ``resolved`` int (1 = issue solved), and ``instance_id``. We parse
each tool call to the file it touches (str_replace_editor view = read; create / str_replace
/ insert = write; bash = run) and model each file event as depending on the prior event on
the same file -- the observed per-file interaction chain, identical to the SWE-agent model.

Run:  python experiment/agent_graph_openhands.py   (downloads one parquet shard once)
"""
import json

from grade import build_graph, characterize

REPO = "nebius/SWE-rebench-openhands-trajectories"
N_RUNS = 600  # first pass on one shard, matching the scale of the other corpora

_WRITE_EDITOR_CMDS = {"create", "str_replace", "insert", "append", "write"}


def _parse_tool_call(tc):
    """Return (name, file, kind) for one OpenHands tool_call; kind in read/write/run."""
    fn = tc.get("function") or {}
    name = fn.get("name")
    raw = fn.get("arguments") or "{}"
    try:
        args = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except (ValueError, TypeError):
        args = {}
    if name in ("str_replace_editor", "str_replace_based_edit_tool", "edit_file"):
        path = args.get("path") or args.get("file")
        cmd = (args.get("command") or "").lower()
        if cmd in _WRITE_EDITOR_CMDS:
            return (name, path, "write")
        return (name, path, "read")  # view / undo_edit / default = read
    if name in ("bash", "execute_bash", "run"):
        return (name, None, "run")
    if name in ("finish", "submit"):
        return (name, None, "write")
    return (name, None, "run")


def to_steps(traj):
    """Typed steps with OBSERVED per-file dependencies (same model as SWE-agent).

    One decision step per assistant turn, then one tool_call step per tool_call. A file
    event depends_on the prior event on the SAME file, so repeated view/edit cycles form
    an observed multi-hop chain.
    """
    steps, idx, last_on_file = [], 0, {}
    for msg in traj:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue  # system / user / tool messages are observations
        steps.append({"idx": idx, "agent": "agent", "kind": "decision", "deps": []})
        idx += 1
        for tc in (msg.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            name, f, kind = _parse_tool_call(tc)
            deps = [last_on_file[f]] if (f and f in last_on_file) else []
            steps.append({"idx": idx, "agent": "env", "kind": "tool_call", "deps": deps,
                          "cmd": name, "file": f, "kind_fs": kind})
            if f:
                last_on_file[f] = idx
            idx += 1
    return steps


def _first_shard():
    from huggingface_hub import list_repo_files
    pqs = sorted(f for f in list_repo_files(REPO, repo_type="dataset")
                 if f.endswith(".parquet"))
    if not pqs:
        raise RuntimeError(f"no parquet shards found in {REPO}")
    return pqs[0]


def load_runs(limit):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    path = hf_hub_download(REPO, _first_shard(), repo_type="dataset")
    pf = pq.ParquetFile(path)
    out = []
    for batch in pf.iter_batches(batch_size=100,
                                 columns=["instance_id", "resolved", "trajectory"]):
        d = batch.to_pydict()
        for i in range(len(d["trajectory"])):
            traj = d["trajectory"][i]
            if isinstance(traj, str):
                traj = json.loads(traj)
            out.append({"instance_id": d["instance_id"][i],
                        "resolved": int(d["resolved"][i]), "traj": traj})
            if len(out) >= limit:
                return out
    return out


def main():
    runs = load_runs(N_RUNS)
    n_res = sum(r["resolved"] for r in runs)
    print(f"OpenHands: {len(runs)} runs, {n_res} resolved "
          f"({n_res / max(len(runs), 1):.0%} pass), {len(runs) - n_res} failed")
    depths = []
    for rec in runs[:200]:
        steps = to_steps(rec["traj"])
        if len(steps) < 2:
            continue
        G = build_graph(steps, dependency="explicit", shared_resource=False)
        c = characterize(G)
        depths.append(c.get("dep_depth", 0))
    if depths:
        import statistics as st
        print(f"sample dep depth: median {st.median(depths)}, max {max(depths)} "
              f"(n={len(depths)})")


if __name__ == "__main__":
    main()
