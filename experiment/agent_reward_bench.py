"""Web-agent corpus: AgentRewardBench expert-labeled trajectories (observed, per-page deps).

A fourth agent class, web navigation, to test whether the representation and the dependency
layer carry beyond tool use and coding. McGill-NLP/agent-reward-bench has web-agent
trajectories across WebArena / VisualWebArena, each with an EXPERT success label and a step
list of (reasoning, action, url). We parse each action to the page (url) it acted on and
model each action as depending on the prior action on the SAME page, the observed per-page
interaction chain (the web analog of per-file edits in coding).

Run:  python experiment/agent_reward_bench.py   (downloads the cleaned trajectories once)
"""
import csv
import glob
import json
import os

from grade import build_graph

REPO = "McGill-NLP/agent-reward-bench"
LOCAL = os.environ.get("ARB_DIR", "./data/agentrewardbench")  # set ARB_DIR to your AgentRewardBench data root
BENCHMARKS = ("webarena", "visualwebarena")
N_RUNS = 600

_WRITE_ACTIONS = {"fill", "click", "select_option", "press", "check", "clear",
                  "dblclick", "hover", "upload_file", "goto", "go_back", "go_forward"}


def _ensure():
    # Fast path: if the annotations + at least one benchmark's cleaned dir are already
    # present, skip the network etag re-verification snapshot_download does every call.
    have_csv = os.path.exists(os.path.join(LOCAL, "data", "annotations.csv"))
    have_traj = any(os.path.isdir(os.path.join(LOCAL, "cleaned", b)) for b in BENCHMARKS)
    if have_csv and have_traj:
        return
    from huggingface_hub import snapshot_download
    patterns = ["data/annotations.csv"] + [f"cleaned/{b}/**" for b in BENCHMARKS]
    snapshot_download(REPO, repo_type="dataset", local_dir=LOCAL, allow_patterns=patterns)


def _labels():
    """(benchmark, model_name, task_id) -> 1 if the expert marked it failed, else 0."""
    out = {}
    with open(os.path.join(LOCAL, "data", "annotations.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["benchmark"], row["model_name"], row["task_id"])
            if key in out:
                continue  # first annotator wins
            out[key] = 1 if row["trajectory_success"].strip().lower().startswith("unsuccess") else 0
    return out


def load_runs(limit):
    _ensure()
    labels = _labels()
    out = []
    for b in BENCHMARKS:
        for p in sorted(glob.glob(os.path.join(LOCAL, "cleaned", b, "*", "*", "*.json"))):
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError):
                continue
            steps_list = data.get("steps") or []
            if len(steps_list) < 2:
                continue
            model = os.path.basename(os.path.dirname(os.path.dirname(p)))  # the model_name dir
            task_id = os.path.splitext(os.path.basename(p))[0]             # e.g. webarena.100
            lab = labels.get((b, model, task_id))
            if lab is None:
                continue  # no expert label for this run
            # Keep only (action, url) per step; the raw JSON carries a full axtree,
            # screenshot path, and bounding boxes per step that would pin GBs in memory.
            light = [{"action": st.get("action") or "", "url": st.get("url") or ""}
                     for st in steps_list]
            out.append({"benchmark": b, "task_id": task_id, "label": lab, "steps": light})
            if len(out) >= limit:
                return out
    return out


def to_steps(steps_list):
    """Typed steps with OBSERVED per-page dependencies (the web analog of per-file)."""
    steps, idx, last_on_url = [], 0, {}
    for st in steps_list:
        action = (st.get("action") or "").strip()
        url = st.get("url") or ""
        steps.append({"idx": idx, "agent": "agent", "kind": "decision", "deps": []})
        idx += 1
        atype = action.split("(", 1)[0].strip() if action else ""
        deps = [last_on_url[url]] if (url and url in last_on_url) else []
        kind = "write" if atype in _WRITE_ACTIONS else "read"
        steps.append({"idx": idx, "agent": "env", "kind": "tool_call", "deps": deps,
                      "cmd": atype or "noop", "file": url or None, "kind_fs": kind})
        if url:
            last_on_url[url] = idx
        idx += 1
    return steps


def main():
    runs = load_runs(N_RUNS)
    n_fail = sum(r["label"] for r in runs)
    print(f"AgentRewardBench (web): {len(runs)} runs, {n_fail} failed "
          f"({n_fail / max(len(runs), 1):.0%}), {len(runs) - n_fail} success")
    ok = 0
    for r in runs[:50]:
        s = to_steps(r["steps"])
        if len(s) >= 2:
            build_graph(s, dependency="explicit", shared_resource=False)
            ok += 1
    print(f"built {ok} sample graphs OK")


if __name__ == "__main__":
    main()
