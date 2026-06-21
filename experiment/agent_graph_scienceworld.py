"""Embodied corpus: ScienceWorld interactive text-game trajectories (observed state deps).

A FIFTH agent class (embodied / interactive text-world), to test whether the representation
and the dependency layer carry beyond DB tool use, coding, and web. The repo is mislabeled
"webshop_expert_trajectories" but the science-experiment files under ``test trajs/`` are
ScienceWorld: a single agent (Thought + Action per turn) acts on a stateful simulated house,
the env returns an observation, and each run has an info.reward in 0..100 (100 = task solved).

Dependency model: world state, primarily LOCATION. ``go to X`` / ``teleport to X`` move the
agent (write location); ``open / focus on / pick up / move`` are only valid in the current
room (read location). So each action depends on the most recent location-changing step, the
observed write-to-read chain through world state (the embodied analog of per-file edits).

Confound control (the SWE-Gym / tau-bench lesson): source files differ by agent_arch / llm /
temperature, which shifts the base rate. We balance resolved vs failed WITHIN each source
file so the dependency features cannot separate source instead of outcome.

Run:  python experiment/agent_graph_scienceworld.py
"""
import json
import re

from grade import build_graph, characterize

REPO = "lclan/webshop_expert_trajectories"
N_PER_CLASS = 200

_WRITE_LOC = ("go to", "teleport to")          # actions that change (write) location
_THOUGHT = re.compile(r"Thought:\s*(.*?)\s*Action:", re.DOTALL | re.IGNORECASE)
_ACTION = re.compile(r"Action:\s*(.+)", re.IGNORECASE)


def _sciworld_files():
    from huggingface_hub import list_repo_files
    return sorted(f for f in list_repo_files(REPO, repo_type="dataset")
                  if "sciworld" in f.lower() and f.lower().endswith(".json"))


def _turns(rec):
    """Yield (role, text) for each conversation turn, tolerating key-name variants."""
    conv = rec.get("conversations") or rec.get("conversation") or []
    for t in conv:
        if not isinstance(t, dict):
            continue
        role = (t.get("role") or t.get("from") or "").lower()
        text = t.get("content") or t.get("value") or ""
        yield role, text


def to_steps(rec):
    """Typed steps with OBSERVED location (world-state) dependencies."""
    steps, idx, last_loc = [], 0, None
    for role, text in _turns(rec):
        if role not in ("assistant", "gpt"):
            continue
        if "action:" not in text.lower():
            continue
        th = _THOUGHT.search(text)
        steps.append({"idx": idx, "agent": "agent", "kind": "decision", "deps": [],
                      "thought": (th.group(1).strip()[:80] if th else "")})
        idx += 1
        am = _ACTION.search(text)
        action = (am.group(1).splitlines()[0].strip() if am else "").lower()
        verb = action.split()[0] if action else "noop"
        is_move = any(action.startswith(w) for w in _WRITE_LOC)
        deps = [last_loc] if last_loc is not None else []
        steps.append({"idx": idx, "agent": "env", "kind": "tool_call", "deps": deps,
                      "cmd": verb, "file": "location", "kind_fs": "write" if is_move else "read"})
        if is_move:
            last_loc = idx
        idx += 1
    return steps


def load_runs(n_per_class=N_PER_CLASS):
    from huggingface_hub import hf_hub_download
    by_src = {}  # source file -> {0: [success...], 1: [failed...]}
    for fn in _sciworld_files():
        if "webshop_gpt" in fn.lower():
            continue  # genuine WebShop dumps at the repo root; off-topic, skip
        path = hf_hub_download(REPO, fn, repo_type="dataset")
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            data = data.get("data") or list(data.values())
        slot = by_src.setdefault(fn, {0: [], 1: []})
        for rec in data:
            if not isinstance(rec, dict):
                continue
            info = rec.get("info") or {}
            reward = info.get("reward", rec.get("reward"))
            if reward is None:
                continue
            label = 0 if int(reward) >= 100 else 1   # 1 = FAILED (not fully solved)
            steps = to_steps(rec)
            if len(steps) < 2:
                continue
            slot[label].append({"label": label, "steps": steps, "reward": int(reward)})
    # Single largest balanced source (one model, one temperature/slice). Pooling sources
    # that differ in sampling temperature shifts the run-length distribution and inflates
    # flat/exec, the same way pooling scaffolds did for SWE-Gym; restrict to one source.
    best = max(by_src.values(), key=lambda v: min(len(v[0]), len(v[1])))
    k = min(len(best[0]), len(best[1]), n_per_class)
    return best[0][:k] + best[1][:k]


def main():
    runs = load_runs()
    nf = sum(r["label"] for r in runs)
    print(f"ScienceWorld: {len(runs)} runs (balanced within source), {nf} failed / {len(runs) - nf} solved")
    depths = []
    for r in runs[:200]:
        G = build_graph(r["steps"], dependency="explicit", shared_resource=False)
        depths.append(characterize(G).get("dep_depth", 0))
    if depths:
        import statistics as st
        print(f"sample dep depth: median {st.median(depths)}, max {max(depths)} (n={len(depths)})")


if __name__ == "__main__":
    main()
