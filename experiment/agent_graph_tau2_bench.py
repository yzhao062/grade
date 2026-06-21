"""Second DB tool-use corpus: tau2-bench trajectories (observed db-id value-flow deps).

A second low-flat DB point, to test whether the tau-bench result (dependency structure beats
run size where size is weak) replicates on the successor benchmark with a fresh model pool.
AgentSuite/tau2-bench-trajectories is 30 per-model JSONL files (telecom/retail/airline), each
run a multi-turn dialogue where the agent calls read tools that return DB entity ids and later
tools consume those ids. Run label = eval_result.score (0.0 = failed, 1.0 = resolved).

Dependency model: db-id value-flow. RESOURCE KEY = an id string a read tool emitted in its
result; a later tool_call that consumes that id depends on the producing step. Keys are
extracted schema-agnostically (recurring id-like JSON string values), since the three domains
have different DB schemas.

Confound control (the SWE-Gym lesson): per-model failure rate spans 43%-95%, so a naive class
balance would separate model identity, not outcome. We balance within model and use a single
model slice.

Run:  python experiment/agent_graph_tau2_bench.py
"""
import json
import re

from grade import build_graph, characterize

REPO = "AgentSuite/tau2-bench-trajectories"
MAX_MODELS = 10          # how many per-model files to scan for the best balanced slice
_IDLIKE = re.compile(r"^[A-Za-z]{0,4}[-_]?\d[\w\-]{2,}$")  # token with a digit, no spaces


def _string_values(obj):
    """Yield every string scalar nested in a dict/list (schema-agnostic)."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _string_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _string_values(v)


def _ids(obj):
    return {v for v in _string_values(obj) if _IDLIKE.match(v)}


def to_steps(messages):
    """Typed steps with OBSERVED db-id value-flow dependencies."""
    steps, idx, last_on_id, last_tc = [], 0, {}, None
    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            steps.append({"idx": idx, "agent": "agent", "kind": "decision", "deps": []})
            idx += 1
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function") or tc
                name = fn.get("name") or "tool"
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                args = args if isinstance(args, dict) else {}
                deps = []
                for v in _ids(args):                      # an arg reusing an earlier id
                    if v in last_on_id:
                        deps = [last_on_id[v]]
                        break
                steps.append({"idx": idx, "agent": "env", "kind": "tool_call", "deps": deps,
                              "cmd": name, "file": None, "kind_fs": "read"})
                last_tc = idx
                idx += 1
        elif role == "tool" and last_tc is not None:
            content = msg.get("content")
            try:
                result = json.loads(content) if isinstance(content, str) else content
            except ValueError:
                result = {}
            for v in _ids(result):                        # ids this result produced
                last_on_id.setdefault(v, last_tc)
    return steps


def _model_files(limit):
    from huggingface_hub import list_repo_files
    fs = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".jsonl"))
    return fs[:limit]


def load_runs(max_models=MAX_MODELS):
    """Single balanced model slice (largest min(success, fail) across scanned models)."""
    from huggingface_hub import hf_hub_download
    by_model = {}  # file -> {0: [success...], 1: [failed...]}
    for fn in _model_files(max_models):
        path = hf_hub_download(REPO, fn, repo_type="dataset")
        slot = by_model.setdefault(fn, {0: [], 1: []})
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                er = rec.get("eval_result") or {}
                if "score" not in er:
                    continue
                label = 1 if float(er["score"]) == 0.0 else 0   # 1 = FAILED
                steps = to_steps(rec.get("messages") or [])
                if len(steps) < 2:
                    continue
                slot[label].append({"label": label, "steps": steps})
    best = max(by_model.values(), key=lambda v: min(len(v[0]), len(v[1])))
    k = min(len(best[0]), len(best[1]))
    return best[0][:k] + best[1][:k]


def main():
    runs = load_runs()
    nf = sum(r["label"] for r in runs)
    print(f"tau2-bench: {len(runs)} runs (single balanced model), {nf} failed / {len(runs) - nf} resolved")
    depths = []
    for r in runs[:200]:
        G = build_graph(r["steps"], dependency="explicit", shared_resource=False)
        depths.append(characterize(G).get("dep_depth", 0))
    if depths:
        import statistics as st
        print(f"sample dep depth: median {st.median(depths)}, max {max(depths)} (n={len(depths)})")


if __name__ == "__main__":
    main()
