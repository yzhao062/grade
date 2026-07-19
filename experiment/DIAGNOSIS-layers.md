# Diagnosis: why does the dependency layer help on tau-bench / SWE-agent but not on OpenHands / web?

Question: the web (AgentRewardBench) and OpenHands ablations look like
negative results (`full` <= `flat`). Is this a **data limit**, a **method/code bug**, or a
**genuine finding**? Diagnostic: `experiment/diagnose_layers.py`.

## Setup recap

Keystone = nested feature ablation `flat (4) ⊂ exec (8) ⊂ full (12)`, logistic regression,
ROC-AUC, 5 seeds × 5-fold CV, label 1 = failed run. Feature groups (`grade/features.py`):

- flat: `n_steps, n_tool_calls, n_decisions, n_agents`
- exec: `max_agent_outdeg, agent_gini, n_agent_transitions, agent_recurrence`  (**agent-level**)
- dep:  `n_dep_edges, dep_depth, max_blast_indeg, max_audit_outdeg`  (**raw counts**)

## Finding 1 — the current exec layer is a protocol-shape artifact on single-agent corpora

(Correction: The earlier draft claimed the exec features "collapse to a
single actor" with `agent_gini=0, n_agent_transitions=0, max_agent_outdeg=n_steps`. That
mis-stated the mechanism: the loaders assign decision steps to `agent` and tool-call steps
to `env`, and `grade.features.layered_features` counts BOTH in `agent_seq`. So the exec
block is measuring assistant/environment alternation and tool-call cadence, not a single
actor.)

Verified on real OpenHands graphs: `n_agents = 2` (agent + env), `agent_gini = 0.0` (the
decision/tool counts are balanced, so the two-actor split is even), `n_agent_transitions = 2`
(the regular agent↔env flip), `agent_recurrence ≈ 0.99`, and `max_agent_outdeg ≈ n_steps/2`
(|r| with `n_steps` = 1.00, so size-collinear). Every one of these is **constant across
runs** within a single-agent corpus, hence non-discriminative (single-feature AUC ≈ 0.50),
but the VALUES are not zero and the layer is **semantically misnamed** for single-agent
runs: it is protocol shape, not agent topology.

Consequence: this does not rescue `full - exec` as the primary contrast; it makes the case
against it stronger. The exec baseline is not just weak, it is uninterpretable as
"execution topology" on single-agent corpora. The primary dependency contrast must be
`flat` vs `flat + dep`, with exec reported only as a secondary diagnostic until its
single-agent meaning is fixed (assign tool calls to the issuing assistant, or compute
agent-topology features over decision nodes only). exec genuinely helps only on the
multi-agent corpus (tau-bench), where there really are multiple agents.

## Finding 2 — the raw dependency features are size-confounded

`n_dep_edges` |r| with `n_steps`: tau 0.70, SWE 0.49, **OpenHands 0.82, web 0.96**. The raw
dep counts mostly re-encode run size, which `flat` already has. Size-normalizing the dep
block (each count / `n_steps`) isolates dependency **structure** from size — and that is the
quantity the paper's "signal beyond size" thesis is actually about.

## Finding 3 — the decisive test: does dependency STRUCTURE beat pure size?

`flat` vs `flat + dep(size-normalized)`, skipping the degenerate exec block:

| corpus | flat | flat+dep(norm) | gain | 5/5? |
|---|---|---|---|---|
| tau-bench (multi-agent DB) | 0.583 | 0.614 | **+0.031** | 5/5 |
| SWE-agent (weak coding, 90% fail) | 0.628 | 0.713 | **+0.085** | 5/5 |
| OpenHands (strong coding, 52% fail) | 0.698 | 0.695 | −0.004 | 1/5 |
| AgentRewardBench (web, 67% fail) | 0.765 | 0.759 | −0.007 | 1/5 |

> **Pre-fix data note (2026-07-19).** The tau-bench numbers in this section (the `0.583 / 0.614 /
> +0.031` row above, and the `+0.031` restated in the prose below) predate the 2026-07-05 upstream
> correction of `AgentSuite/tau-bench-trajectories` (commit `382e57d`, which fixed mislabeled gpt-4.1
> and Kimi-K2-Instruct base-model runs). They are the as-submitted GRADE analysis on the pre-fix
> corpus; the loader is now pinned to the corrected commit. Regenerate this row before reusing it. The
> other four corpora are unaffected.

## Answer: it is BOTH, and they are now separable

- **Not a code bug.** Loaders produce real, non-degenerate dependency edges (SWE dep_depth
  median ~10, OpenHands ~6, web ~9). The nulls are not parsing failures.
- **Partly a method artifact (now fixed in the analysis).** The visible `full < flat` on web
  is dominated by the **agent-level exec features being degenerate for single-agent runs**,
  plus **raw dep counts being size-confounded**. The honest comparison is `flat` vs
  `flat+dep`, with size-normalized dep features.
- **Partly a genuine data limit.** Even with the clean comparison, the dependency layer does
  **not** beat size on OpenHands or web: where run size alone already predicts failure
  (flat AUC 0.70, 0.77), dependency structure adds nothing. On tau-bench and SWE-agent,
  where size is a weak predictor (flat 0.58, 0.63), dependency structure adds +0.031 / +0.085.

**Honest headline:** the dependency layer beats run size on 2 of 4 corpora, exactly those
where size alone is a weak failure predictor. The boundary (flat ≈ 0.65) is the result, not
a defect. This is "structure earns its place where size is weak," measured.

## Update: SWE-Gym added as a fifth corpus (after two confound controls)

SWE-Gym/OpenHands-Sampled-Trajectories was added on the low-flat coding side. It took two
confound controls to make it trustworthy, both caught before use:

1. **Model confound.** The raw dump is ~8% pass and the unresolved majority is one gpt-4o
   run_id, while the resolved runs are a mix including claude. A naive class balance made the
   dependency features separate model/scaffold, not outcome (single-feature AUC 0.855,
   gain +0.171). Fixed by balancing within run_id.
2. **Pooling-heterogeneity confound.** Even balanced within run_id, pooling run_ids with
   different maxiter caps (30 vs 50) gives different step-length distributions; a single
   model exploits scaffold structure, inflating flat (0.663 -> 0.747) and exec (0.882) alike.
   Fixed by restricting to a single run_id (one model, one maxiter, one temperature).

Single-scaffold SWE-Gym (gpt-4o, maxiter-50, 376 balanced runs): flat 0.663, flat+dep(norm)
0.804, primary gain **+0.142** (5/5). It lands on the low-flat "helps" side and holds the
observed pattern. (Caveat retained: its secondary exec block is unusually high, so exec is
not interpreted for SWE-Gym either; only the flat-vs-flat+dep primary is used.)

Five-corpus observed pattern (flat vs flat+dep, size-normalized): helps on tau-bench
(+0.031), SWE-agent (+0.085), SWE-Gym (+0.142); null on OpenHands (-0.004), web (-0.007).
The three "helps" corpora are the three with the weakest flat baseline (0.58-0.66); the two
nulls are the two with the strongest (0.70-0.77). Reported as an observed pattern, not a
universal threshold.

## Notes

1. **Adopt size-normalized dependency features** in `features.py` (principled: the thesis is
   "beyond size"; raw counts at r=0.96 with size cannot test it). Report `flat` vs
   `flat+dep` as the primary contrast; keep `+exec` as a secondary bar that visibly helps
   only in the multi-agent corpus (this motivates the dependency layer).
2. **Reframe Figure 2 / claims** to the honest 2-of-4 boundary; keep OpenHands and web as
   negatives. Drop the "robust half / 3-of-4" framing, which leaned on the degenerate exec
   baseline.
3. **Optional, more data:** add SWE-Gym (weak coding) and a second multi-agent/DB corpus to
   populate the low-flat side of the boundary, where structure is predicted to help.

## Open questions

- Is size-normalizing the dep block principled, or does it read as p-hacking? (It is applied
  uniformly to all corpora and still leaves OpenHands/web null.)
- Is `flat` vs `flat+dep` the right primary contrast given exec degeneracy on single-agent
  corpora, or should exec be redesigned to be meaningful for single-agent runs?
- Any loader bug that could spuriously flatten the dependency signal on OpenHands/web?
