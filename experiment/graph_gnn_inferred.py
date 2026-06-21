"""Cheap GNN ablation: run the SAME source-blind GIN on the OBSERVED dependency
graph and on the full-history INFERRED dependency graph, for the six observed
corpora. This is the GNN analogue of the logistic gating experiment: if a generic
message-passing network is also fooled by the saturated (rho->1) inferred layer,
GIN(inf) should collapse toward run size within corpus and invert in transfer,
while GIN(obs) reproduces the paper's Table 4 GIN column.

The GIN, to_pyg, and train_eval are copied verbatim from
grade/experiment/graph_gnn.py (importing it would run its module-level
observed-graph sweep as a side effect). The step extraction is copied from
agent_failure_gating.py so the graphs are built identically to the master harness.

Run:  python graph_gnn_inferred.py
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from grade import build_graph

import agent_graph_swe_agent as swe
import agent_graph_tau_bench as tau
import agent_graph_openhands as oh
import agent_graph_swegym as swegym
import agent_reward_bench as web
import agent_graph_tau2_bench as tau2

NTYPES = ["agent", "decision", "tool_call", "dependency_resource"]


# ---- copied verbatim from graph_gnn.py ----
def to_pyg(G, label):
    nodes = list(G.nodes(data=True))
    idx_map = {n: i for i, (n, _) in enumerate(nodes)}
    sidx = [d.get("idx", 0) for _, d in nodes if d.get("ntype") in ("decision", "tool_call")]
    mx = max(sidx) if sidx else 1
    feats = []
    for _, d in nodes:
        nt = [1.0 if d.get("ntype") == t else 0.0 for t in NTYPES]
        kf = d.get("kind_fs")
        k = [1.0 if kf == "read" else 0.0, 1.0 if kf == "write" else 0.0, 1.0 if not kf else 0.0]
        pos = [(d.get("idx", 0) / mx) if d.get("ntype") in ("decision", "tool_call") else 0.0]
        feats.append(nt + k + pos)
    x = torch.tensor(feats, dtype=torch.float)
    src, dst = [], []
    for u, v, _ in G.edges(data=True):
        src.append(idx_map[u]); dst.append(idx_map[v])
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=ei, y=torch.tensor([label], dtype=torch.float))


class GIN(nn.Module):
    def __init__(self, in_dim, hid=64):
        super().__init__()
        self.c1 = GINConv(nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, hid)))
        self.c2 = GINConv(nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, hid)))
        self.head = nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hid, 1))

    def forward(self, x, edge_index, batch):
        h = F.relu(self.c1(x, edge_index))
        h = F.relu(self.c2(h, edge_index))
        hg = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)
        return self.head(hg).squeeze(-1)


def train_eval(train_data, test_data, in_dim, epochs=40, seed=0):
    torch.manual_seed(seed)
    model = GIN(in_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ys = torch.tensor([d.y.item() for d in train_data])
    pos = float(ys.mean().clamp(0.05, 0.95))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos))
    tl = DataLoader(train_data, batch_size=32, shuffle=True)
    for _ in range(epochs):
        model.train()
        for b in tl:
            opt.zero_grad()
            loss = lossf(model(b.x, b.edge_index, b.batch), b.y)
            loss.backward()
            opt.step()
    model.eval()
    ps, ys2 = [], []
    with torch.no_grad():
        for b in DataLoader(test_data, batch_size=128):
            ps.append(torch.sigmoid(model(b.x, b.edge_index, b.batch)))
            ys2.append(b.y)
    return roc_auc_score(torch.cat(ys2).numpy(), torch.cat(ps).numpy())


# ---- step extraction copied from gating_experiment.py ----
def _pairs():
    P = {}
    print("loading tau-bench...", flush=True)
    paths = tau._ensure_files()
    P["tau-bench"] = [
        (tau.to_steps(rec["messages"], rec.get("model_path", "model")),
         0 if bool((rec.get("eval_result") or {}).get("db_match")) else 1)
        for rec in tau.load_runs(paths)
        if isinstance(rec.get("messages"), list) and len(rec["messages"]) >= 3
        and "db_match" in (rec.get("eval_result") or {})]
    print("loading SWE-agent...", flush=True)
    P["SWE-agent"] = [(swe.to_steps(rec["traj"]), 0 if rec["target"] else 1)
                      for rec in swe.load_runs(swe.N_RUNS)]
    print("loading SWE-Gym...", flush=True)
    P["SWE-Gym"] = [(rec["steps"], 0 if rec["resolved"] else 1)
                    for rec in swegym.load_runs()]
    print("loading OpenHands...", flush=True)
    P["OpenHands"] = [(oh.to_steps(rec["traj"]), 0 if rec["resolved"] else 1)
                      for rec in oh.load_runs(oh.N_RUNS)]
    print("loading tau2-bench...", flush=True)
    P["tau2-bench"] = [(rec["steps"], rec["label"]) for rec in tau2.load_runs()]
    print("loading web...", flush=True)
    P["web"] = [(web.to_steps(rec["steps"]), rec["label"]) for rec in web.load_runs(web.N_RUNS)]
    return P


def _build(pairs, mode):
    data, y = [], []
    for steps, label in pairs:
        if len(steps) < 2:
            continue
        g = build_graph(steps, dependency=mode, shared_resource=False)
        data.append(to_pyg(g, label))
        y.append(label)
    return data, np.array(y)


def main():
    np.random.seed(0)
    P = _pairs()
    order = ["tau-bench", "SWE-agent", "SWE-Gym", "OpenHands", "tau2-bench", "web"]
    OBS, INF, Y = {}, {}, {}
    for nm in order:
        OBS[nm], Y[nm] = _build(P[nm], "explicit")
        INF[nm], _ = _build(P[nm], "full_context")
        print(f"built {nm}: {len(Y[nm])} runs", flush=True)
    in_dim = OBS[order[0]][0].x.shape[1]

    print("\n=== GIN WITHIN-CORPUS (5-fold, seed 0, ROC-AUC) ===", flush=True)
    print("%-12s %9s %9s" % ("corpus", "GIN(obs)", "GIN(inf)"), flush=True)
    for nm in order:
        y = Y[nm]
        a_obs, a_inf = [], []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y):
            a_obs.append(train_eval([OBS[nm][i] for i in tr], [OBS[nm][i] for i in te], in_dim))
            a_inf.append(train_eval([INF[nm][i] for i in tr], [INF[nm][i] for i in te], in_dim))
        print("%-12s %9.3f %9.3f" % (nm, np.mean(a_obs), np.mean(a_inf)), flush=True)

    print("\n=== GIN POOLED LEAVE-ONE-CORPUS-OUT TRANSFER (ROC-AUC) ===", flush=True)
    print("%-12s %9s %9s" % ("held-out", "GIN(obs)", "GIN(inf)"), flush=True)
    for held in order:
        tr = [nm for nm in order if nm != held]
        tr_obs = [d for nm in tr for d in OBS[nm]]
        tr_inf = [d for nm in tr for d in INF[nm]]
        a_obs = train_eval(tr_obs, OBS[held], in_dim)
        a_inf = train_eval(tr_inf, INF[held], in_dim)
        print("%-12s %9.3f %9.3f" % (held, a_obs, a_inf), flush=True)


if __name__ == "__main__":
    main()
