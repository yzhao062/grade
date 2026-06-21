"""Relation-aware graph-network baselines for the off-the-shelf comparison. The paper's
claim that off-the-shelf graph networks underperform the source-aware features was first
tested only against a single edge-type-blind GIN, which by construction folds all five
relations into one adjacency. A standard relation-aware GNN (R-GCN, HGT) is equally
off-the-shelf yet reads the edge type, so the dependency layer (and, on a graded graph, the
attachment grade carried as an edge type) is a channel the message passing can use directly.

This script runs THREE off-the-shelf graph networks on the SAME observed two-layer graphs,
across the six corpora, under one protocol:
  GIN    : edge-type-blind baseline (reproduces the paper's GIN column)
  R-GCN  : one transform per relation (RGCNConv, 5 relations)
  HGT    : heterogeneous attention over (node-type, relation, node-type) (HGTConv)
The five relations are emits / handoff_to (execution layer) and depends_on / reads / writes
(dependency layer). Node features are identical across all three models (8-dim: node-type
one-hot, read/write tag, normalized position), so the ONLY thing that changes is whether
the model can read the relation.

Reported per model: within-corpus 5-fold ROC-AUC and pooled leave-one-corpus-out transfer
ROC-AUC, each averaged over 5 seeds with a seed-block 95% CI (t(0.975,df=4)=2.776), the
same CI machinery as the keystone. This both tests whether a relation-aware off-the-shelf
GNN matches the hand-designed features once it can read the edge type, and gives the GNN
transfer the confidence intervals the single-seed GIN lacked.

Knobs (env vars) for fast iteration vs the full run:
  GR_SEEDS=5  GR_EPOCHS=40  GR_CORPORA=  GR_MODELS=GIN,R-GCN,HGT
  smoke test:  GR_SEEDS=1 GR_EPOCHS=8 GR_CORPORA=tau-bench,SWE-Gym python graph_gnn_relational.py

Run:  python graph_gnn_relational.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (GINConv, RGCNConv, HGTConv,
                                 global_mean_pool, global_max_pool)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from grade import build_graph

import agent_graph_swe_agent as swe
import agent_graph_tau_bench as tau
import agent_graph_openhands as oh
import agent_graph_swegym as swegym
import agent_reward_bench as web
import agent_graph_tau2_bench as tau2

# ---- schema: node types, relations, and the homogeneous->heterogeneous map -------
NTYPES = ["agent", "decision", "tool_call", "dependency_resource"]
RELATIONS = ["emits", "handoff_to", "depends_on", "reads", "writes"]
REL_ID = {r: i for i, r in enumerate(RELATIONS)}
N_REL = len(RELATIONS)

# HGT collapses decision/tool_call into one structural "step" type (the decision vs
# tool_call distinction is preserved in the node FEATURES, not the node type).
HET = {"agent": "agent", "decision": "step", "tool_call": "step",
       "dependency_resource": "resource"}
NODE_TYPES = ["agent", "step", "resource"]
EDGE_TYPES = [("agent", "emits", "step"),
              ("step", "handoff_to", "step"),
              ("step", "depends_on", "step"),
              ("step", "reads", "resource"),
              ("step", "writes", "resource")]
METADATA = (NODE_TYPES, EDGE_TYPES)
IN_DIM = len(NTYPES) + 3 + 1  # node-type one-hot + read/write/none + position = 8

N_SEEDS = int(os.environ.get("GR_SEEDS", "5"))
EPOCHS = int(os.environ.get("GR_EPOCHS", "40"))
_ONLY = [s.strip() for s in os.environ.get("GR_CORPORA", "").split(",") if s.strip()]
_MODELS = [s.strip() for s in os.environ.get("GR_MODELS", "GIN,R-GCN,HGT").split(",") if s.strip()]
DEP_MODE = os.environ.get("GR_DEP", "explicit")  # explicit = observed graph, full_context = inferred
_T975_4 = 2.776  # t(0.975, df=4): seed-block 95% CI half-width over 5 seeds
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- feature extraction (shared by all three models) ----------------------------

def _feat_map(G):
    """{node -> 8-dim feature}; position normalized over step nodes, as in graph_gnn.py."""
    nodes = list(G.nodes(data=True))
    sidx = [d.get("idx", 0) for _, d in nodes if d.get("ntype") in ("decision", "tool_call")]
    mx = max(sidx) if sidx else 1
    fm = {}
    for n, d in nodes:
        nt = [1.0 if d.get("ntype") == t else 0.0 for t in NTYPES]
        kf = d.get("kind_fs")
        k = [1.0 if kf == "read" else 0.0, 1.0 if kf == "write" else 0.0, 1.0 if not kf else 0.0]
        pos = [(d.get("idx", 0) / mx) if d.get("ntype") in ("decision", "tool_call") else 0.0]
        fm[n] = nt + k + pos
    return fm, nodes


def to_homog(G, label):
    """Homogeneous Data with x, edge_index, edge_type. GIN ignores edge_type; R-GCN reads it."""
    fm, nodes = _feat_map(G)
    idx = {n: i for i, (n, _) in enumerate(nodes)}
    x = torch.tensor([fm[n] for n, _ in nodes], dtype=torch.float)
    src, dst, et = [], [], []
    for u, v, d in G.edges(data=True):
        src.append(idx[u]); dst.append(idx[v]); et.append(REL_ID[d["etype"]])
    if src:
        ei = torch.tensor([src, dst], dtype=torch.long)
        etype = torch.tensor(et, dtype=torch.long)
    else:
        ei = torch.zeros((2, 0), dtype=torch.long)
        etype = torch.zeros((0,), dtype=torch.long)
    return Data(x=x, edge_index=ei, edge_type=etype, y=torch.tensor([float(label)]))


def to_hetero(G, label):
    """HeteroData with three node types and five canonical edge types, for HGT."""
    fm, nodes = _feat_map(G)
    local, kind = {}, {}
    bins = {nt: [] for nt in NODE_TYPES}
    for n, d in nodes:
        ht = HET[d.get("ntype")]
        kind[n] = ht
        local[n] = len(bins[ht])
        bins[ht].append(fm[n])
    data = HeteroData()
    for nt in NODE_TYPES:
        data[nt].x = (torch.tensor(bins[nt], dtype=torch.float) if bins[nt]
                      else torch.zeros((0, IN_DIM), dtype=torch.float))
    eb = {et: ([], []) for et in EDGE_TYPES}
    for u, v, d in G.edges(data=True):
        et = (kind[u], d["etype"], kind[v])
        if et in eb:
            eb[et][0].append(local[u]); eb[et][1].append(local[v])
    for et in EDGE_TYPES:
        s, t = eb[et]
        data[et].edge_index = (torch.tensor([s, t], dtype=torch.long) if s
                               else torch.zeros((2, 0), dtype=torch.long))
    data.y = torch.tensor([float(label)])
    return data


# ---- the three models -----------------------------------------------------------

class GIN(nn.Module):
    def __init__(self, in_dim, hid=64):
        super().__init__()
        self.c1 = GINConv(nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, hid)))
        self.c2 = GINConv(nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, hid)))
        self.head = nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hid, 1))

    def forward(self, b):
        h = F.relu(self.c1(b.x, b.edge_index))
        h = F.relu(self.c2(h, b.edge_index))
        hg = torch.cat([global_mean_pool(h, b.batch), global_max_pool(h, b.batch)], dim=1)
        return self.head(hg).squeeze(-1)


class RGCN(nn.Module):
    def __init__(self, in_dim, n_rel, hid=64):
        super().__init__()
        self.c1 = RGCNConv(in_dim, hid, n_rel)
        self.c2 = RGCNConv(hid, hid, n_rel)
        self.head = nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hid, 1))

    def forward(self, b):
        h = F.relu(self.c1(b.x, b.edge_index, b.edge_type))
        h = F.relu(self.c2(h, b.edge_index, b.edge_type))
        hg = torch.cat([global_mean_pool(h, b.batch), global_max_pool(h, b.batch)], dim=1)
        return self.head(hg).squeeze(-1)


class HGT(nn.Module):
    def __init__(self, in_dim, metadata, hid=64, heads=4):
        super().__init__()
        self.node_types = metadata[0]
        self.lin = nn.ModuleDict({nt: nn.Linear(in_dim, hid) for nt in self.node_types})
        self.c1 = HGTConv(hid, hid, metadata, heads=heads)
        self.c2 = HGTConv(hid, hid, metadata, heads=heads)
        self.head = nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hid, 1))

    def forward(self, b):
        x_dict = {nt: F.relu(self.lin[nt](x)) for nt, x in b.x_dict.items()}
        # HGTConv only returns node types that are an edge destination; carry the others
        # (the source-only "agent" type) forward so the next layer can still read them.
        out = self.c1(x_dict, b.edge_index_dict)
        x_dict = {nt: (F.relu(out[nt]) if nt in out else x_dict[nt]) for nt in x_dict}
        out = self.c2(x_dict, b.edge_index_dict)
        x_dict = {nt: (out[nt] if nt in out else x_dict[nt]) for nt in x_dict}
        bn = b.num_graphs
        # union pooling over all node types tolerates graphs missing a type (size=bn)
        hs = torch.cat([x_dict[nt] for nt in self.node_types if nt in x_dict], dim=0)
        bs = torch.cat([b.batch_dict[nt] for nt in self.node_types if nt in x_dict], dim=0)
        hg = torch.cat([global_mean_pool(hs, bs, size=bn), global_max_pool(hs, bs, size=bn)], dim=1)
        return self.head(hg).squeeze(-1)


MODEL_SPEC = {
    "GIN": (lambda: GIN(IN_DIM), "homog"),
    "R-GCN": (lambda: RGCN(IN_DIM, N_REL), "homog"),
    "HGT": (lambda: HGT(IN_DIM, METADATA), "hetero"),
}


def train_eval(make_model, train_data, test_data, seed=0, epochs=EPOCHS):
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = make_model().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ys = torch.tensor([float(d.y.item()) for d in train_data])
    pos = float(ys.mean().clamp(0.05, 0.95))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos).to(DEVICE))
    tl = DataLoader(train_data, batch_size=32, shuffle=True)
    for _ in range(epochs):
        model.train()
        for b in tl:
            b = b.to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(b), b.y.float())
            loss.backward()
            opt.step()
    model.eval()
    ps, ys2 = [], []
    with torch.no_grad():
        for b in DataLoader(test_data, batch_size=128):
            b = b.to(DEVICE)
            ps.append(torch.sigmoid(model(b)).cpu())
            ys2.append(b.y.cpu())
    return roc_auc_score(torch.cat(ys2).numpy(), torch.cat(ps).numpy())


def _ci(vals):
    a = np.asarray(vals, dtype=float)
    if len(a) < 2:
        return float(a.mean()), 0.0
    return float(a.mean()), float(_T975_4 * a.std(ddof=1) / np.sqrt(len(a)))


# ---- corpora (step extraction identical to graph_gnn_inferred.py / gating) -------

def _pairs():
    P, order = {}, []

    def add(name, fn):
        if _ONLY and name not in _ONLY:
            return
        try:
            print(f"loading {name}...", flush=True)
            P[name] = list(fn())
            order.append(name)
        except Exception as e:
            print(f"skip {name}: {type(e).__name__}: {str(e)[:90]}", flush=True)

    def _tau():
        paths = tau._ensure_files()
        return [(tau.to_steps(rec["messages"], rec.get("model_path", "model")),
                 0 if bool((rec.get("eval_result") or {}).get("db_match")) else 1)
                for rec in tau.load_runs(paths)
                if isinstance(rec.get("messages"), list) and len(rec["messages"]) >= 3
                and "db_match" in (rec.get("eval_result") or {})]

    add("tau-bench", _tau)
    add("SWE-agent", lambda: [(swe.to_steps(rec["traj"]), 0 if rec["target"] else 1)
                              for rec in swe.load_runs(swe.N_RUNS)])
    add("SWE-Gym", lambda: [(rec["steps"], 0 if rec["resolved"] else 1)
                            for rec in swegym.load_runs()])
    add("OpenHands", lambda: [(oh.to_steps(rec["traj"]), 0 if rec["resolved"] else 1)
                              for rec in oh.load_runs(oh.N_RUNS)])
    add("tau2-bench", lambda: [(rec["steps"], rec["label"]) for rec in tau2.load_runs()])
    add("web", lambda: [(web.to_steps(rec["steps"]), rec["label"])
                        for rec in web.load_runs(web.N_RUNS)])
    return P, order


def _build(pairs, conv):
    data, y = [], []
    for steps, label in pairs:
        if len(steps) < 2:
            continue
        g = build_graph(steps, dependency=DEP_MODE, shared_resource=False)
        data.append(conv(g, label))
        y.append(label)
    return data, np.array(y)


def main():
    np.random.seed(0)
    P, order = _pairs()
    if not order:
        print("no corpora loaded; check data paths", flush=True)
        return

    # build both representations once per corpus
    HOM, HET_D, Y = {}, {}, {}
    for nm in order:
        HOM[nm], Y[nm] = _build(P[nm], to_homog)
        HET_D[nm], _ = _build(P[nm], to_hetero)
        nf = int(Y[nm].sum())
        print(f"built {nm}: {len(Y[nm])} runs ({nf} failed)", flush=True)
    order = [nm for nm in order if min(int(Y[nm].sum()), len(Y[nm]) - int(Y[nm].sum())) >= 5]

    def store(nm, rep):
        return HET_D[nm] if rep == "hetero" else HOM[nm]

    models = [m for m in _MODELS if m in MODEL_SPEC]
    print(f"\nmodels={models}  seeds={N_SEEDS}  epochs={EPOCHS}  dep={DEP_MODE}  device={DEVICE}  corpora={order}", flush=True)

    print("\n=== WITHIN-CORPUS 5-fold ROC-AUC (mean over %d seeds, seed-block 95%% CI) ==="
          % N_SEEDS, flush=True)
    header = "%-12s" % "corpus" + "".join("  %18s" % m for m in models)
    print(header, flush=True)
    within = {m: {} for m in models}
    for nm in order:
        cells = []
        for m in models:
            make, rep = MODEL_SPEC[m]
            data = store(nm, rep)
            y = Y[nm]
            seed_means = []
            for s in range(N_SEEDS):
                fold_aucs = []
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(np.zeros(len(y)), y):
                    try:
                        fold_aucs.append(train_eval(make, [data[i] for i in tr],
                                                    [data[i] for i in te], seed=s))
                    except Exception as e:
                        print(f"  !! {m}/{nm} within seed{s}: {type(e).__name__}: {str(e)[:80]}", flush=True)
                if fold_aucs:
                    seed_means.append(float(np.mean(fold_aucs)))
            mean, half = _ci(seed_means) if seed_means else (float("nan"), 0.0)
            within[m][nm] = (mean, half)
            cells.append("%8.3f +/-%5.3f" % (mean, half))
        print("%-12s" % nm + "".join("  %18s" % c for c in cells), flush=True)

    print("\n=== POOLED LEAVE-ONE-CORPUS-OUT TRANSFER ROC-AUC (mean over %d seeds, 95%% CI) ==="
          % N_SEEDS, flush=True)
    print(header, flush=True)
    transfer = {m: {} for m in models}
    for held in order:
        tr_names = [nm for nm in order if nm != held]
        cells = []
        for m in models:
            make, rep = MODEL_SPEC[m]
            tr_data = [d for nm in tr_names for d in store(nm, rep)]
            te_data = store(held, rep)
            aucs = []
            for s in range(N_SEEDS):
                try:
                    aucs.append(train_eval(make, tr_data, te_data, seed=s))
                except Exception as e:
                    print(f"  !! {m}/{held} transfer seed{s}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            mean, half = _ci(aucs) if aucs else (float("nan"), 0.0)
            transfer[m][held] = (mean, half)
            cells.append("%8.3f +/-%5.3f" % (mean, half))
        print("%-12s" % held + "".join("  %18s" % c for c in cells), flush=True)

    print("\nReading: GIN is edge-type-blind; R-GCN and HGT read the relation. If the "
          "relation-aware models still do not match the hand-designed source-aware logistic "
          "features (keystone/gating tables), the 'off-the-shelf graph networks misread these "
          "runs' claim generalizes beyond GIN. If they DO match once given the edge-type "
          "channel, the claim narrows to edge-type-blind models.", flush=True)


if __name__ == "__main__":
    main()
