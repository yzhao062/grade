"""Graph-native test: does a GNN on the FULL two-layer graph beat the 8-aggregate logistic
regression at failure prediction, and does it transfer across agent classes? This probes the
hypothesis that compressing the graph to a handful of size/shape scalars undersells the
representation (the "our solution was wrong, not the data" hypothesis).

A GIN reads node types (agent / decision / tool_call / resource), the read/write tag, and
position, over the combined execution + dependency edges, with mean+max readout to a graph
embedding, then a linear head. Reported per corpus: within-corpus 5-fold AUC (vs the aggregate
baseline printed by agent_failure_detection.py) and pooled leave-one-corpus-out transfer AUC.

Run from the repository root (CPU is fine; the graphs are tiny):
  python experiment/graph_gnn.py
"""
import os

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

import agent_failure_detection as H

NTYPES = ["agent", "decision", "tool_call", "dependency_resource"]


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


LOADERS = [("tau-bench", H.load_tau), ("SWE-agent", H.load_swe), ("SWE-Gym", H.load_swegym),
           ("OpenHands", H.load_openhands), ("tau2-bench", H.load_tau2), ("web", H.load_web)]

np.random.seed(0)
ALL = {}
for name, loader in LOADERS:
    try:
        graphs, labels = loader()
    except Exception as e:  # e.g. web data path not present
        print(f"skip {name}: {type(e).__name__}: {str(e)[:80]}", flush=True)
        continue
    if min(int(np.sum(labels)), len(labels) - int(np.sum(labels))) < 5:
        print(f"skip {name}: one class too small", flush=True)
        continue
    ALL[name] = ([to_pyg(G, l) for G, l in zip(graphs, labels)], np.array(labels))
    print(f"loaded {name}: {len(labels)} runs", flush=True)

names = list(ALL.keys())
if not names:
    raise SystemExit("no corpora loaded; check data paths")
IN_DIM = ALL[names[0]][0][0].x.shape[1]

print("\n== within-corpus GIN 5-fold AUC (compare to aggregate flat+dep baseline) ==", flush=True)
for name in names:
    data, y = ALL[name]
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y):
        aucs.append(train_eval([data[i] for i in tr], [data[i] for i in te], IN_DIM))
    print(f"{name:12s} GIN {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}", flush=True)

print("\n== pooled leave-one-corpus-out transfer (GIN) ==", flush=True)
for held in names:
    tr_data = [d for n in names if n != held for d in ALL[n][0]]
    auc = train_eval(tr_data, ALL[held][0], IN_DIM)
    print(f"hold out {held:12s} GIN transfer AUC {auc:.3f}", flush=True)
