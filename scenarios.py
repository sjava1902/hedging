# scenarios.py
from dataclasses import dataclass
from typing import List, Optional
from copy import deepcopy
from gcurve import GCurve
from datetime import timedelta

QUARTER_LEN_DAYS = 91

@dataclass
class Node:
    level: int           # 0..5 (6 этапов)
    parent: Optional[int]
    date: object         # datetime
    gcurve_snapshot: dict
    acc_mult_to_child: float  # множитель наращения до следующего узла (1 + r_1y/4)

def build_tree(g: GCurve, levels: int = 6, branch: int = 10) -> List[Node]:
    nodes: List[Node] = []
    idx_by_level = []  # списки индексов узлов каждого уровня
    # корень
    nodes.append(Node(0, None, g.t_curr, g.snapshot(), 1.0))
    idx_by_level.append([0])

    for L in range(1, levels):
        idx_by_level.append([])
        new_nodes = []
        for p_idx in idx_by_level[L-1]:
            # из снапшота восстановим временную кривую для ветвления
            base = {m: nodes[p_idx].gcurve_snapshot[m] for m in [0,3,6,12,24]}
            # для каждой ветки копируем кривую и «прокручиваем» квартал случайно
            for _ in range(branch):
                g_local = GCurve(nodes[p_idx].date, base, seed=None)  # seed=None для разнообразия
                g_local.step(QUARTER_LEN_DAYS)
                snap = g_local.snapshot()
                # множитель наращения: 1 + r_1y(parent)/4
                r1y_parent = nodes[p_idx].gcurve_snapshot[12]
                acc_mult = 1.0 + float(r1y_parent) / 4.0
                node = Node(L, p_idx, g_local.t_curr, snap, acc_mult)
                idx_by_level[L].append(len(nodes) + len(new_nodes))
                new_nodes.append(node)
        nodes.extend(new_nodes)
    return nodes
