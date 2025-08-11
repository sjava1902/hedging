# optimizer.py
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from scenarios import build_tree

SWAP_FLOAT_TERM = 3

def swap_coupon_quarter(notional: float, fixed_rate: float, float_rate_q: float, direction: str) -> float:
    fixed_leg = notional * fixed_rate / 4.0
    float_leg = notional * float_rate_q / 4.0
    if direction == "pay_fixed":
        return float_leg - fixed_leg
    else:
        return fixed_leg - float_leg

def swap_fixed_rate_at_node(node_snapshot: Dict, term_months: int) -> float:
    return float(node_snapshot[term_months])

@dataclass
class Decision:
    x_6: float
    x_12: float
    x_24: float

def simulate_terminal_pnl(nodes: List, decision: Decision, notional_unit: float, alpha: float=0.95) -> np.ndarray:
    # найдём листья
    levels = max(n.level for n in nodes) + 1
    leaf_indices = [i for i,n in enumerate(nodes) if n.level == levels-1]

    # фиксированные ставки на корне
    root = [n for n in nodes if n.level == 0][0]
    r_fix = {6: float(root.gcurve_snapshot[6]),
             12: float(root.gcurve_snapshot[12]),
             24: float(root.gcurve_snapshot[24])}

    # родительские связи
    parent = {i: nodes[i].parent for i in range(len(nodes))}
    pnl = []
    for leaf in leaf_indices:
        # путь корень->лист
        path = []
        cur = leaf
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path = path[::-1]

        acc = 0.0
        for s in range(1, len(path)):
            p_idx = path[s-1]; c_idx = path[s]
            p_node = nodes[p_idx]
            r_flt = float(p_node.gcurve_snapshot[SWAP_FLOAT_TERM])

            # разложим знак на направление ножек
            dir6  = 'receive_fixed' if decision.x_6  >= 0 else 'pay_fixed'
            dir12 = 'receive_fixed' if decision.x_12 >= 0 else 'pay_fixed'
            dir24 = 'receive_fixed' if decision.x_24 >= 0 else 'pay_fixed'

            c6  = swap_coupon_quarter(abs(decision.x_6),  r_fix[6],  r_flt, dir6)
            c12 = swap_coupon_quarter(abs(decision.x_12), r_fix[12], r_flt, dir12)
            c24 = swap_coupon_quarter(abs(decision.x_24), r_fix[24], r_flt, dir24)

            coupon = c6 + c12 + c24
            acc = (acc + coupon) * nodes[c_idx].acc_mult_to_child
        pnl.append(acc)
    return np.array(pnl, dtype=float)

def cvar_of_losses(losses: np.ndarray, alpha: float = 0.95) -> Tuple[float, float]:
    """
    CVaR_α = E[ Loss | Loss >= VaR_α ]. Возвращает (CVaR, VaR).
    """
    x = np.sort(np.asarray(losses, dtype=float))
    S = x.size
    if S == 0:
        return 0.0, 0.0
    k = int(np.ceil(alpha * S)) - 1
    k = max(0, min(S-1, k))
    var = x[k]
    tail = x[k:]
    cvar = float(tail.mean()) if tail.size else float(var)
    return cvar, float(var)

def grid_search_cvar(nodes: List, notional_unit: float, alpha: float = 0.95,
                     mu: float = 0.0, max_abs_units: int = 2) -> Tuple[Decision, dict]:
    """
    Грубый, но беззависимый от внешних либ грид-поиск по x_6,x_12,x_24 (в «юнитах»).
    Возвращает Decision в НОМИНАЛАХ (x_T * notional_unit) и метрики.
    """
    best_score = None
    best_dec = Decision(0.0, 0.0, 0.0)
    tried = 0
    for n6 in range(-max_abs_units, max_abs_units+1):
        for n12 in range(-max_abs_units, max_abs_units+1):
            for n24 in range(-max_abs_units, max_abs_units+1):
                if n6 == 0 and n12 == 0 and n24 == 0:
                    continue
                # переведём в НОМИНАЛЫ
                dec = Decision(n6 * notional_unit, n12 * notional_unit, n24 * notional_unit)
                pnl = simulate_terminal_pnl(nodes, dec, notional_unit, alpha)
                mean_pnl = float(np.mean(pnl))
                if mean_pnl < mu:
                    continue
                losses = -pnl
                cvar, _ = cvar_of_losses(losses, alpha)
                score = cvar  # минимизируем хвостовой риск
                if (best_score is None) or (score < best_score) or \
                   (np.isclose(score, best_score) and mean_pnl > 0):
                    best_score = score
                    best_dec = dec
                tried += 1
    info = {"alpha": alpha, "mu": mu, "tried": tried, "best_cvar": best_score}
    return best_dec, info

def rebalance_once(engine,
                   levels: int = 5, branch: int = 5,
                   alpha: float = 0.95, mu: float = 0.0,
                   unit_frac: float = 0.10, max_abs_units: int = 2) -> Decision:
    """
    Точка входа для движка. Строит дерево, делает грид-поиск CVaR и
    возвращает Decision (номиналы) для добавления свопов.
    unit_frac — доля от суммарного V портфеля на 1 «юнит»;
    max_abs_units — предел по |юнитам| на срок.
    """
    nodes = build_tree(engine.gcurve, levels=levels, branch=branch)
    V = getattr(engine.portfolio, "V", 1_000_000.0)
    notional_unit = float(V) * float(unit_frac)
    decision, info = grid_search_cvar(nodes, notional_unit, alpha=alpha, mu=mu, max_abs_units=max_abs_units)
    # можно временно распечатать инфо:
    # print("Rebalance info:", info, "Decision:", decision)
    return decision
