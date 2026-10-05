"""主动学习闭环 — 委员会分歧选点 → 电子结构后端标注 → 增量训练。

工作流 (每轮):
    1. 在已标注集上训练共享原子能量委员会 (n 个随机种子成员);
    2. 对候选池计算委员会标准差 (数据外不确定性);
    3. 挑选标准差最大的 batch_size 个候选 (排除已标注);
    4. 用电子结构后端 (Calculator) 标注新点, 加入训练集。

终止: 达到 n_iterations 或候选耗尽。每轮记录训练集规模、候选池
RMSE (委员会看不到池内真值, 仅作诊断) 与不确定性统计 — 这是
D-optimality/GHOST 等采样策略的最简实现基座。

参考文献: Behler JCP 152, 079201 (2020) (主动学习综述章节)。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from aqpes.nn.ensemble import (
    AtomicEnergyCommittee, AtomicTrainingConfig, train_atomic_committee)
from aqpes.nn.symmetry import SymmetryFunctionParams
from aqpes.core.validation import data_fingerprint, environment_info


@dataclass
class ActiveLearningResult:
    committee: AtomicEnergyCommittee
    labeled_coords: np.ndarray          # (n_labeled, N, 3)
    labeled_energies: np.ndarray        # (n_labeled,)
    history: List[Dict[str, Any]] = field(default_factory=list)
    pool_rmse_history: List[float] = field(default_factory=list)

    @property
    def final_rmse(self) -> Optional[float]:
        return self.pool_rmse_history[-1] if self.pool_rmse_history else None


def run_active_learning(symbols: Sequence[str],
                        calculator,
                        initial_coords: np.ndarray,
                        candidate_pool: np.ndarray,
                        n_iterations: int = 5,
                        batch_size: int = 8,
                        n_models: int = 4,
                        training_config: Optional[AtomicTrainingConfig] = None,
                        sf_params: Optional[SymmetryFunctionParams] = None,
                        seed: int = 0,
                        verbose: bool = True) -> ActiveLearningResult:
    """委员会分歧驱动的主动学习循环。

    Parameters
    ----------
    calculator : Calculator
        标注后端 (只需 ``energy(coords)``; 有梯度时未来可接力力训练)。
    initial_coords : (n0, N, 3)
        初始标注集 (建议覆盖训练分布; 由 sample_geometries 生成)。
    candidate_pool : (n_c, N, 3)
        候选池 — 循环从中挑点。池内真值仅用于 RMSE 诊断, 委员会看不到。
    """
    training_config = training_config or AtomicTrainingConfig()
    c0 = np.asarray(initial_coords, dtype=float)
    pool = np.asarray(candidate_pool, dtype=float)
    if c0.ndim == 2:
        c0 = c0[None]
    if pool.ndim == 2:
        pool = pool[None]
    if c0.shape[1:] != pool.shape[1:]:
        raise ValueError(
            f"初始集 {c0.shape} 与候选池 {pool.shape} 的 (N, 3) 不一致")

    def label(coords: np.ndarray) -> np.ndarray:
        return np.array([calculator.energy(c) for c in coords], dtype=float)

    labeled_c = list(c0)
    labeled_e = list(label(c0))
    pool_true = label(pool)                     # 诊断用真值 (不进训练)
    labeled_idx: set = set()
    history: List[Dict[str, Any]] = []
    pool_rmse_history: List[float] = []
    committee: Optional[AtomicEnergyCommittee] = None

    for it in range(n_iterations):
        t0 = time.time()
        lc = np.stack(labeled_c)
        le = np.array(labeled_e)
        committee, info = train_atomic_committee(
            symbols, lc, le, n_models=n_models,
            config=training_config, sf_params=sf_params,
            seed=seed + 100 * it)

        pool_pred = committee.predict(pool)
        pool_rmse = float(np.sqrt(np.mean((pool_pred - pool_true) ** 2)))
        pool_std = committee.std(pool)
        pool_rmse_history.append(pool_rmse)

        # 挑选: 候选池中标准差最大且未标注过的 batch_size 个
        pick_order = np.argsort(-pool_std)
        picked = [int(j) for j in pick_order
                  if j not in labeled_idx][:batch_size]
        if not picked:
            if verbose:
                print(f"  [{it}] 候选池耗尽, 提前结束")
            break
        for j in picked:
            labeled_idx.add(j)
            labeled_c.append(pool[j])
            labeled_e.append(pool_true[j])

        history.append({
            "iteration": it,
            "n_labeled": len(labeled_c),
            "pool_rmse": pool_rmse,
            "pool_std_mean": float(pool_std.mean()),
            "pool_std_max": float(pool_std.max()),
            "picked_indices": picked,
            "seconds": round(time.time() - t0, 2),
        })
        if verbose:
            print(f"  [{it}] 标注 {len(labeled_c)} | 池 RMSE {pool_rmse:.4e} | "
                  f"池 std 均值 {pool_std.mean():.4e}")

    lc = np.stack(labeled_c)
    return ActiveLearningResult(
        committee=committee,
        labeled_coords=lc,
        labeled_energies=np.array(labeled_e),
        history=history,
        pool_rmse_history=pool_rmse_history)
