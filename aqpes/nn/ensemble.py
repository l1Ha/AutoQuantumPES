"""共享原子能量委员会 (committee) — 严格置换不变的 NN 势能面代理,
并以委员会标准差给出数据外 (OOD) 不确定性, 作为主动学习的采样信号。

架构: E(coords) = Σ_i E_atom(Φ_i(coords)), E_atom 为所有原子共享的
小型 MLP (同种原子参数共享)。原子置换只置换求和项 → 总能量按构造
严格不变, 平移/旋转不变性由对称函数保证。

不确定性: 训练 n 个不同随机种子的成员, 总能量的委员会标准差在训练
分布外显著升高 — 主动学习据此挑选下一批采样点 (D-optimality、GHOST、
委员会分歧等策略均建立在这个信号上)。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from aqpes.nn.model import FeedForwardNN
from aqpes.nn.optim import Adam
from aqpes.nn.symmetry import SymmetryFunctionSet, SymmetryFunctionParams


@dataclass
class AtomicTrainingConfig:
    """共享原子能量委员会的训练超参数 (数据类)。

    Attributes:
        hidden_layers: 原子网络隐藏层宽度。
        lr: 学习率 (总能量损失下 0.01 会发散)。
        epochs: 每个成员的训练轮数。
        normalize: 是否标准化特征 (统计量随委员会持久化)。
    """
    hidden_layers: Tuple[int, ...] = (64, 64)
    # 0.01 在总能量损失下会发散; 0.005 经玩具系统验证稳定收敛
    lr: float = 0.005
    epochs: int = 500
    normalize: bool = True


class AtomicEnergyCommittee:
    """对称函数 + 共享原子能量委员会。"""

    def __init__(self, symbols: Sequence[str],
                 sf_params: Optional[SymmetryFunctionParams] = None,
                 members: Optional[List[FeedForwardNN]] = None,
                 feature_mean: Optional[np.ndarray] = None,
                 feature_scale: Optional[np.ndarray] = None,
                 energy_scale: float = 1.0,
                 energy_offset: float = 0.0):
        self.symbols = list(symbols)
        self.sf = SymmetryFunctionSet(symbols, sf_params)
        self.members: List[FeedForwardNN] = members or []
        # 特征/能量归一化常数 — 预测路径必须复用训练时的统计量
        self.feature_mean = feature_mean
        self.feature_scale = feature_scale
        self.energy_scale = float(energy_scale)
        self.energy_offset = float(energy_offset)

    # ------------------------------------------------------------------
    def _standardize(self, feats: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_scale is None:
            return feats
        return (feats - self.feature_mean) / self.feature_scale

    def atomic_energies(self, coords: np.ndarray) -> np.ndarray:
        """每个成员的原子能量 (n_m, n, N), 物理单位。"""
        feats = self.sf.compute(coords)                    # (n, N, F)
        flat = self._standardize(feats).reshape(-1, feats.shape[-1])
        out = []
        for m in self.members:
            e = m.predict(flat).reshape(feats.shape[:2])  # (n, N) 标准化单位
            e = e * self.energy_scale + self.energy_offset
            out.append(e)
        return np.stack(out)                              # (n_m, n, N)

    def predict(self, coords: np.ndarray) -> np.ndarray:
        """总能量均值 (n,) — 对原子置换严格不变。"""
        return self.atomic_energies(coords).sum(axis=-1).mean(axis=0)

    def predict_with_uncertainty(self, coords: np.ndarray
                                 ) -> Tuple[np.ndarray, np.ndarray]:
        """委员会总能量均值与标准差 (OOD 不确定性信号)。

        Args:
            coords: (n, N, 3) 或 (N, 3) 原子坐标 (Bohr)。

        Returns:
            (均值 (n,), 标准差 (n,)); 单成员时标准差为全零。
        """
        totals = self.atomic_energies(coords).sum(axis=-1)  # (n_m, n)
        return totals.mean(axis=0), totals.std(axis=0, ddof=0) \
            if totals.shape[0] > 1 else np.zeros(totals.shape[1])

    def std(self, coords: np.ndarray) -> np.ndarray:
        """委员会标准差 (n,) — 主动学习的采样信号。"""
        return self.predict_with_uncertainty(coords)[1]

    def save(self, path: str) -> None:
        """保存委员会 (成员用可 pickle 的原始数组, 避开 lambda 属性)。"""
        import pickle
        payload = {
            "symbols": self.symbols,
            "sf_params": self.sf.params,
            "feature_mean": self.feature_mean,
            "feature_scale": self.feature_scale,
            "energy_scale": self.energy_scale,
            "energy_offset": self.energy_offset,
            "members": [{
                "layers": m.layers,
                "activation": m.activation_name,
                "weights": m.weights,
                "biases": m.biases,
                "x_mean": m.x_mean,
                "x_scale": m.x_scale,
                "y_mean": m.y_mean,
                "y_scale": m.y_scale,
            } for m in self.members],
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "AtomicEnergyCommittee":
        """从 pickle 文件恢复委员会 (含对称函数参数与归一化统计量)。

        Args:
            path: ``save`` 写出的文件路径。

        Returns:
            AtomicEnergyCommittee: 恢复的委员会实例。
        """
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
        members = []
        for md in d["members"]:
            m = FeedForwardNN(md["layers"], md["activation"])
            m.weights = md["weights"]
            m.biases = md["biases"]
            m.x_mean, m.x_scale = md["x_mean"], md["x_scale"]
            m.y_mean, m.y_scale = md["y_mean"], md["y_scale"]
            members.append(m)
        return cls(d["symbols"], d["sf_params"], members,
                   feature_mean=d.get("feature_mean"),
                   feature_scale=d.get("feature_scale"),
                   energy_scale=d.get("energy_scale", 1.0),
                   energy_offset=d.get("energy_offset", 0.0))


def _atomic_grads(model: FeedForwardNN, feats_flat: np.ndarray,
                  sample_weight: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """共享原子能量的样本加权梯度。

    总能量 E = Σ_i out_i, 损失 L = mean((E·scale − y)²), 则
    dL/dθ = (2·scale/n)·Σ_i (E−y)·∂out_i/∂θ。实现: 输出层为线性,
    其伴随就是逐样本权重 s_i (含残差与 2·scale/n), 再标准反传。
    """
    a, z = model.forward(feats_flat)
    delta = sample_weight[:, None]          # 线性输出层: ∂out/∂z = 1
    dw = [None] * len(model.weights)
    db = [None] * len(model.biases)
    dw[-1] = a[-2].T @ delta
    db[-1] = np.sum(delta, axis=0, keepdims=True)
    for i in range(len(model.weights) - 2, -1, -1):
        delta = (delta @ model.weights[i + 1].T) * model.activation_deriv(z[i])
        dw[i] = a[i].T @ delta
        db[i] = np.sum(delta, axis=0, keepdims=True)
    return dw, db


def train_atomic_committee(symbols: Sequence[str],
                           coords: np.ndarray,
                           energies: np.ndarray,
                           n_models: int = 4,
                           config: Optional[AtomicTrainingConfig] = None,
                           sf_params: Optional[SymmetryFunctionParams] = None,
                           seed: int = 0) -> Tuple[AtomicEnergyCommittee, Dict]:
    """在总能量标签上训练共享原子能量委员会 (置换严格不变)。

    梯度通过"原子能量求和"链式法则: 对每个成员的原子网络, 以
    ∂E/∂(原子输出)=1 反传并求和。
    """
    config = config or AtomicTrainingConfig()
    c = np.asarray(coords, dtype=float)
    if c.ndim == 2:
        c = c[None]
    y = np.asarray(energies, dtype=float).ravel()
    if y.size != c.shape[0]:
        raise ValueError(f"能量数 {y.size} 与几何数 {c.shape[0]} 不符")
    if not np.all(np.isfinite(y)):
        raise ValueError("能量标签含非有限值")

    sf = SymmetryFunctionSet(symbols, sf_params)
    feats_all = sf.compute(c)                     # (n, N, F)
    n, N, F = feats_all.shape
    feats_flat = feats_all.reshape(n * N, F)
    E_per_center = np.repeat(y[:, None], N, axis=1)  # 仅用于归一化统计
    f_mean = feats_flat.mean(axis=0)
    f_scale = feats_flat.std(axis=0)
    # 近常数通道 (如无邻居的角度项) 的 std 可低至 1e-10: 直接除会把
    # 标准化特征放大到 1e10, tanh 饱和后训练彻底停滞
    f_scale = np.where(f_scale < 1e-8, 1.0, f_scale)
    e_mean, e_scale = float(y.mean()), float(y.std() or 1.0)

    members: List[FeedForwardNN] = []
    history: List[float] = []
    for mi in range(n_models):
        model = FeedForwardNN([F] + list(config.hidden_layers) + [1],
                              seed=seed + 1000 * mi)
        params = model.weights + model.biases
        opt = Adam([p.shape for p in params], lr=config.lr)
        feat_std = (feats_flat - f_mean) / f_scale
        for _ in range(config.epochs):
            # 前向: 标准化特征 → 原子能量 → 求和
            out = model.predict(feat_std).reshape(n, N)  # (n,N)
            total = out.sum(axis=1)
            pred_total = total * e_scale + N * e_mean
            loss = float(np.mean((pred_total - y) ** 2))
            # 反向: 样本权重 s_i = (E−y)·2·e_scale/n (逐中心复制)
            s = (pred_total - y) * (2.0 * e_scale / n)
            dw, db = _atomic_grads(model, feat_std, np.repeat(s, N))
            opt.step(model.weights + model.biases, dw + db)
        members.append(model)
        history.append(loss)

    committee = AtomicEnergyCommittee(
        symbols, sf_params, members,
        feature_mean=f_mean, feature_scale=f_scale,
        energy_scale=e_scale, energy_offset=e_mean)
    pred = committee.predict(c)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return committee, {"rmse": rmse, "final_losses": history,
                       "n_features": F, "n_centers": N}
