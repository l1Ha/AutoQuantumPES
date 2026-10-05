import numpy as np
from typing import Callable, Dict, Any, Tuple, Optional, Sequence


class AbInitioData:
    """PES 训练数据容器 (点 / 能量 / 可选梯度)。

    注意: 本类**不执行**电子结构计算 — 它只承载外部来源 (解析面、
    文件、量子化学程序) 给出的数据。用 ``sample_function`` 可从任意
    可调用势能函数生成带有限差分梯度的训练集。
    """

    def __init__(self):
        self.points: Optional[np.ndarray] = None
        self.energies: Optional[np.ndarray] = None
        self.gradients: Optional[np.ndarray] = None

    def from_arrays(self, points: np.ndarray, energies: np.ndarray,
                    gradients: Optional[np.ndarray] = None):
        """从数组直接填充数据 (points 一维输入时重塑为 (n, 1))。

        Args:
            points: (n, d) 采样点 (物理单位)。
            energies: (n,) 能量 (Hartree)。
            gradients: 可选 (n, d) 梯度 (Hartree/Bohr)。
        """
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(-1, 1)
        self.points = points
        self.energies = np.asarray(energies, dtype=float).ravel()
        self.gradients = (np.asarray(gradients, dtype=float)
                          if gradients is not None else None)

    def from_analytic_pes(self, builder, grid: np.ndarray,
                          noise_level: float = 0.0) -> "AbInitioData":
        """从解析 PES builder 的一维网格评估生成数据集。

        Args:
            builder: 提供 ``evaluate(grid)`` 的解析势对象。
            grid: 一维采样网格 (Bohr)。
            noise_level: 可选高斯噪声幅度 (乘以 max|V|)。

        Returns:
            self (支持链式调用)。
        """
        values = builder.evaluate(grid)
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * np.max(values), size=values.shape)
            values = values + noise
        self.points = grid
        self.energies = values
        return self

    @classmethod
    def sample_function(cls, func: Callable,
                        ranges: Sequence[Tuple[float, float]],
                        n_per_dim: int,
                        noise_level: float = 0.0,
                        grad_h: float = 1e-5,
                        seed: Optional[int] = None) -> "AbInitioData":
        """在规则网格上采样标量函数 V(x)，生成带中心差分梯度的训练集。

        Parameters
        ----------
        func : Callable
            接受 (n, d) 点数组、返回 (n,) 能量的可调用对象,
            例如 ``lambda pts: pes(pts[:, 0], pts[:, 1])``。
        ranges : [(lo, hi), ...]
            每维采样范围, d = len(ranges)。
        n_per_dim : int
            每维网格点数 (总点数 n_per_dim**d)。
        noise_level : float
            加在能量上的高斯噪声幅度 (乘以 max|V|)。
        grad_h : float
            中心差分步长 (能量单位)。
        """
        if n_per_dim < 2:
            raise ValueError("n_per_dim 必须 >= 2 (差分需要)")
        d = len(ranges)
        axes = [np.linspace(lo, hi, n_per_dim) for lo, hi in ranges]
        mesh = np.meshgrid(*axes, indexing="ij")
        points = np.column_stack([m.ravel() for m in mesh])
        values = np.asarray(func(points), dtype=float).ravel()

        gradients = np.zeros_like(points)
        for j in range(d):
            hp = np.zeros(d); hp[j] = grad_h
            vp = np.asarray(func(points + hp), dtype=float).ravel()
            vm = np.asarray(func(points - hp), dtype=float).ravel()
            gradients[:, j] = (vp - vm) / (2.0 * grad_h)

        if noise_level > 0:
            rng = np.random.RandomState(seed)
            values = values + rng.normal(0.0, noise_level * np.max(np.abs(values)),
                                         size=values.shape)

        data = cls()
        data.from_arrays(points, values, gradients)
        return data

    @property
    def n_points(self) -> int:
        """数据点数; 未填充数据时为 0。"""
        return len(self.points) if self.points is not None else 0

    @classmethod
    def sample_geometries(cls, calculator,
                          ref_coords: np.ndarray,
                          ranges: Tuple[Tuple[float, float], ...],
                          n_per_dim: int,
                          active_atoms: Optional[Sequence[int]] = None,
                          axes: Sequence[int] = (0, 1, 2),
                          min_distance: float = 1.2,
                          max_points: int = 5000,
                          seed: int = 42,
                          verbose: bool = False) -> "AbInitioData":
        """用电子结构后端在笛卡尔位移网格上采样 (能量+梯度)。

        采样方案: 参考几何 ``ref_coords`` (N,3) Bohr, 指定原子在指定
        坐标轴方向按 ``ranges``/``n_per_dim`` 均匀位移; 剔除原子间距
        小于 ``min_distance`` 的坍缩几何; 均匀子采样至 ``max_points``。

        返回的 ``points`` 为**笛卡尔坐标展平** (N*3,) —— 直接作为 NN
        训练特征 (无置换对称性处理, 见 README 限制)。
        """
        from itertools import product

        ref = np.asarray(ref_coords, dtype=float)
        n_atoms = ref.shape[0]
        atoms = list(active_atoms) if active_atoms is not None \
            else list(range(n_atoms))
        dims = [(a, ax) for a in atoms for ax in axes]
        if not dims:
            raise ValueError("active_atoms/axes 未选出任何采样维度")
        if len(ranges) == 1:
            ranges = tuple(ranges) * len(dims)
        if len(ranges) != len(dims):
            raise ValueError(f"ranges 长度 {len(ranges)} 与采样维度数 "
                             f"{len(dims)} 不符 (单元素范围可广播)")

        axes_vals = [np.linspace(lo, hi, n_per_dim) for lo, hi in ranges]
        grids = np.meshgrid(*axes_vals, indexing="ij")
        combos = np.stack([g.ravel() for g in grids], axis=1)  # (M, n_dim)
        n_raw = combos.shape[0]

        coords_list = np.repeat(ref[None, :, :], n_raw, axis=0)
        for k, (atom, ax) in enumerate(dims):
            coords_list[:, atom, ax] += combos[:, k]

        # 最小原子间距过滤
        d2 = ((coords_list[:, :, None, :] - coords_list[:, None, :, :]) ** 2
              ).sum(-1)
        iu = np.triu_indices(n_atoms, k=1)
        pair_d2 = d2[:, iu[0], iu[1]]                  # (M, n_pairs)
        keep = np.all(pair_d2 >= min_distance ** 2, axis=1)   # (M,)
        coords_list = coords_list[keep]
        if coords_list.shape[0] == 0:
            raise ValueError("min_distance 过滤后无剩余几何; 请放宽阈值")

        if max_points and coords_list.shape[0] > max_points:
            rng = np.random.RandomState(seed)
            idx = rng.choice(coords_list.shape[0], max_points, replace=False)
            coords_list = coords_list[np.sort(idx)]

        n = coords_list.shape[0]
        energies = np.zeros(n)
        gradients = np.zeros_like(coords_list)
        for i in range(n):
            e, g = calculator.energy_and_gradient(coords_list[i])
            energies[i] = e
            gradients[i] = g
            if verbose and (i + 1) % max(1, n // 10) == 0:
                print(f"  sampled {i + 1}/{n}")

        points = coords_list.reshape(n, -1)  # (n, 3N) 笛卡尔特征
        data = cls()
        data.from_arrays(points, energies, gradients)
        data.provenance = dict(getattr(calculator, "provenance", {}))
        data.geometry = coords_list.copy()
        return data

    def save_npz(self, path: str, provenance: Optional[Dict] = None,
                 geometry: Optional[np.ndarray] = None):
        """保存数据集 (points/energies/gradients + provenance JSON)。"""
        import json
        if self.points is None:
            raise ValueError("无数据可保存")
        arrays = {"points": self.points, "energies": self.energies}
        if self.gradients is not None:
            arrays["gradients"] = self.gradients
        symbols = getattr(self, "symbols", None)
        if symbols is not None:
            arrays["symbols"] = np.array(list(symbols))
        prov = provenance or getattr(self, "provenance", None)
        if prov:
            arrays["provenance_json"] = np.array(json.dumps(prov))
        geom = geometry if geometry is not None else getattr(self, "geometry", None)
        if geom is not None:
            arrays["geometry"] = geom
        np.savez(path, **arrays)

    @classmethod
    def load_npz(cls, path: str) -> "AbInitioData":
        """加载 ``save_npz`` 保存的数据集 (含 provenance/geometry/symbols)。

        Args:
            path: .npz 文件路径。

        Returns:
            AbInitioData: 恢复的数据容器。
        """
        import json
        with np.load(path, allow_pickle=False) as f:
            data = cls()
            data.from_arrays(f["points"], f["energies"],
                             f["gradients"] if "gradients" in f else None)
            if "provenance_json" in f:
                data.provenance = json.loads(str(f["provenance_json"]))
            if "geometry" in f:
                data.geometry = f["geometry"]
            if "symbols" in f:
                data.symbols = [str(x) for x in f["symbols"]]
        return data

    def split(self, train_ratio: float = 0.8) -> Tuple["AbInitioData", "AbInitioData"]:
        """随机打乱并按比例切分为训练/测试两个数据容器。

        Args:
            train_ratio: 训练集比例。

        Returns:
            (train, test) 两个 AbInitioData (含梯度, 若原始数据有)。
        """
        n = self.n_points
        indices = np.random.permutation(n)
        n_train = int(n * train_ratio)

        train = AbInitioData()
        test = AbInitioData()

        train.points = self.points[indices[:n_train]]
        train.energies = self.energies[indices[:n_train]]
        if self.gradients is not None:
            train.gradients = self.gradients[indices[:n_train]]
        test.points = self.points[indices[n_train:]]
        test.energies = self.energies[indices[n_train:]]
        if self.gradients is not None:
            test.gradients = self.gradients[indices[n_train:]]

        return train, test
