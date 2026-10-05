import numpy as np
from typing import List, Optional, Callable


class FeedForwardNN:
    """纯 NumPy 前馈网络 (回归: 线性输出层)。

    支持任意维输入 ``layers[0] = d``。内置可选的输入/输出归一化
    (由 ``NNTrainer`` 在训练数据上拟合): ``predict``/``gradient``
    接受**物理单位**数据, 内部自动标准化/反标准化; ``train_step``
    则始终作用于原始 (标准化后) 数据, 由训练器负责喂标准化数组。

    ``gradient(X)`` 返回解析输入梯度 dŷ/dX (经反标准化链式修正),
    与有限差分校验一致 — 为后续力训练提供基础。
    """

    def __init__(self, layers: List[int],
                 activation: str = "tanh",
                 seed: Optional[int] = None):
        self.layers = list(layers)
        self.activation_name = activation
        self.activation_fn = self._get_activation(activation)
        self.activation_deriv = self._get_derivative(activation)
        self.activation_second_deriv = self._get_second_derivative(activation)

        if seed is not None:
            np.random.seed(seed)

        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            lim = np.sqrt(6 / (layers[i] + layers[i + 1]))
            self.weights.append(np.random.uniform(-lim, lim, (layers[i], layers[i + 1])))
            self.biases.append(np.zeros((1, layers[i + 1])))

        # 归一化参数 (默认未启用)
        self.x_mean: Optional[np.ndarray] = None
        self.x_scale: Optional[np.ndarray] = None
        self.y_mean: Optional[float] = None
        self.y_scale: Optional[float] = None

    # ------------------------------------------------------------------
    @staticmethod
    def _get_activation(name: str) -> Callable:
        if name == "tanh":
            return lambda x: np.tanh(x)
        elif name == "sigmoid":
            return lambda x: 1 / (1 + np.exp(-np.clip(x, -100, 100)))
        elif name == "relu":
            return lambda x: np.maximum(0, x)
        elif name == "linear":
            return lambda x: x
        else:
            raise ValueError(f"Unknown activation: {name}")

    @staticmethod
    def _get_derivative(name: str) -> Callable:
        if name == "tanh":
            return lambda x: 1 - np.tanh(x) ** 2
        elif name == "sigmoid":
            s = lambda x: 1 / (1 + np.exp(-np.clip(x, -100, 100)))
            return lambda x: s(x) * (1 - s(x))
        elif name == "relu":
            return lambda x: (x > 0).astype(float)
        elif name == "linear":
            return lambda x: np.ones_like(x)
        else:
            raise ValueError(f"Unknown activation: {name}")

    @staticmethod
    def _get_second_derivative(name: str) -> Callable:
        """激活函数的二阶导 (力训练 double-backprop 需要)。"""
        if name == "tanh":
            t = lambda x: np.tanh(x)
            return lambda x: -2.0 * t(x) * (1.0 - t(x) ** 2)
        elif name == "sigmoid":
            s = lambda x: 1 / (1 + np.exp(-np.clip(x, -100, 100)))
            return lambda x: s(x) * (1 - s(x)) * (1 - 2 * s(x))
        elif name == "relu":
            return lambda x: np.zeros_like(x)
        elif name == "linear":
            return lambda x: np.zeros_like(x)
        else:
            raise ValueError(f"Unknown activation: {name}")

    # ------------------------------------------------------------------
    def set_normalization(self, x_mean: np.ndarray, x_scale: np.ndarray,
                          y_mean: float, y_scale: float):
        """设置归一化参数; x_scale/y_scale 为 0 时按 1 处理。"""
        self.x_mean = np.asarray(x_mean, dtype=float).ravel()
        self.x_scale = np.asarray(x_scale, dtype=float).ravel()
        self.x_scale = np.where(self.x_scale == 0, 1.0, self.x_scale)
        self.y_mean = float(y_mean)
        self.y_scale = float(y_scale) if float(y_scale) != 0 else 1.0

    def _shape_input(self, X: np.ndarray) -> np.ndarray:
        """(n,) → (n,1) [d=1] 或 (d,) → (1,d) [d>1]。"""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1) if self.layers[0] == 1 else X.reshape(1, -1)
        return X

    def _normalize_x(self, X: np.ndarray) -> np.ndarray:
        if self.x_mean is None:
            return X
        return (X - self.x_mean) / self.x_scale

    def _denormalize_y(self, y: np.ndarray) -> np.ndarray:
        if self.y_mean is None:
            return y
        return y * self.y_scale + self.y_mean

    # ------------------------------------------------------------------
    def forward(self, X: np.ndarray) -> tuple:
        """原始前向 (不做归一化 — 训练器喂标准化数据)。

        隐藏层施加激活函数, 输出层线性 (PES 值可超出有界激活值域)。
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        activations = [X]
        zs = []
        n_layers = len(self.weights)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ w + b
            zs.append(z)
            if i == n_layers - 1:
                # 线性输出层: PES 值为实数, 可能超出 tanh 等有界激活的值域
                activations.append(z)
            else:
                activations.append(self.activation_fn(z))

        return activations, zs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """物理单位预测。输入 (n, d) / (d,) [单点] / (n,) [d=1]。"""
        X = self._shape_input(X)
        Xn = self._normalize_x(X)
        activations, _ = self.forward(Xn)
        return self._denormalize_y(activations[-1]).ravel()

    def input_gradient(self, X: np.ndarray) -> np.ndarray:
        """原始网络输入梯度 dŷ/dX (不做归一化链式修正), 形状 (n, d)。

        反向传播起始于线性输出 (导数 1), 逐层乘 Wᵀ 与隐藏层激活导数。
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        activations = [X]
        zs = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ w + b
            zs.append(z)
            activations.append(z if i == len(self.weights) - 1
                               else self.activation_fn(z))

        # d ŷ / d x: 从线性输出回传
        g = np.ones_like(zs[-1])
        for i in range(len(self.weights) - 1, -1, -1):
            g = g @ self.weights[i].T
            if i > 0:
                g = g * self.activation_deriv(zs[i - 1])
        return g

    def gradient(self, X: np.ndarray) -> np.ndarray:
        """解析输入梯度 dŷ/dX, 物理单位, 形状 (n, d)。

        经归一化链式修正 y_scale / x_scale; 与中心差分校验一致。
        """
        X = self._shape_input(X)
        Xn = self._normalize_x(X)
        g = self.input_gradient(Xn)
        if self.x_mean is not None:
            g = g * (self.y_scale / self.x_scale)
        return g

    def input_gradient_backward(self, X: np.ndarray,
                                adjG: np.ndarray) -> tuple:
        """力训练核心: 反向的反向 (reverse-over-reverse double backprop)。

        计算 Φ = Σ_ij adjG_ij · (input_gradient(X))_ij 对全部权重/偏置
        的梯度, 返回 (dW, db)。∂Φ/∂θ = Σ adjG · ∂g/∂θ, 其中输入梯度
        g 的计算图 (G/T 状态链) 被再次反向传播; g 通过激活导数 f'(z)
        依赖前向图, 该依赖经 f''(z) 项进入前向图的 bar{z} 并被标准
        反向传播继续处理 (偏置梯度即由此产生)。
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        adjG = np.asarray(adjG, dtype=float)
        if adjG.shape != X.shape:
            raise ValueError(f"adjG 形状 {adjG.shape} 与 X {X.shape} 不符")

        L = len(self.weights)

        # --- 前向 (存 z, a) ---
        activations = [X]
        zs = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ w + b
            zs.append(z)
            activations.append(z if i == L - 1 else self.activation_fn(z))

        # --- pass 1: 输入梯度计算图的状态 ---
        # G[l] = dŷ/dz_l, T[l] = G[l] @ W_lᵀ = dŷ/da_l;  g = T[0]
        G = [None] * L
        T = [None] * L
        G[L - 1] = np.ones_like(zs[-1])
        for l in range(L - 1, -1, -1):
            T[l] = G[l] @ self.weights[l].T
            if l > 0:
                G[l - 1] = T[l] * self.activation_deriv(zs[l - 1])

        # --- pass 2: 对 pass-1 图反向传播, 种子 bar{T_0} = adjG ---
        # 注意: pass-1 的映射是 T = G @ Wᵀ (转置在前向的另一侧),
        # 故 dW = barTᵀ @ G, barG = barT @ W (与前向 backprop 相反)。
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        barT = [None] * L
        barG = [None] * L
        barz = [np.zeros_like(z) for z in zs]

        # l = 0: g = T[0] = G[0] @ W_0ᵀ (无激活因子)
        barT[0] = adjG
        dW[0] += barT[0].T @ G[0]
        barG[0] = barT[0] @ self.weights[0]

        for l in range(1, L):
            # G[l-1] = T[l] ⊙ f'(z_{l-1})
            d_act = self.activation_deriv(zs[l - 1])
            barT[l] = barG[l - 1] * d_act
            barz[l - 1] += (barG[l - 1] * T[l]
                            * self.activation_second_deriv(zs[l - 1]))
            dW[l] += barT[l].T @ G[l]
            barG[l] = barT[l] @ self.weights[l]
        # barG[L-1] 是常数 ones 的伴随 → 丢弃

        # --- 前向图的 bar{z} 继续标准反向传播 ---
        for l in range(L - 1, -1, -1):
            dW[l] += activations[l].T @ barz[l]
            db[l] += np.sum(barz[l], axis=0, keepdims=True)
            if l > 0:
                barz[l - 1] += ((barz[l] @ self.weights[l].T)
                                * self.activation_deriv(zs[l - 1]))

        return dW, db

    def _backward(self, X: np.ndarray, y: np.ndarray,
                  activations: List[np.ndarray],
                  zs: List[np.ndarray]) -> tuple:
        m = X.shape[0]
        y = y.reshape(-1, 1)

        delta = activations[-1] - y
        dw = [activations[-2].T @ delta / m]
        db = [np.sum(delta, axis=0, keepdims=True) / m]

        for i in range(len(self.weights) - 2, -1, -1):
            delta = (delta @ self.weights[i + 1].T) * self.activation_deriv(zs[i])
            dw.insert(0, activations[i].T @ delta / m)
            db.insert(0, np.sum(delta, axis=0, keepdims=True) / m)

        return dw, db

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.001):
        """执行单步朴素梯度下降 (作用于原始标准化数据, 由训练器调用)。

        Args:
            X: 标准化输入 (n, d)。
            y: 标准化目标 (n,)。
            lr: 学习率。
        """
        activations, zs = self.forward(X)
        dw, db = self._backward(X, y, activations, zs)

        for i in range(len(self.weights)):
            self.weights[i] -= lr * dw[i]
            self.biases[i] -= lr * db[i]

    def save(self, path: str, extra: Optional[dict] = None):
        """pickle 序列化模型 (权重/偏置/归一化参数/可选附加元数据)。

        Args:
            path: 输出文件路径。
            extra: 附加写入的键值对 (如 training_card)。
        """
        import pickle
        payload = {
            "layers": self.layers,
            "activation": self.activation_name,
            "weights": self.weights,
            "biases": self.biases,
            "x_mean": self.x_mean,
            "x_scale": self.x_scale,
            "y_mean": self.y_mean,
            "y_scale": self.y_scale,
        }
        if extra:
            payload.update(extra)
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "FeedForwardNN":
        """从 pickle 文件加载模型。

        Args:
            path: 模型文件路径。

        Returns:
            FeedForwardNN: 恢复的模型实例 (兼容无归一化字段的旧文件)。
        """
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(data["layers"], data["activation"])
        model.weights = data["weights"]
        model.biases = data["biases"]
        # 兼容无归一化字段的旧模型文件
        model.x_mean = data.get("x_mean")
        model.x_scale = data.get("x_scale")
        model.y_mean = data.get("y_mean")
        model.y_scale = data.get("y_scale")
        model.training_card = data.get("training_card", {})
        return model


class PESNN:
    """势能面代理模型: 支持一维 (n,) 与二维 (n, d) 物理单位调用。

    模型携带 ``training_card`` (训练元数据: 数据指纹/RMSE/超参/
    aqpes 版本), 随 save/load 持久化 — 模型文件自解释其训练来源。
    """

    def __init__(self, model: FeedForwardNN,
                 training_card: Optional[dict] = None):
        if isinstance(model, PESNN):      # 防重复包装
            training_card = training_card or model.training_card
            model = model.model
        self.model = model
        self.training_card: dict = (
            training_card if training_card is not None
            else getattr(model, "training_card", {}) or {})

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """预测势能 V(x) (物理单位, Hartree)。"""
        return self.model.predict(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """``evaluate`` 的别名 (物理单位势能预测)。"""
        return self.evaluate(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """解析梯度 dV/dx, 形状 (n, d)。"""
        return self.model.gradient(x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x)

    def save(self, path: str):
        """保存模型 (附带 training_card 训练元数据, 随文件持久化)。"""
        self.model.save(path, extra={"training_card": self.training_card})

    @classmethod
    def load(cls, path: str) -> "PESNN":
        """加载 PESNN (自动恢复随文件持久化的 training_card)。"""
        model = FeedForwardNN.load(path)
        return cls(model, getattr(model, "training_card", {}))


def nn_pes_2d(model: PESNN):
    """把多维 PESNN 包装为二维传播子需要的 V(R, r) 闭包。

    供手工数据驱动工作流使用 (不经 engine): 训练好的代理面可直接
    喂给 ``WavePacket2DPropagator``。

    >>> pes = nn_pes_2d(model)
    >>> prop = WavePacket2DPropagator(pes, R, r, mass_R, mass_r, dt=0.5)
    """
    def V(R, r):
        """NN 代理势 V(R, r) (Hartree), 广播 R/r 并逐点预测。"""
        Rb = np.asarray(R, dtype=float)
        rb = np.asarray(r, dtype=float)
        Rb, rb = np.broadcast_arrays(Rb, rb)
        pts = np.column_stack([Rb.ravel(), rb.ravel()])
        return model.predict(pts).reshape(Rb.shape)
    return V
