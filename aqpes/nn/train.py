import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field

from aqpes.nn.model import FeedForwardNN, PESNN


@dataclass
class TrainingConfig:
    """NN 训练超参数配置 (数据类)。

    Attributes:
        hidden_layers: 隐藏层宽度序列。
        activation: 隐藏层激活函数名 (tanh/sigmoid/relu/linear)。
        epochs: 最大训练轮数。
        lr: 学习率。
        train_split: 训练/验证划分比例。
        batch_size: 批大小; 0 = 全批量。
        seed: 随机种子 (初始化与数据划分)。
        early_stop_patience: 早停容忍轮数。
        normalize: 是否在训练划分上拟合输入/输出标准化。
        max_train_points: >0 时等距子采样到该点数。
        force_weight: 力训练损失权重; 0 = 纯能量拟合。
        adam_beta1/beta2/eps: Adam 参数 (beta1=0 退回朴素梯度下降)。
    """
    hidden_layers: List[int] = field(default_factory=lambda: [64, 64, 32])
    activation: str = "tanh"
    epochs: int = 500
    lr: float = 0.001
    train_split: float = 0.8
    batch_size: int = 0
    seed: int = 42
    # 无相对改进 (阈值 1e-3) 的容忍轮数; 全批量 Adam 尾部收敛慢, 需留足
    early_stop_patience: int = 100
    # 输入/输出标准化 (在训练划分上拟合, 随模型持久化)
    normalize: bool = True
    # >0 时等距子采样到该点数 (大数据网格加速训练)
    max_train_points: int = 0
    # 力训练: dY (n, d) 为 ∂V/∂x 监督目标, force_weight 为其损失权重;
    # 0 = 纯能量拟合
    force_weight: float = 0.0
    # Adam 优化器 (beta1=0 时退回朴素梯度下降)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8


class NNTrainer:
    """多维 PES 回归训练器。

    - 输入 X: (n, d) 或 (n,) [d=1]; 目标 y: (n,)
    - 自动标准化 (x/y 各自 train 划分统计), 随模型保存/加载
    - 早停并恢复验证损失最低的权重
    - history 损失为物理单位 MSE
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

    def train(self, X: np.ndarray, y: np.ndarray,
              dY: np.ndarray = None) -> Tuple[PESNN, Dict[str, List[float]]]:
        """训练 PES 回归模型 (标准化 + Adam + 早停 + 可选力训练)。

        Args:
            X: 输入 (n, d) 或 (n,) [d=1], 物理单位。
            y: 能量目标 (n,), 物理单位 (Hartree)。
            dY: 可选梯度目标 ∂V/∂x (n, d); ``force_weight > 0`` 时启用。

        Returns:
            (PESNN, history): 携带 training_card 的代理模型与物理单位的
            train/val 损失历史; 返回的权重为验证损失最低的早停快照。
        """
        cfg = self.config
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape
        if y.size != n:
            raise ValueError(f"X/y 样本数不一致: {n} vs {y.size}")

        force = cfg.force_weight > 0 and dY is not None
        if dY is not None:
            dY = np.asarray(dY, dtype=float)
            if dY.shape != X.shape:
                raise ValueError(f"dY 形状 {dY.shape} 应与 X {X.shape} 相同")
            if cfg.force_weight <= 0:
                print("  [warn] 提供了 dY 但 force_weight=0, 忽略力目标")

        # 等距子采样 (保持网格代表性)
        if cfg.max_train_points and n > cfg.max_train_points:
            idx = np.linspace(0, n - 1, cfg.max_train_points).astype(int)
            X, y = X[idx], y[idx]
            if dY is not None:
                dY = dY[idx]
            n = cfg.max_train_points

        n_train = int(n * cfg.train_split)
        indices = np.random.RandomState(cfg.seed).permutation(n)
        train_idx, val_idx = indices[:n_train], indices[n_train:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        dY_train = dY[train_idx] if dY is not None else None
        dY_val = dY[val_idx] if dY is not None else None

        model = FeedForwardNN(
            layers=[d] + list(cfg.hidden_layers) + [1],
            activation=cfg.activation,
            seed=cfg.seed,
        )

        if cfg.normalize:
            x_mean = X_train.mean(axis=0)
            x_scale = X_train.std(axis=0)
            # 近常数维度不参与缩放 (否则标准化后数值爆炸)
            x_scale = np.where(x_scale < 1e-12, 1.0, x_scale)
            y_mean = float(y_train.mean())
            y_scale = float(y_train.std()) or 1.0
            model.set_normalization(x_mean, x_scale, y_mean, y_scale)
            Xn_train = (X_train - x_mean) / x_scale
            Xn_val = (X_val - x_mean) / x_scale
            yn_train = (y_train - y_mean) / y_scale
            # 力目标标准化: dV_std/dx_std = dV/dx · x_scale / y_scale
            t_std = dY_train * (x_scale / y_scale) if force else None
        else:
            Xn_train, Xn_val, yn_train = X_train, X_val, y_train
            t_std = dY_train if force else None

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        # 联合损失 (物理单位): MSE_V + force_weight · MSE_F
        def combined_loss(Xp, yp, dYp):
            """物理单位联合损失: MSE(V) + force_weight·MSE(梯度)。"""
            mse_v = np.mean((model.predict(Xp) - yp) ** 2)
            if not force or dYp is None:
                return mse_v
            g_phys = model.gradient(Xp)
            return mse_v + cfg.force_weight * np.mean((g_phys - dYp) ** 2)

        # Adam 状态 (作用于标准化数据)
        use_adam = self.config.adam_beta1 > 0
        if force and not use_adam:
            print("  [warn] adam_beta1=0 (朴素梯度下降) 不支持力训练, 力目标被忽略")
        from aqpes.nn.optim import Adam
        opt = Adam([w.shape for w in model.weights] + [b.shape for b in model.biases],
                   lr=cfg.lr, beta1=cfg.adam_beta1, beta2=cfg.adam_beta2,
                   eps=cfg.adam_eps)

        def adam_step(Xb, yb, tb=None):
            """一步 Adam 更新 (能量损失 + 可选力目标伴随的合并梯度)。"""
            activations, zs = model.forward(Xb)
            dw, db = model._backward(Xb, yb, activations, zs)
            if force and tb is not None:
                # 力损失 L_F = mean((g_std - t_std)²), 伴随 = 2(g-t)/(n·d)
                g_std = model.input_gradient(Xb)
                adj = 2.0 * (g_std - tb) / Xb.size * cfg.force_weight
                dwf, dbf = model.input_gradient_backward(Xb, adj)
                dw = [a + b_ for a, b_ in zip(dw, dwf)]
                db = [a + b_ for a, b_ in zip(db, dbf)]
            opt.step(model.weights + model.biases, dw + db)

        for epoch in range(cfg.epochs):
            if cfg.batch_size > 0:
                bs = cfg.batch_size
                for i in range(0, len(Xn_train), bs):
                    sl = slice(i, i + bs)
                    tb = t_std[sl] if force else None
                    if use_adam:
                        adam_step(Xn_train[sl], yn_train[sl], tb)
                    else:
                        model.train_step(Xn_train[sl], yn_train[sl], cfg.lr)
            else:
                if use_adam:
                    adam_step(Xn_train, yn_train, t_std if force else None)
                else:
                    model.train_step(Xn_train, yn_train, cfg.lr)

            # 物理单位联合损失
            train_loss = combined_loss(X_train, y_train, dY_train)
            val_loss = combined_loss(X_val, y_val, dY_val)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if epoch % 100 == 0:
                print(f"  Epoch {epoch:4d}/{cfg.epochs} | "
                      f"train_loss: {train_loss:.6e} | val_loss: {val_loss:.6e}")

            if val_loss < best_val_loss * (1.0 - 1e-3):
                best_val_loss = val_loss
                best_state = ([w.copy() for w in model.weights],
                              [b.copy() for b in model.biases])
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stop_patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        # 恢复验证损失最低时的权重 (早停/过拟合时避免返回已退化的末轮模型)
        if best_state is not None:
            model.weights, model.biases = best_state

        pes_nn = PESNN(model)
        # 训练卡: 模型文件自解释其训练来源 (科学计算软件硬要求)
        from aqpes.core.validation import data_fingerprint, environment_info
        import datetime
        full_pred = pes_nn.predict(X)
        pes_nn.training_card = {
            "created_utc": datetime.datetime.now(
                datetime.timezone.utc).isoformat(timespec="seconds"),
            "n_points": int(n),
            "input_dim": int(d),
            "data_sha256": data_fingerprint(
                X, y, dY if dY is not None else np.zeros(0)),
            "force_used": bool(dY is not None and cfg.force_weight > 0),
            "epochs_run": len(self.history["train_loss"]),
            "rmse_full": float(np.sqrt(np.mean((full_pred - y) ** 2))),
            "val_loss_best": float(best_val_loss),
            "hyperparameters": {
                "hidden_layers": list(cfg.hidden_layers),
                "activation": cfg.activation,
                "lr": cfg.lr,
                "train_split": cfg.train_split,
                "batch_size": cfg.batch_size,
                "seed": cfg.seed,
                "force_weight": cfg.force_weight,
                "normalize": cfg.normalize,
            },
            "environment": environment_info(),
        }
        return pes_nn, self.history
