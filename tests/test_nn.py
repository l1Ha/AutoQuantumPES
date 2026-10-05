import unittest
import numpy as np
from aqpes.nn.model import FeedForwardNN


class TestNNModel(unittest.TestCase):
    def test_forward_shape(self):
        model = FeedForwardNN([1, 10, 1], activation="tanh", seed=42)
        x = np.linspace(0, 1, 50)
        y = model.predict(x)
        self.assertEqual(y.shape, (50,))

    def test_training_converges(self):
        model = FeedForwardNN([1, 20, 20, 1], activation="tanh", seed=42)
        x = np.linspace(0, 1, 100).reshape(-1, 1)
        y = (x ** 2).flatten()
        for _ in range(100):
            model.train_step(x, y, lr=0.01)
        pred = model.predict(x)
        loss = np.mean((pred - y) ** 2)
        self.assertLess(loss, 0.1)

    def test_linear_output_layer(self):
        # Regression: the output layer must be linear. With a tanh (or other
        # bounded) output activation, predictions saturate within [-1, 1] and
        # cannot represent PES values outside that range. Here the network is
        # driven to a constant pre-activation of 3.0, which a squashed output
        # layer would clip to ~1.0.
        model = FeedForwardNN([1, 4, 1], activation="tanh", seed=0)
        for w in model.weights[:-1]:
            w[:] = 0.0
        for b in model.biases[:-1]:
            b[:] = 0.0
        model.weights[-1][:] = 1.0
        model.biases[-1][:] = 3.0

        x = np.linspace(-2, 2, 21)
        pred = model.predict(x)
        np.testing.assert_allclose(pred, 3.0, atol=1e-12)

    def test_fit_outside_tanh_range(self):
        # Regression: targets beyond [-1, 1] must be fittable now that the
        # output layer is linear. The prediction-mean assertion would fail
        # unconditionally with a tanh output layer (bounded by 1.0).
        model = FeedForwardNN([1, 32, 1], activation="tanh", seed=42)
        x = np.linspace(0, 1, 60).reshape(-1, 1)
        y = (3.0 * x + 1.0).flatten()  # range [1, 4]
        for _ in range(500):
            model.train_step(x, y, lr=0.01)
        pred = model.predict(x)
        self.assertGreater(np.mean(pred), 1.5)
        self.assertLess(np.mean((pred - y) ** 2), 0.05)

    def test_finite_difference_mse_gradient(self):
        # The backprop gradients returned by _backward must match the numerical
        # gradient of the mean squared error up to the factor 1/2 implied by
        # the delta = (pred - y) output convention
        # (d(MSE)/dW = 2 * delta-based dw for a linear output layer).
        rng = np.random.RandomState(7)
        model = FeedForwardNN([1, 6, 1], activation="tanh", seed=3)
        x = rng.uniform(-1, 1, size=(12, 1))
        y = rng.uniform(-1, 1, size=12)

        activations, zs = model.forward(x)
        dw, db = model._backward(x, y, activations, zs)

        def mse_loss():
            return np.mean((model.predict(x) - y) ** 2)

        eps = 1e-6
        for i in range(len(model.weights)):
            for arr, grad in ((model.weights[i], dw[i]), (model.biases[i], db[i])):
                num_grad = np.zeros_like(arr)
                it = np.nditer(arr, flags=["multi_index"])
                while not it.finished:
                    idx = it.multi_index
                    orig = arr[idx]
                    arr[idx] = orig + eps
                    loss_plus = mse_loss()
                    arr[idx] = orig - eps
                    loss_minus = mse_loss()
                    arr[idx] = orig
                    num_grad[idx] = (loss_plus - loss_minus) / (2 * eps)
                    it.iternext()
                np.testing.assert_allclose(
                    grad, 0.5 * num_grad, rtol=1e-5, atol=1e-9
                )

    def test_early_stop_restores_best_weights(self):
        # 早停后返回的模型必须对应验证损失最低的那一轮, 而非已退化的末轮
        from aqpes.nn.train import NNTrainer, TrainingConfig

        rng = np.random.RandomState(11)
        x = rng.uniform(-1, 1, 80).reshape(-1, 1)
        y_clean = np.sin(3 * x).flatten()
        y = y_clean + rng.normal(0, 0.15, x.size)  # 噪声使验证损失先降后升
        config = TrainingConfig(hidden_layers=[32, 32], epochs=400,
                                lr=0.005, early_stop_patience=20,
                                train_split=0.5, seed=1)
        trainer = NNTrainer(config)
        model, history = trainer.train(x, y)

        # 与历史最优验证损失一致性 (相对改进判据 1e-3 内的容差):
        # 恢复的权重对应最后一次"有意义改进"的轮次
        n = x.shape[0]
        idx = np.random.RandomState(config.seed).permutation(n)
        val_idx = idx[int(n * config.train_split):]
        val_loss_best = min(history["val_loss"])
        val_loss_recomputed = np.mean(
            (model.predict(x[val_idx]) - y[val_idx]) ** 2)
        self.assertLessEqual(val_loss_recomputed,
                             val_loss_best * (1 + 1e-3) + 1e-9)

    def test_multidim_fit(self):
        # 二维光滑函数拟合 (归一化开启, 输出线性)
        from aqpes.nn.train import NNTrainer, TrainingConfig

        def f(x):
            return np.sin(x[:, 0]) * np.exp(-0.5 * x[:, 1] ** 2) + 0.3 * x[:, 0]

        g1, g2 = np.meshgrid(np.linspace(-2, 2, 41), np.linspace(-2, 2, 41),
                             indexing="ij")
        X = np.column_stack([g1.ravel(), g2.ravel()])
        y = f(X)
        config = TrainingConfig(hidden_layers=[40, 40], epochs=2000, lr=0.01,
                                seed=3)
        model, history = NNTrainer(config).train(X, y)
        pred = model.predict(X)
        rmse = np.sqrt(np.mean((pred - y) ** 2))
        self.assertLess(rmse, 2e-2, f"2D 拟合 RMSE 过大: {rmse:.2e}")

    def test_gradient_matches_finite_difference(self):
        # 解析输入梯度 vs 中心差分 (含归一化链式修正路径)
        from aqpes.nn.train import NNTrainer, TrainingConfig

        rng = np.random.RandomState(5)
        X = rng.uniform(-1, 1, (60, 2))
        y = np.sin(2 * X[:, 0]) + 0.5 * X[:, 1] ** 2
        model, _ = NNTrainer(TrainingConfig(hidden_layers=[24, 24],
                                            epochs=300, lr=0.01,
                                            seed=1)).train(X, y)

        pts = rng.uniform(-0.8, 0.8, (12, 2))
        g = model.gradient(pts)
        h = 1e-6
        for j in range(2):
            ep = np.zeros(2); ep[j] = h
            num = (model.predict(pts + ep) - model.predict(pts - ep)) / (2 * h)
            np.testing.assert_allclose(g[:, j], num, rtol=1e-4, atol=1e-6)

    def test_normalization_persistence(self):
        # 归一化参数随模型保存/加载, 预测逐点一致
        import tempfile, os
        from aqpes.nn.train import NNTrainer, TrainingConfig

        X = np.linspace(-3, 3, 80).reshape(-1, 1)
        y = 5.0 * X.ravel() + 3.0
        model, _ = NNTrainer(TrainingConfig(hidden_layers=[32], epochs=600,
                                            lr=0.01, seed=0)).train(X, y)
        pred_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.pkl")
            model.save(path)
            loaded = type(model).load(path)
        pred_after = loaded.predict(X)
        np.testing.assert_allclose(pred_after, pred_before, rtol=1e-12)
        self.assertIsNotNone(loaded.model.x_mean)

        # 未归一化模型 (手工构造) 保存/加载也应工作
        from aqpes.nn.model import FeedForwardNN, PESNN
        raw = PESNN(FeedForwardNN([1, 8, 1], seed=0))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "raw.pkl")
            raw.save(path)
            raw2 = PESNN.load(path)
        self.assertIsNone(raw2.model.x_mean)

    def test_force_gradient_backward_fd(self):
        # 力训练核心: input_gradient_backward (double backprop) 的解析
        # 梯度必须与 Φ = Σ adjG·input_gradient 的有限差分一致
        rng = np.random.RandomState(9)
        model = FeedForwardNN([3, 7, 6, 1], activation="tanh", seed=2)
        X = rng.uniform(-1, 1, (8, 3))
        adjG = rng.normal(size=(8, 3))

        def phi():
            return float(np.sum(adjG * model.input_gradient(X)))

        dW, db = model.input_gradient_backward(X, adjG)
        eps = 1e-6
        for i in range(len(model.weights)):
            for arr, grad in ((model.weights[i], dW[i]),
                              (model.biases[i], db[i])):
                num = np.zeros_like(arr)
                it = np.nditer(arr, flags=["multi_index"])
                while not it.finished:
                    idx = it.multi_index
                    orig = arr[idx]
                    arr[idx] = orig + eps
                    lp = phi()
                    arr[idx] = orig - eps
                    lm = phi()
                    arr[idx] = orig
                    num[idx] = (lp - lm) / (2 * eps)
                    it.iternext()
                np.testing.assert_allclose(grad, num, rtol=1e-5, atol=1e-8)

    def test_force_training_improves_gradient(self):
        # 力训练 (fw=1) 应显著提升代理面的梯度保真度, 且不劣化能量拟合
        from aqpes.nn.train import NNTrainer, TrainingConfig

        g1, g2 = np.meshgrid(np.linspace(-2, 2, 31), np.linspace(-2, 2, 31),
                             indexing="ij")
        X = np.column_stack([g1.ravel(), g2.ravel()])
        y = np.sin(2 * X[:, 0]) + 0.3 * X[:, 1] ** 2
        dY = np.column_stack([2 * np.cos(2 * X[:, 0]),
                              0.6 * X[:, 1]])

        def fit(fw):
            m, _ = NNTrainer(TrainingConfig(hidden_layers=[32, 32],
                                            epochs=1200, lr=0.01, seed=4,
                                            force_weight=fw)).train(
                X, y, dY=dY if fw > 0 else None)
            v = np.sqrt(np.mean((m.predict(X) - y) ** 2))
            g = np.sqrt(np.mean((m.gradient(X) - dY) ** 2))
            return v, g

        v0, g0 = fit(0.0)
        v1, g1_ = fit(1.0)
        self.assertLess(g1_, 0.5 * g0,
                        f"力训练未提升梯度精度: {g1_:.3e} vs {g0:.3e}")
        self.assertLess(v1, 2.0 * v0 + 1e-3,
                        f"力训练严重劣化能量拟合: {v1:.3e} vs {v0:.3e}")


if __name__ == "__main__":
    unittest.main()
