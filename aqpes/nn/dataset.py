import numpy as np


class PESDataset:
    def __init__(self, points: np.ndarray, energies: np.ndarray):
        self.points = np.asarray(points, dtype=np.float32)
        self.energies = np.asarray(energies, dtype=np.float32)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx], self.energies[idx]
