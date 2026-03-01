"""LSTM-based workload predictor for carbon-aware scheduling."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class WorkloadLSTM(nn.Module):
    def __init__(self, input_size: int = 3, hidden_size: int = 64, num_layers: int = 2, output_size: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


class WorkloadPredictor:
    """
    Predicts future GPU cluster utilization from historical patterns.
    Input features: [time_of_day_sin, time_of_day_cos, current_utilization]
    Target: utilization at next timestep
    """

    SEQ_LEN = 12  # 12 steps of history

    def __init__(self, seed: int = 42):
        torch.manual_seed(seed)
        self.model = WorkloadLSTM()
        self.is_trained = False
        self.scaler_mean = 0.5
        self.scaler_std = 0.25

    def _time_features(self, time_minutes: float) -> tuple:
        t = (time_minutes % 1440) / 1440.0 * 2 * np.pi
        return np.sin(t), np.cos(t)

    def _build_dataset(self, times: np.ndarray, utilizations: np.ndarray):
        """Build sequence dataset from time series."""
        X, y = [], []
        for i in range(self.SEQ_LEN, len(times)):
            seq = []
            for j in range(i - self.SEQ_LEN, i):
                sin_t, cos_t = self._time_features(times[j])
                u = (utilizations[j] - self.scaler_mean) / self.scaler_std
                seq.append([sin_t, cos_t, u])
            X.append(seq)
            y.append([(utilizations[i] - self.scaler_mean) / self.scaler_std])
        return torch.FloatTensor(X), torch.FloatTensor(y)

    def train(self, times: np.ndarray, utilizations: np.ndarray, epochs: int = 30):
        """Train on historical utilization data."""
        X, y = self._build_dataset(times, utilizations)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        self.model.train()
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(loader))
        self.is_trained = True
        return losses

    def predict(self, times: np.ndarray, utilizations: np.ndarray) -> float:
        """Predict next utilization value given recent history."""
        if not self.is_trained or len(times) < self.SEQ_LEN:
            return float(np.mean(utilizations[-self.SEQ_LEN:]) if len(utilizations) >= 1 else 0.5)
        self.model.eval()
        with torch.no_grad():
            seq = []
            for j in range(max(0, len(times) - self.SEQ_LEN), len(times)):
                sin_t, cos_t = self._time_features(times[j])
                u = (utilizations[j] - self.scaler_mean) / self.scaler_std
                seq.append([sin_t, cos_t, u])
            # pad if needed
            while len(seq) < self.SEQ_LEN:
                seq.insert(0, seq[0] if seq else [0, 1, 0])
            x = torch.FloatTensor([seq])
            pred_norm = self.model(x).item()
            pred = pred_norm * self.scaler_std + self.scaler_mean
            return float(np.clip(pred, 0.0, 1.0))

    def generate_training_data(self, seed: int = 42) -> tuple:
        """Synthesize realistic historical utilization data for training."""
        rng = np.random.default_rng(seed)
        times = np.arange(0, 1440, 5, dtype=float)  # 5-min resolution over 24h
        utilizations = []
        for t in times:
            hour = (t / 60) % 24
            # Business-hours load pattern
            base = 0.3 + 0.45 * np.exp(-0.5 * ((hour - 14) / 4) ** 2)
            base += 0.15 * np.exp(-0.5 * ((hour - 22) / 2) ** 2)  # night batch jobs
            noise = rng.normal(0, 0.05)
            utilizations.append(float(np.clip(base + noise, 0.05, 0.98)))
        return times, np.array(utilizations)
