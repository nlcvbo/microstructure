import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class DeepOF(nn.Module):
    """
    See https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3900141
    """

    def __init__(self, levels, window_size, horizons):
        super().__init__()
        # First conv processes flow imbalance across 2L channels
        self.conv1 = nn.Conv1d(in_channels=2 * levels, out_channels=32, kernel_size=1)
        # Followed by deepLOB’s CNN-LSTM stack
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.horizons = horizons
        self.output = nn.Linear(64, horizons)

    def forward(self, x):
        # x: (B, T, 2L)
        z = x.transpose(1, 2)  # to (B, 2L, T)
        z = torch.relu(self.conv1(z))
        z = torch.relu(self.conv2(z))
        z = torch.relu(self.conv3(z))
        z = self.pool(z).squeeze(-1)  # (B,64)
        # optional: use LSTM on sequence instead:
        # seq_out, (hn, _) = self.lstm(z.transpose(1,2))
        return self.output(z)  # (B, H) regression outputs


class DummyDeepOFDataset(Dataset):
    def __init__(self, N=1000, T=50, L=5, horizons=5):
        self.X = torch.randn(N, T, 2 * L)
        self.y = torch.randn(N, horizons)  # continuous returns

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_deepof(model, loader, optimizer, criterion, device, epochs=10):
    model.to(device)
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)  # (B, H)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {ep + 1}, loss = {total_loss / len(loader):.6f}")


if __name__ == "__main__":
    dummy_ds = DummyDeepOFDataset()
    loader = DataLoader(dummy_ds, batch_size=32, shuffle=True)
    model = DeepOF(levels=5, window_size=50, horizons=len([1, 2, 3, 5]))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_deepof(model, loader, optimizer, criterion, device, epochs=5)

    # Real data:
    # df = pl.from_arrow(table)
    # real_ds = OFIDataset(df, window_size=50, horizons=[1,2,3,5], levels=5)
    # real_loader = DataLoader(real_ds, batch_size=64, shuffle=True)
    # train_deepof(model, real_loader, optimizer, criterion, device, epochs=10)
