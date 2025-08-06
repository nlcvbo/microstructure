import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class DeepLOB(nn.Module):
    """
    See https://arxiv.org/pdf/1808.03668
    """

    def __init__(self, levels=5, time_steps=50, horizons=[1]):
        super().__init__()
        # Spatial convolution: price-level × time
        self.conv1 = nn.Conv2d(2, 32, kernel_size=(levels, 1))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(1, 2), padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0))
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0))
        self.conv5 = nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0))
        self.dropout = nn.Dropout(0.2)
        self.time_steps = time_steps
        self.horizons = horizons
        self.lstm = nn.LSTM(
            input_size=64 * (time_steps), hidden_size=64, batch_first=True
        )
        self.fc = nn.Linear(64, 3 * len(horizons))

    def forward(self, x):
        # x: (B, time_steps, levels, 2)
        z = x.permute(0, 3, 2, 1)  # → (B,2,levels,time)
        z = torch.relu(self.conv1(z))
        z = torch.relu(self.conv2(z))
        z = torch.relu(self.conv3(z))
        z = self.pool(z)  # halve temporal dimension
        z = torch.relu(self.conv4(z))
        z = torch.relu(self.conv5(z))
        B, C, L, T2 = z.size()
        z = z.view(B, C * L, T2).transpose(1, 2)  # → (B, T2, features)
        z = self.dropout(z)
        out, (hn, _) = self.lstm(z)
        h = hn[-1]  # (B,64)
        logits = self.fc(h)
        logits = logits.view(B, len(self.horizons), 3)
        return torch.softmax(logits, dim=-1)


class DummyDeepLOB(Dataset):
    def __init__(self, N=1000, T=50, levels=5, horizons=[1, 2, 3]):
        self.X = torch.randn(N, T, levels, 2)
        self.y = torch.randint(0, 3, (N, len(horizons)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_deeplob(model, loader, optimizer, criterion, device, epochs=10):
    model.to(device)
    for ep in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        model.train()
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)  # (B, K, 3)
            loss = 0
            for k in range(logits.size(1)):
                loss += criterion(logits[:, k, :], y[:, k])
                preds = logits[:, k, :].argmax(dim=1)
                correct += (preds == y[:, k]).sum().item()
                total += y.size(0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            f"Epoch {ep + 1}: Loss={total_loss / len(loader):.4f}, "
            f"Acc={correct / total:.4f}"
        )


if __name__ == "__main__":
    # Dummy test
    dummy = DummyDeepLOB()
    loader = DataLoader(dummy, batch_size=32, shuffle=True)
    model = DeepLOB(levels=5, time_steps=50, horizons=[1, 2, 3])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_deeplob(model, loader, optimizer, criterion, device, epochs=5)

    # Real data
    # df = pl.from_arrow(table)
    # real_ds = DeepLOBDataset(df, window_size=50, horizons=[1,2,3], levels=5)
    # real_loader = DataLoader(real_ds, batch_size=64, shuffle=True)
    # train_deeplob(model, real_loader, optimizer, criterion, device, epochs=10)
