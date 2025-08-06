import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# --- 1. DeepVOL model from https://arxiv.org/abs/2211.13777


class DeepVOL(nn.Module):
    """
    See https://arxiv.org/abs/2211.13777
    """

    def __init__(self, levels, multi_horizons=None):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=(levels, 1), stride=(levels, 1))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(1, 2), stride=1)
        self.temp1 = nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(3, 0), groups=32)
        self.temp2 = nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(3, 0), groups=32)
        self.spatial = nn.Conv2d(32, 32, kernel_size=(1, 1))
        self.temp3 = nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(3, 0), groups=32)
        self.temp4 = nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(3, 0), groups=32)
        self.inc1 = nn.Conv1d(32, 64, kernel_size=1)
        self.inc3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.inc5 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.incmax = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.inc_lin = nn.Conv1d(64 * 4, 64, kernel_size=1)
        self.encoder = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.multi_horizons = multi_horizons
        if multi_horizons:
            self.decoder = nn.LSTM(input_size=3, hidden_size=64, batch_first=True)
            self.out = nn.Linear(64, 3)
        else:
            self.fc = nn.Linear(64, 3)

    def forward(self, x):
        B, T, W, _ = x.shape
        v = x.permute(0, 3, 2, 1)  # (B,2,W,T)
        z = torch.relu(self.conv1(v))
        z = torch.relu(self.conv2(z))
        z = torch.relu(self.temp1(z))
        z = torch.relu(self.temp2(z))
        z = torch.relu(self.spatial(z))
        z = torch.relu(self.temp3(z))
        z = torch.relu(self.temp4(z))
        z = z.squeeze(2)  # shape (B,32,T)
        i1 = self.inc1(z)
        i3 = self.inc3(z)
        i5 = self.inc5(z)
        im = self.inc_lin(self.incmax(z))
        zi = torch.cat([i1, i3, i5, im], dim=1)  # (B,256,T)
        h = self.inc_lin(zi).transpose(1, 2)  # (B,T,64)
        enc_out, (h_n, c_n) = self.encoder(h)
        h_last = h_n[-1]  # (B,64)
        if self.multi_horizons:
            inputs = torch.zeros(B, self.multi_horizons, 3, device=x.device)
            dec_out, _ = self.decoder(inputs, (h_n, c_n))
            logits = self.out(dec_out)  # (B, K, 3)
            return logits
        else:
            return torch.softmax(self.fc(h_last), dim=-1)


# --- 2. Dummy Dataset Class (replace with real loader) ---


class DummyLOBSet(Dataset):
    def __init__(self, N=1000, T=50, W=10, H=5):
        self.X = torch.randn(N, T, W, 2)
        self.y = torch.randint(0, 3, (N, H))  # 3-class mid-price move

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --- 3. Training Function ---


def train_deepvol(model, dataloader, optimizer, criterion, device, epochs=10):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)  # (B, K, 3)
            B, K, C = out.shape
            loss = 0
            for k in range(K):
                loss += criterion(out[:, k, :], y[:, k])
                preds = out[:, k, :].argmax(dim=1)
                correct += (preds == y[:, k]).sum().item()
                total += len(y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        acc = correct / total
        print(f"Epoch {epoch + 1}: Loss={total_loss:.4f}, Accuracy={acc:.4f}")


# --- 4. Run Training Loop ---

if __name__ == "__main__":
    T, W, H = 50, 10, 5  # time steps, levels, horizons

    # Dummy dataset
    dataset = DummyLOBSet(N=1000, T=T, W=W, H=H)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = DeepVOL(levels=W, multi_horizons=H)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_deepvol(model, loader, optimizer, criterion, device, epochs=10)

    # Real dataset
    # df = pl.from_arrow(table)
    # dataloader = make_dataloader(df)
    # HORIZONS = dataloader.horizons
    # model = DeepVOL(levels=5, multi_horizons=len(HORIZONS))
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # criterion = torch.nn.CrossEntropyLoss()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train_deepvol(model, dataloader, optimizer, criterion, device, epochs=10)
