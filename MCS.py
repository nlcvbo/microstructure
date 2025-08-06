# mcs_eval.py

import numpy as np
import torch
from arch.bootstrap import ModelConfidenceSet
from torch.utils.data import DataLoader

# === Import models from your other files ===
from deepLOB import DeepLOB
from deepOF import DeepOF
from deepVOL import DeepVOL


# === Loss Evaluation Function ===
def evaluate_model(model, test_loader, device, criterion):
    model.eval()
    all_losses = []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            batch_losses = []
            for k in range(logits.size(1)):  # multi-horizon
                loss = criterion(logits[:, k, :], y[:, k])
                batch_losses.append(loss.cpu().item())
            all_losses.append(batch_losses)
    return np.array(all_losses).mean(axis=0)  # mean loss per horizon


# === Run MCS for multiple models ===
def run_mcs(models, test_loader, criterion, device, alpha=0.1, reps=1000):
    losses = []
    for model in models:
        model.to(device)
        losses.append(evaluate_model(model, test_loader, device, criterion))
    losses = np.array(losses)  # shape: (n_models, n_horizons)
    print("Loss matrix (rows=models, cols=horizons):\n", losses)

    # Run MCS on each horizon separately
    results = {}
    for h, horizon_losses in enumerate(losses.T):
        mcs = ModelConfidenceSet(horizon_losses[:, None], size=1 - alpha, reps=reps)
        included = mcs.compute()
        results[f"horizon_{h + 1}"] = included
    return results


# === Dummy test loader (for standalone testability) ===
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, N=200, T=50, levels=5, horizons=[1, 2, 3]):
        self.X = torch.randn(N, T, levels, 2)
        self.y = torch.randint(0, 3, (N, len(horizons)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dummy_test_loader(batch_size=32):
    dummy = DummyDataset()
    return DataLoader(dummy, batch_size=batch_size, shuffle=False)


# === CLI or testing entry point ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = get_dummy_test_loader()
    criterion = torch.nn.CrossEntropyLoss()
    horizons = [1, 2, 3]

    # Create and load untrained models for demo (or load trained weights)
    models = [
        DeepLOB(levels=5, time_steps=50, horizons=horizons),
        DeepVOL(levels=5, time_steps=50, horizons=horizons),
        DeepOF(levels=5, time_steps=50, horizons=horizons),
    ]

    print("Running MCS evaluation on dummy models...")
    mcs_results = run_mcs(models, test_loader, criterion, device)
    for horizon, included in mcs_results.items():
        print(f"{horizon}: models in 90% MCS → {included}")

    # TODO: with df
