import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- Parameters ---
WINDOW = 50       # T
HORIZONS = [1, 2, 3, 5]  # future step offsets for labels
LEVELS = 5        # LOB depth per side
EPSILON = 1e-4    # threshold for price movement discretization

# --- 1. Extract volume and price arrays from Polars df ---
def polars_to_numpy(df: pl.DataFrame):
    df = df.sort("TIMESTAMP")
    ask_sizes = df.select([f"ASK_SIZE{i}" for i in range(1, LEVELS+1)]).to_numpy()
    bid_sizes = df.select([f"BID_SIZE{i}" for i in range(1, LEVELS+1)]).to_numpy()
    ask_prices = df.select([f"ASK_PRICE{i}" for i in range(1, LEVELS+1)]).to_numpy()
    bid_prices = df.select([f"BID_PRICE{i}" for i in range(1, LEVELS+1)]).to_numpy()
    timestamps = df["TIMESTAMP"].to_numpy()
    return ask_sizes, bid_sizes, ask_prices, bid_prices, timestamps

def compute_mid_prices(ask_prices, bid_prices):
    return (ask_prices[:, 0] + bid_prices[:, 0]) / 2

def discretize_return(r):
    if r > EPSILON:
        return 2  # up
    elif r < -EPSILON:
        return 0  # down
    else:
        return 1  # no-change

class VOLWindowDataset(Dataset):
    def __init__(self, df: pl.DataFrame, window_size=WINDOW, horizons=HORIZONS):
        self.ask_sizes, self.bid_sizes, self.ask_prices, self.bid_prices, self.timestamps = polars_to_numpy(df)
        self.mid_prices = compute_mid_prices(self.ask_prices, self.bid_prices)
        self.window_size = window_size
        self.horizons = horizons
        self.X, self.y = self.create_windows()

    def create_windows(self):
        X_list, y_list = [], []
        N = len(self.mid_prices)
        for t in range(self.window_size, N - max(self.horizons)):
            # --- Build volume representation ---
            ask_vol = self.ask_sizes[t - self.window_size:t]  # (T, 5)
            bid_vol = self.bid_sizes[t - self.window_size:t]  # (T, 5)
            vol_tensor = np.stack([bid_vol, ask_vol], axis=-1)  # (T, 5, 2)

            # --- Build multi-horizon labels ---
            y_t = []
            p_now = self.mid_prices[t-1]
            for h in self.horizons:
                p_future = self.mid_prices[t + h - 1]
                ret = (p_future - p_now) / p_now
                y_t.append(discretize_return(ret))
            X_list.append(vol_tensor)
            y_list.append(y_t)
        return torch.tensor(np.array(X_list), dtype=torch.float32), torch.tensor(np.array(y_list), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def make_dataloader_VOL(df: pl.DataFrame, batch_size=32, shuffle=True):
    dataset = VOLWindowDataset(df)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def compute_ofi(curr, prev):
    # curr, prev arrays: (levels,)
    # apply level-wise Cont et al. rules for each side
    def flow(curr_p, curr_q, prev_p, prev_q):
        flows = np.zeros_like(curr_q)
        for i in range(len(curr_q)):
            if curr_p[i] > prev_p[i]:
                flows[i] = curr_q[i]
            elif curr_p[i] == prev_p[i]:
                flows[i] = curr_q[i] - prev_q[i]
            else:
                flows[i] = -prev_q[i]
        return flows
    ask = flow(curr['ask_p'], curr['ask_q'], prev['ask_p'], prev['ask_q'])
    bid = flow(curr['bid_p'], curr['bid_q'], prev['bid_p'], prev['bid_q'])
    return ask, bid

class OFIDataset(Dataset):
    def __init__(self, df: pl.DataFrame, window_size=50, horizons=[1,2,3,5], levels=5):
        df = df.sort("TIMESTAMP")
        npdf = df.to_pandas()
        N = len(npdf)
        ask_p = npdf[[f"ASK_PRICE{i}" for i in range(1,levels+1)]].values
        ask_q = npdf[[f"ASK_SIZE{i}"  for i in range(1,levels+1)]].values
        bid_p = npdf[[f"BID_PRICE{i}" for i in range(1,levels+1)]].values
        bid_q = npdf[[f"BID_SIZE{i}"  for i in range(1,levels+1)]].values
        mid = (ask_p[:,0] + bid_p[:,0]) / 2.0
        X, y = [], []
        for t in range(1, N):
            ask_flow, bid_flow = compute_ofi(
                dict(ask_p=ask_p[t], ask_q=ask_q[t],
                     bid_p=bid_p[t], bid_q=bid_q[t]),
                dict(ask_p=ask_p[t-1], ask_q=ask_q[t-1],
                     bid_p=bid_p[t-1], bid_q=bid_q[t-1])
            )
            ofi_vec = np.concatenate([ask_flow, bid_flow], axis=0)
            X.append(ofi_vec)
        X = np.array(X)  # shape (N−1, 2L)
        for t in range(window_size, N-1-max(horizons)):
            Xw = X[t-window_size:t]
            returns = []
            p0 = mid[t]
            for h in horizons:
                ph = mid[t+h]
                returns.append((ph - p0)/p0)
            y.append(returns)
        self.X = torch.tensor(np.stack([X[i-window_size:i] for i in range(window_size, len(X)-max(horizons))], dtype=np.float32))
        self.y = torch.tensor(np.array(y, dtype=np.float32))
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def make_dataloader_OF(df: pl.DataFrame, batch_size=32, shuffle=True):
    dataset = OFIDataset(df, window_size=WINDOW, horizons=HORIZONS, levels=LEVELS)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def discretize(r, eps=1e-4):
    return np.where(r > eps, 2, np.where(r < -eps, 0, 1))

class DeepLOBDataset(Dataset):
    def __init__(self, df: pl.DataFrame, window_size=50, horizons=[1,2,3], levels=5):
        df = df.sort("TIMESTAMP")
        ask = df.select([f"ASK_SIZE{i}" for i in range(1,levels+1)]).to_numpy()
        bid = df.select([f"BID_SIZE{i}" for i in range(1,levels+1)]).to_numpy()
        ask_p = df.select([f"ASK_PRICE{i}" for i in range(1,levels+1)]).to_numpy()
        bid_p = df.select([f"BID_PRICE{i}" for i in range(1,levels+1)]).to_numpy()
        mid = (ask_p[:,0] + bid_p[:,0]) / 2.0

        X, Y = [], []
        N = len(mid)
        for t in range(window_size, N - max(horizons)):
            vol_w = np.stack([
                bid[t-window_size:t],
                ask[t-window_size:t]
            ], axis=-1)  # (T, levels, 2)
            ret0 = mid[t-1]
            labels = []
            for h in horizons:
                labels.append(discretize((mid[t+h] - ret0) / ret0))
            X.append(vol_w.astype(np.float32))
            Y.append(labels)
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(np.array(Y), dtype=torch.long)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def make_dataloader_LOB(df: pl.DataFrame, batch_size=32, shuffle=True):
    dataset = DeepLOBDataset(df, window_size=WINDOW, horizons=HORIZONS, levels=LEVELS)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

