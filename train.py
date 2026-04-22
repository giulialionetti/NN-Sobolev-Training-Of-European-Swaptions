import argparse
import os
from typing import TypedDict
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/swaption_data.csv")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--patience", type=int, default=30)
# Batch size lowered to 512 to prevent OOM errors, as second-order 
# graphs consume significantly more VRAM.
parser.add_argument("--batch", type=int, default=512)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--hidden",type=int, default=512)
parser.add_argument("--layers", type=int, default=5)
parser.add_argument("--lambda_v",type=float, default=2.0)
parser.add_argument("--lambda_d",type=float, default=2.0)
parser.add_argument("--lambda_volga", type=float, default=1.0)
parser.add_argument("--lambda_gamma", type=float, default=1.0)
parser.add_argument("--clip_grad", type=float, default=1.0)
parser.add_argument("--log_every", type=int, default=10)
parser.add_argument("--out",default="swaption_model.pt")
parser.add_argument("--plot",default="sobolev_loss.png")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

df = pd.read_csv(args.data)
df = df.dropna(subset=["price", "vega", "delta", "volga", "gamma"]).reset_index(drop=True)
print(f"Loaded {len(df)} samples from {args.data}")

INPUT_COLS = ["a", "sigma", "r0", "T", "swap_length", "K"]
SIGMA_IDX  = INPUT_COLS.index("sigma")
R0_IDX     = INPUT_COLS.index("r0")

X_raw  = torch.tensor(df[INPUT_COLS].values, dtype=torch.float32)
price  = torch.tensor(df["price"].values,    dtype=torch.float32).unsqueeze(1)
vega   = torch.tensor(df["vega"].values,     dtype=torch.float32).unsqueeze(1)
delta  = torch.tensor(df["delta"].values,    dtype=torch.float32).unsqueeze(1)
volga  = torch.tensor(df["volga"].values,    dtype=torch.float32).unsqueeze(1)
gamma  = torch.tensor(df["gamma"].values,    dtype=torch.float32).unsqueeze(1)

X_mean = X_raw.mean(0)
X_std  = X_raw.std(0).clamp(min=1e-8)
X_norm = (X_raw - X_mean) / X_std

N      = len(df)
perm   = torch.randperm(N)
tr_idx = perm[:int(0.9 * N)]
va_idx = perm[int(0.9 * N):]

vega_std   = vega[tr_idx].std().clamp(min=1e-8)
delta_std  = delta[tr_idx].std().clamp(min=1e-8)
volga_std  = volga[tr_idx].std().clamp(min=1e-8)
gamma_std  = gamma[tr_idx].std().clamp(min=1e-8)
price_mean = price[tr_idx].mean().clamp(min=1e-6)

print(f"\nTraining set statistics:")
print(f"  price_mean = {price_mean.item():.6f}")
print(f"  vega_std   = {vega_std.item():.6f}")
print(f"  delta_std  = {delta_std.item():.6f}")
print(f"  volga_std  = {volga_std.item():.6f}")
print(f"  gamma_std  = {gamma_std.item():.6f}")

X_norm     = X_norm.to(device)
X_std      = X_std.to(device)
price      = price.to(device)
vega       = vega.to(device)
delta      = delta.to(device)
volga      = volga.to(device)
gamma      = gamma.to(device)
vega_std   = vega_std.to(device)
delta_std  = delta_std.to(device)
volga_std  = volga_std.to(device)
gamma_std  = gamma_std.to(device)
price_mean = price_mean.to(device)

class MLP(nn.Module):
    def __init__(self, n_in: int = 6, n_hidden: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(n_in, n_hidden), nn.Softplus()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Softplus()]
        layers.append(nn.Linear(n_hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

_base_model = MLP(n_hidden=args.hidden, n_layers=args.layers).to(device)

n_gpus = torch.cuda.device_count()
if n_gpus > 1:
    model = nn.DataParallel(_base_model)
    effective_batch = args.batch * n_gpus
else:
    model = _base_model
    effective_batch = args.batch

optimizer = torch.optim.Adam(_base_model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

class LossDict(TypedDict):
    total:     torch.Tensor
    price:     torch.Tensor
    sobolev_1: torch.Tensor
    sobolev_2: torch.Tensor

class TrainingHistory(TypedDict):
    epoch:           list[int]
    train_total:     list[float]
    train_price:     list[float]
    train_sobolev_1: list[float]
    train_sobolev_2: list[float]
    val_rmse:        list[float]

def sobolev_loss(
    x_n:     torch.Tensor,
    y_price: torch.Tensor,
    y_vega:  torch.Tensor,
    y_delta: torch.Tensor,
    y_volga: torch.Tensor,
    y_gamma: torch.Tensor,
) -> LossDict:
    x_n    = x_n.detach().requires_grad_(True)
    y_pred = _base_model(x_n.to(device))

    
    price_loss = (y_pred - y_price).abs().mean()

    
    lambdas = torch.zeros(len(INPUT_COLS), device=x_n.device)
    lambdas[SIGMA_IDX] = args.lambda_v
    lambdas[R0_IDX]    = args.lambda_d

    signs = 2 * torch.randint(0, 2, (x_n.shape[0], len(INPUT_COLS)), device=x_n.device).float() - 1
    u = signs * lambdas.sqrt()

    # create_graph=True is critical here so we can differentiate through it again
    grad = torch.autograd.grad(y_pred.sum(), x_n, create_graph=True)[0]

    grad_vega_scale  = 1.0 / (X_std[SIGMA_IDX] * vega_std)
    grad_delta_scale = 1.0 / (X_std[R0_IDX] * delta_std)

    net_dd = (grad[:, SIGMA_IDX:SIGMA_IDX+1] * grad_vega_scale  * u[:, SIGMA_IDX:SIGMA_IDX+1]
            + grad[:, R0_IDX:R0_IDX+1]       * grad_delta_scale * u[:, R0_IDX:R0_IDX+1])

    label_dd = (u[:, SIGMA_IDX:SIGMA_IDX+1] * (y_vega / vega_std)
              + u[:, R0_IDX:R0_IDX+1]       * (y_delta / delta_std))

    sobolev_1_val = ((net_dd - label_dd) ** 2).mean()

    
    grad_sigma = grad[:, SIGMA_IDX]
    grad_r0    = grad[:, R0_IDX]

    # Differentiate the gradients w.r.t inputs again to get the Hessian diagonals.
    # We must keep create_graph=True so PyTorch can backpropagate the final loss to model weights.
    grad2_sigma = torch.autograd.grad(grad_sigma.sum(), x_n, create_graph=True)[0]
    grad2_r0    = torch.autograd.grad(grad_r0.sum(), x_n, create_graph=True)[0]

    # Convert the normalized second derivative to raw greek space: divide by std^2
    net_volga_raw = grad2_sigma[:, SIGMA_IDX:SIGMA_IDX+1] / (X_std[SIGMA_IDX] ** 2)
    net_gamma_raw = grad2_r0[:, R0_IDX:R0_IDX+1]          / (X_std[R0_IDX] ** 2)

    # Normalize by the variance over the training set and compute MSE
    volga_loss = args.lambda_volga * (((net_volga_raw - y_volga) / volga_std) ** 2).mean()
    gamma_loss = args.lambda_gamma * (((net_gamma_raw - y_gamma) / gamma_std) ** 2).mean()

    sobolev_2_val = volga_loss + gamma_loss

    return {
        "total":     price_loss + sobolev_1_val + sobolev_2_val,
        "price":     price_loss,
        "sobolev_1": sobolev_1_val,
        "sobolev_2": sobolev_2_val,
    }

best_val_rmse  = float("inf")
best_state     = None
epochs_no_impr = 0

history: TrainingHistory = {
    "epoch":           [],
    "train_total":     [],
    "train_price":     [],
    "train_sobolev_1": [],
    "train_sobolev_2": [],
    "val_rmse":        [],
}

for epoch in range(1, args.epochs + 1):
    model.train()
    perm_tr     = tr_idx[torch.randperm(len(tr_idx))]
    tr_total    = 0.0
    tr_price    = 0.0
    tr_sob_1    = 0.0
    tr_sob_2    = 0.0
    n_batches   = 0

    for i in range(0, len(perm_tr), effective_batch):
        b         = perm_tr[i : i + args.batch]
        loss_dict = sobolev_loss(X_norm[b], price[b], vega[b], delta[b], volga[b], gamma[b])
        
        optimizer.zero_grad()
        loss_dict["total"].backward()
        if args.clip_grad > 0:
            nn.utils.clip_grad_norm_(_base_model.parameters(), max_norm=args.clip_grad)
        optimizer.step()
        
        tr_total  += loss_dict["total"].item()
        tr_price  += loss_dict["price"].item()
        tr_sob_1  += loss_dict["sobolev_1"].item()
        tr_sob_2  += loss_dict["sobolev_2"].item()
        n_batches += 1

    scheduler.step()

    model.eval()
    with torch.no_grad():
        val_pred = _base_model(X_norm[va_idx])
        val_rmse = ((val_pred - price[va_idx]) ** 2).mean().item() ** 0.5

    history["epoch"].append(epoch)
    history["train_total"].append(tr_total / n_batches)
    history["train_price"].append(tr_price / n_batches)
    history["train_sobolev_1"].append(tr_sob_1 / n_batches)
    history["train_sobolev_2"].append(tr_sob_2 / n_batches)
    history["val_rmse"].append(val_rmse)

    if val_rmse < best_val_rmse:
        best_val_rmse  = val_rmse
        best_state     = {k: v.clone() for k, v in model.state_dict().items()}
        epochs_no_impr = 0
    else:
        epochs_no_impr += 1

    if epoch % args.log_every == 0:
        print(f"Epoch {epoch:3d}  total={tr_total/n_batches:.5f}  "
              f"[P={tr_price/n_batches:.5f} S1={tr_sob_1/n_batches:.5f} S2={tr_sob_2/n_batches:.5f}]  "
              f"val_rmse={val_rmse:.6f}  no_impr={epochs_no_impr}")

    if epochs_no_impr >= args.patience:
        print(f"Early stopping at epoch {epoch}  best_val_rmse={best_val_rmse:.6f}")
        break

print(f"Restoring best weights (val_rmse={best_val_rmse:.6f})")
model.load_state_dict(best_state)

if args.plot:
    epochs_arr   = np.array(history["epoch"])
    price_arr    = np.array(history["train_price"])
    sob1_arr     = np.array(history["train_sobolev_1"])
    sob2_arr     = np.array(history["train_sobolev_2"])

    stem, ext = os.path.splitext(args.plot)
    ext = ext or ".png"
    
    def ema(x: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        out = np.empty_like(x)
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
        return out

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#0e1117")
    ax.set_facecolor("#131720")
    ax.tick_params(colors="#c8ccd8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2f3f")
    ax.grid(True, color="#1e2230", linewidth=0.6, linestyle="--")

    ax.plot(epochs_arr, ema(price_arr), color="#f7c59f", linewidth=2.0, label="Price MAE")
    ax.plot(epochs_arr, ema(sob1_arr),  color="#7ec8e3", linewidth=2.0, label="1st Order Sobolev (Vega, Delta)")
    ax.plot(epochs_arr, ema(sob2_arr),  color="#a8d8a8", linewidth=2.0, label="2nd Order Sobolev (Volga, Gamma)")
    
    ax.legend(facecolor="#1e2230", edgecolor="#2a2f3f", labelcolor="#c8ccd8")
    fig.savefig(f"{stem}_components{ext}", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

torch.save({
    "state_dict":  _base_model.state_dict(),
    "X_mean":      X_mean.cpu(),
    "X_std":       X_std.cpu(),
    "input_cols":  INPUT_COLS,
    "n_hidden":    args.hidden,
    "n_layers":    args.layers,
}, args.out)