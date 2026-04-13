"""

Sobolev training for an analytical swaption pricer neural network.

The network learns the mapping: phi(a, sigma, r0, T, swap_length, K)  -->  swaption price

Loss function follows Saadeddine (2022) "Fast Calibration using Complex-Step
Sobolev Training", Propositions 1 & 2:

    L = RelMSE(price) + E[(u^T ∇φ  -  u^T ∇label)²]

where u_k ~ ±sqrt(lambda_k) (Rademacher, scaled) is drawn fresh each batch.
This replaces the per-greek MSE terms with a single directional derivative,
equivalent in expectation (Prop. 1) and optimal in variance (Prop. 2), while
requiring only ONE autograd.grad call regardless of the number of supervised dims.


Usage:
    python train.py                   # expects data/swaption_data.csv
    python train.py --lambda_v 5.0    # reduce greek weight if loss is unstable

"""

import argparse
import torch 
import torch.nn as nn
import pandas as pd


parser = argparse.ArgumentParser()

# Training hyperparameters and options
parser.add_argument("--data",         default="data/swaption_data.csv")
parser.add_argument("--epochs",       type=int,   default=3000,
                    help="max epochs; early stopping may halt training sooner")
parser.add_argument("--patience",     type=int,   default=200,
                    help="early-stopping patience in epochs")
parser.add_argument("--batch",        type=int,   default=512,
                    help="smaller batch for more gradient steps/epoch and implicit regularisation")
parser.add_argument("--lr",           type=float, default=1e-3)
parser.add_argument("--hidden",       type=int,   default=128)
parser.add_argument("--layers",       type=int,   default=4,
                    help="3 layers keeps param count well below 20%% of 10k samples")
# Sobolev weights — scale the Rademacher direction per Proposition 2.
# Increased from 5.0 → 10.0: greek supervision is the primary regulariser
# when sample density is low.
parser.add_argument("--lambda_v",     type=float, default=5.0,
                    help="weight on vega   (d price / d sigma)")
parser.add_argument("--lambda_d",     type=float, default=5.0,
                    help="weight on delta  (d price / d r0)")
parser.add_argument("--out",          default="swaption_model.pt")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}"
      + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))



df = pd.read_csv(args.data)
print(f"Loaded {len(df)} samples from {args.data}")
if len(df) > 50_000:
    print("WARNING: dataset has >50k samples — consider reverting to default "
          "hyperparameters (layers=4, hidden=128, batch=512, lambda_v/d=5.0).")

INPUT_COLS = ["a", "sigma", "r0", "T", "swap_length", "K"]
SIGMA_IDX  = INPUT_COLS.index("sigma")   # = 1
R0_IDX     = INPUT_COLS.index("r0")     # = 2

X_raw  = torch.tensor(df[INPUT_COLS].values, dtype=torch.float32)
price  = torch.tensor(df["price"].values,    dtype=torch.float32).unsqueeze(1)
vega   = torch.tensor(df["vega"].values,     dtype=torch.float32).unsqueeze(1)
delta  = torch.tensor(df["delta"].values,    dtype=torch.float32).unsqueeze(1)
volga  = torch.tensor(df["volga"].values,    dtype=torch.float32).unsqueeze(1)
gamma  = torch.tensor(df["gamma"].values,    dtype=torch.float32).unsqueeze(1)


#  Normalisation 
X_mean = X_raw.mean(0)
X_std  = X_raw.std(0).clamp(min=1e-8)
X_norm = (X_raw - X_mean) / X_std

# Move everything to GPU once 
X_norm = X_norm.to(device)
X_std  = X_std.to(device)   # used inside sobolev_loss for de-normalisation
price  = price.to(device)
vega   = vega.to(device)
delta  = delta.to(device)
volga  = volga.to(device)
gamma  = gamma.to(device)


# ── Train / validation split ──────────────────────────────────────────────────
N      = len(df)
perm   = torch.randperm(N)
tr_idx = perm[:int(0.9 * N)]
va_idx = perm[int(0.9 * N):]

n_params_budget = int(0.20 * len(tr_idx))
print(f"Training samples: {len(tr_idx)}  |  20%%-param budget: {n_params_budget}")


# ── Model ─────────────────────────────────────────────────────────────────────
# Softplus is analytic and infinitely differentiable everywhere — required for
# the second-order Sobolev terms and consistent with the literature.
class MLP(nn.Module):
    def __init__(self, n_in=6, n_hidden=64, n_layers=3):
        super().__init__()
        layers = [nn.Linear(n_in, n_hidden), nn.Softplus()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Softplus()]
        layers.append(nn.Linear(n_hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model     = MLP(n_hidden=args.hidden, n_layers=args.layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# Cosine LR decay: smoothly anneals from lr → eta_min over all epochs.
# More effective than a fixed LR for Sobolev training at small data regimes.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=1e-5
)
n_params  = sum(p.numel() for p in model.parameters())

print(f"Model: {args.layers} hidden layers x {args.hidden} units  ({n_params} parameters)")
if n_params > n_params_budget:
    print(f"WARNING: model has {n_params} params but 20%% budget is {n_params_budget}. "
          "Consider reducing --hidden or --layers to avoid overfitting.")
print(f"Sobolev weights (Rademacher):  lambda_v={args.lambda_v}  lambda_d={args.lambda_d}")


# ── Sobolev loss (Saadeddine Complex-Step Sobolev Training, Propositions 1 & 2) ──
#
# Instead of supervising each greek separately (O(n_greeks) autograd.grad calls),
# we replace the sum of greek MSE terms with a single directional derivative along
# a random Rademacher direction scaled by sqrt(lambda_k) per input dimension.
#
# From Proposition 1 (Saadeddine 2022):
#   E[(u^T ∇φ - u^T ∇label)²] = Σ_k lambda_k * E[(∂_k φ - ∂_k label)²]
# provided cov(u) = diag(lambda_1, ..., lambda_{n+2}).
#
# From Proposition 2, the Rademacher distribution u_k = ±sqrt(lambda_k) is
# optimal — it minimises the additional variance from randomising the direction.
#
# Result: one autograd.grad call regardless of how many dimensions are supervised,
# with exact equivalence in expectation to the per-greek MSE formulation.
def sobolev_loss(x_n, y_price, y_vega, y_delta, x_std):
    # Fresh leaf tensor so autograd can track derivatives w.r.t. inputs.
    x_n    = x_n.detach().requires_grad_(True)
    y_pred = model(x_n)                                    # (B, 1)

    # Relative price MSE: equalises importance across the price range.
    # The 1e-4 floor prevents division by zero for near-zero prices.
    loss = (((y_pred - y_price) / (y_price.abs() + 1e-4)) ** 2).mean()

    # ── Optimal Rademacher direction (Proposition 2) ──────────────────────────
    # lambda_k > 0 only for the dimensions we supervise (sigma → vega, r0 → delta).
    # u_k ~ ±sqrt(lambda_k); unsupervised dimensions get u_k = 0.
    lambdas = torch.zeros(len(INPUT_COLS), device=x_n.device)
    lambdas[SIGMA_IDX] = args.lambda_v
    lambdas[R0_IDX]    = args.lambda_d

    # Draw fresh Rademacher signs each forward pass for unbiased estimation.
    signs = 2 * torch.randint(0, 2, (x_n.shape[0], len(INPUT_COLS)),
                              device=x_n.device).float() - 1   # (B, n_in), values in {-1, +1}
    u = signs * lambdas.sqrt()                                  # (B, n_in), zero for unsupervised dims

    # ── Network directional derivative: u^T ∇φ(x) ────────────────────────────
    # One autograd.grad call — cost is the same regardless of how many dims
    # are supervised, unlike the per-greek formulation which scales linearly.
    grad   = torch.autograd.grad(y_pred.sum(), x_n, create_graph=True)[0]  # (B, n_in)
    net_dd = (grad * u).sum(dim=1, keepdim=True)                            # (B, 1)

    # ── Analytical label directional derivative: u^T ∇label(x) ──────────────
    # Re-normalise greeks back to the normalised-input scale to match grad's scale.
    # (grad is d(output)/d(x_norm), so label must also be in normalised units.)
    vega_norm  = y_vega  * x_std[SIGMA_IDX]   # (B, 1)
    delta_norm = y_delta * x_std[R0_IDX]      # (B, 1)
    label_dd   = (u[:, SIGMA_IDX:SIGMA_IDX+1] * vega_norm
                + u[:, R0_IDX:R0_IDX+1]       * delta_norm)    # (B, 1)

    loss += ((net_dd - label_dd) ** 2).mean()

    return loss


# ── Training loop with early stopping ────────────────────────────────────────
best_val_rmse  = float("inf")
best_state     = None
epochs_no_impr = 0   # consecutive epochs without validation improvement

for epoch in range(1, args.epochs + 1):
    # train/eval mode does NOT affect autograd.grad; requires_grad_(True) on
    # the input tensor (inside sobolev_loss) is what enables Sobolev differentiation.
    # train() is called here as good practice in case dropout/batchnorm are added later.
    model.train()
    perm_tr   = tr_idx[torch.randperm(len(tr_idx))]
    tr_loss   = 0.0
    n_batches = 0

    for i in range(0, len(perm_tr), args.batch):
        b    = perm_tr[i : i + args.batch]
        loss = sobolev_loss(X_norm[b], price[b], vega[b], delta[b], X_std)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tr_loss   += loss.item()
        n_batches += 1

    scheduler.step()   # cosine LR decay step

    # ── Validation + early stopping (every epoch, cheap) ─────────────────────
    model.eval()
    with torch.no_grad():
        val_pred = model(X_norm[va_idx])
        val_rmse = ((val_pred - price[va_idx]) ** 2).mean().item() ** 0.5

    if val_rmse < best_val_rmse:
        best_val_rmse  = val_rmse
        best_state     = {k: v.clone() for k, v in model.state_dict().items()}
        epochs_no_impr = 0
    else:
        epochs_no_impr += 1

    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d}  "
              f"train_loss={tr_loss/n_batches:.6f}  "
              f"val_price_rmse={val_rmse:.6f}  "
              f"best={best_val_rmse:.6f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"no_impr={epochs_no_impr}")

    if epochs_no_impr >= args.patience:
        print(f"Early stopping at epoch {epoch}  "
              f"(no improvement for {args.patience} epochs)  "
              f"best_val_rmse={best_val_rmse:.6f}")
        break

# Restore best weights before saving
print(f"Restoring best weights (val_rmse={best_val_rmse:.6f})")
model.load_state_dict(best_state) # type: ignore

torch.save({
    "state_dict": model.state_dict(),
    "X_mean":     X_mean.cpu(),
    "X_std":      X_std.cpu(),
    "input_cols": INPUT_COLS,
    "n_hidden":   args.hidden,
    "n_layers":   args.layers,
}, args.out)
print(f"Saved {args.out}")


# ── Final validation metrics ──────────────────────────────────────────────────
model.eval()

# Price MAE: clean no_grad forward pass — no stale tensors, no spurious grad_fn.
with torch.no_grad():
    y_va_pred_clean = model(X_norm[va_idx])
    price_mae      = (y_va_pred_clean - price[va_idx]).abs().mean().item()
    price_rel_mae  = ((y_va_pred_clean - price[va_idx]).abs()
                      / (price[va_idx].abs() + 1e-4)).mean().item()

# Greek MAEs: separate forward pass with requires_grad enabled on the input.
x_va      = X_norm[va_idx].detach().requires_grad_(True)
y_va_pred = model(x_va)
grad_va   = torch.autograd.grad(y_va_pred.sum(), x_va)[0]

with torch.no_grad():
    vega_pred_va  = grad_va[:, SIGMA_IDX:SIGMA_IDX+1] / X_std[SIGMA_IDX]
    delta_pred_va = grad_va[:, R0_IDX:R0_IDX+1]       / X_std[R0_IDX]
    vega_mae      = (vega_pred_va  - vega[va_idx]).abs().mean().item()
    delta_mae     = (delta_pred_va - delta[va_idx]).abs().mean().item()

print(f"Val MAE      price={price_mae:.6f}  vega={vega_mae:.6f}  delta={delta_mae:.6f}")
print(f"Val rel.MAE  price={price_rel_mae:.4%}")