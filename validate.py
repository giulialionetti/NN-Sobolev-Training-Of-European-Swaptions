import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--val",          default="data/swaption_validation.csv")
parser.add_argument("--model",        default="swaption_model.pt")
parser.add_argument("--plot",         action="store_true")
parser.add_argument("--plot_stem",    default="validation")
parser.add_argument("--save_results", default=None)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

ckpt       = torch.load(args.model, map_location=device, weights_only=False)
INPUT_COLS = ckpt["input_cols"]
X_mean     = ckpt["X_mean"].to(device)
X_std      = ckpt["X_std"].to(device)
SIGMA_IDX  = INPUT_COLS.index("sigma")
R0_IDX     = INPUT_COLS.index("r0")

class MLP(nn.Module):
    def __init__(self, n_in, n_hidden, n_layers):
        super().__init__()
        layers = [nn.Linear(n_in, n_hidden), nn.Softplus()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Softplus()]
        layers.append(nn.Linear(n_hidden, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

model = MLP(len(INPUT_COLS), ckpt["n_hidden"], ckpt["n_layers"]).to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print(f"Loaded model : {ckpt['n_layers']} layers x {ckpt['n_hidden']} units  from {args.model}\n")

def predict(X_raw_tensor):
    X_norm = (X_raw_tensor - X_mean) / X_std
    x      = X_norm.detach().requires_grad_(True)
    p      = model(x)
    
    # First Order
    grad = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
    
    # Second Order (Gamma and Volga)
    grad_sigma = grad[:, SIGMA_IDX]
    grad_r0    = grad[:, R0_IDX]
    
    # Compute Hessians by differentiating the gradients
    grad2_sigma = torch.autograd.grad(grad_sigma.sum(), x, retain_graph=True)[0]
    grad2_r0    = torch.autograd.grad(grad_r0.sum(), x)[0]
    
    with torch.no_grad():
        vega  = grad[:, SIGMA_IDX:SIGMA_IDX+1] / X_std[SIGMA_IDX]
        delta = grad[:, R0_IDX:R0_IDX+1]       / X_std[R0_IDX]
        volga = grad2_sigma[:, SIGMA_IDX:SIGMA_IDX+1] / (X_std[SIGMA_IDX]**2)
        gamma = grad2_r0[:, R0_IDX:R0_IDX+1]          / (X_std[R0_IDX]**2)
        
    return p.detach(), vega.detach(), delta.detach(), volga.detach(), gamma.detach()

def metrics(pred, true, name, threshold=1e-2):
    pred = np.array(pred).flatten()
    true = np.array(true).flatten()
    ae   = np.abs(pred - true)
    re   = ae / (np.abs(true) + 1e-4)
    mask = np.abs(true) > threshold
    print(f"  {name}")
    print(f"    MAE         : {ae.mean():.6f}")
    print(f"    RMSE        : {np.sqrt((ae**2).mean()):.6f}")
    print(f"    rel.MAE     : {re.mean():.2%}  (all samples)")
    if mask.sum() > 0:
        print(f"    rel.MAE     : {re[mask].mean():.2%}  (on {mask.sum()} samples with |value| > {threshold})")
    print(f"    max |error| : {ae.max():.6f}")
    print()
    return ae, re

def bucket_breakdown(pred, true, label="price"):
    pred   = np.array(pred).flatten()
    true   = np.array(true).flatten()
    ae     = np.abs(pred - true)
    ptiles = np.percentile(true, [0, 25, 50, 75, 90, 100])
    print(f"  {label} error by price percentile bucket:")
    for lo, hi in zip(ptiles[:-1], ptiles[1:]):
        mask = (true >= lo) & (true < hi)
        if mask.sum() == 0:
            continue
        print(f"    [{lo:.5f}, {hi:.5f})  n={mask.sum():4d}  MAE={ae[mask].mean():.6f}  max={ae[mask].max():.6f}")
    print()

df = pd.read_csv(args.val)
df = df.dropna().reset_index(drop=True)
print(f"Loaded {len(df)} validation samples from {args.val}")

X_raw = torch.tensor(df[INPUT_COLS].values, dtype=torch.float32).to(device)

an_price = df["an_price"].to_numpy(dtype=np.float64)
an_vega  = df["an_vega"].to_numpy(dtype=np.float64)
an_delta = df["an_delta"].to_numpy(dtype=np.float64)
an_volga = df["an_volga"].to_numpy(dtype=np.float64)
an_gamma = df["an_gamma"].to_numpy(dtype=np.float64)

mc_price = df["mc_price"].to_numpy(dtype=np.float64)
mc_vega  = df["mc_vega"].to_numpy(dtype=np.float64)
mc_delta = df["mc_delta"].to_numpy(dtype=np.float64)
mc_volga = df["mc_volga"].to_numpy(dtype=np.float64)
mc_gamma = df["mc_gamma"].to_numpy(dtype=np.float64)

p_p, v_p, d_p, vo_p, g_p = predict(X_raw)
nn_price = p_p.cpu().numpy().flatten()
nn_vega  = v_p.cpu().numpy().flatten()
nn_delta = d_p.cpu().numpy().flatten()
nn_volga = vo_p.cpu().numpy().flatten()
nn_gamma = g_p.cpu().numpy().flatten()

print("=" * 70)
print("1. NN vs ANALYTICAL")
print("=" * 70)
print("--- Price ---")
ae_nn_price, re_nn_price = metrics(nn_price, an_price, "NN vs Analytical")
bucket_breakdown(nn_price, an_price, "price")
print("--- Vega ---")
metrics(nn_vega, an_vega, "NN vs Analytical", threshold=1e-2)
print("--- Delta ---")
metrics(nn_delta, an_delta, "NN vs Analytical", threshold=1e-3)
print("--- Volga ---")
metrics(nn_volga, an_volga, "NN vs Analytical", threshold=1e-1)
print("--- Gamma ---")
metrics(nn_gamma, an_gamma, "NN vs Analytical", threshold=1e-1)

print("=" * 70)
print("2. MC vs ANALYTICAL (noise floor)")
print("=" * 70)
print("--- Price ---")
metrics(mc_price, an_price, "MC vs Analytical")
print("--- Vega ---")
metrics(mc_vega, an_vega, "MC vs Analytical", threshold=1e-2)
print("--- Delta ---")
metrics(mc_delta, an_delta, "MC vs Analytical", threshold=1e-3)
print("--- Volga ---")
metrics(mc_volga, an_volga, "MC vs Analytical", threshold=1e-1)
print("--- Gamma ---")
metrics(mc_gamma, an_gamma, "MC vs Analytical", threshold=1e-1)

print("=" * 70)
print("3. SUMMARY")
print("=" * 70)
nn_rmse = np.sqrt(((nn_price - an_price)**2).mean())
mc_rmse = np.sqrt(((mc_price - an_price)**2).mean())

print(f"\n  {'Metric':<25} {'NN':>12} {'MC (floor)':>12} {'NN/MC':>8}")
print(f"  {'-'*60}")
print(f"  {'Price RMSE':<25} {nn_rmse:>12.6f} {mc_rmse:>12.6f} {nn_rmse/mc_rmse:>7.2f}x")
print(f"  {'Vega MAE':<25} {np.abs(nn_vega - an_vega).mean():>12.6f} {np.abs(mc_vega - an_vega).mean():>12.6f}")
print(f"  {'Delta MAE':<25} {np.abs(nn_delta - an_delta).mean():>12.6f} {np.abs(mc_delta - an_delta).mean():>12.6f}")
print(f"  {'Volga MAE':<25} {np.abs(nn_volga - an_volga).mean():>12.6f} {np.abs(mc_volga - an_volga).mean():>12.6f}")
print(f"  {'Gamma MAE':<25} {np.abs(nn_gamma - an_gamma).mean():>12.6f} {np.abs(mc_gamma - an_gamma).mean():>12.6f}")
print()
print(f"  Price verdict: {'NN beats MC noise floor' if nn_rmse < mc_rmse else 'MC still better'}")

if args.plot:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Agg")
    
    S = args.plot_stem
    DARK = {
        "bg":     "#0e1117", 
        "ax_bg":  "#131720", 
        "text":   "#c8ccd8", 
        "nn":     "#7ec8e3", 
        "mc":     "#f7c59f", 
        "vega":   "#a8d8a8",
        "delta":  "#c9a8e0",
        "ref":    "#ffe066"
    }

    def make_plot(x, y, title, xlabel, ylabel, suffix, color):
        x_clean = x[~np.isnan(x)]
        y_clean = y[~np.isnan(y)]
        
        if len(x_clean) == 0 or len(y_clean) == 0:
            print(f"      Skipping {suffix}: No valid data points.")
            return

        fig, ax = plt.subplots(figsize=(7, 5), facecolor=DARK["bg"])
        ax.set_facecolor(DARK["ax_bg"])
        
        ax.scatter(x, y, s=6, alpha=0.4, color=color, edgecolors='none', rasterized=True)
        
        combined = np.concatenate([x_clean, y_clean])
        lo = np.percentile(combined, 1)
        hi = np.percentile(combined, 99)
        
        ax.plot([lo, hi], [lo, hi], color=DARK["ref"], lw=1.5, ls="--", label="y=x")
        
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        
        ax.set_title(title, color=DARK["text"], fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel, color=DARK["text"])
        ax.set_ylabel(ylabel, color=DARK["text"])
        ax.tick_params(colors=DARK["text"])
        
        ax.grid(True, color="#1e2230", alpha=0.5)
        ax.legend(facecolor=DARK["ax_bg"], edgecolor="#2a2f3f", labelcolor=DARK["text"])
        
        fig.tight_layout()
        filename = f"{S}_{suffix}.png"
        fig.savefig(filename, facecolor=fig.get_facecolor(), dpi=150)
        plt.close(fig)
        print(f"  -> {filename}")

    print("\nGenerating scatter plots:")
    make_plot(an_price, nn_price, "Price: NN vs Analytical", "Analytical Price", "NN Price", "price_scatter", DARK["nn"])
    make_plot(an_vega, nn_vega, "Vega: NN vs Analytical", "Analytical Vega", "NN Vega", "vega_scatter", DARK["vega"])
    make_plot(an_delta, nn_delta, "Delta: NN vs Analytical", "Analytical Delta", "NN Delta", "delta_scatter", DARK["delta"])
    make_plot(an_volga, nn_volga, "Volga: NN vs Analytical", "Analytical Volga", "NN Volga", "volga_scatter", "#ff9999")
    make_plot(an_gamma, nn_gamma, "Gamma: NN vs Analytical", "Analytical Gamma", "NN Gamma", "gamma_scatter", "#ffff99")

    print("All plots generated successfully.")