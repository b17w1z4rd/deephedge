"""
run.py - train the deep hedger and render the figure set.

Trains once, caches everything the figures need to figures/cache.npz, then draws
seven dark-themed, colorblind-safe figures. Re-run with --retrain to recompute.

    python3 run.py            # use cache if present, else compute
    python3 run.py --retrain  # force recompute
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

from market import gbm_paths, bs_price, bs_delta
from hedge import Policy, train, deep_wealth_np, delta_wealth_np, metrics

# --- validated palette (dark surface) -------------------------------------- #
SURFACE, PAGE = "#1a1a19", "#0d0d0d"
INK, INK2, MUTED = "#ffffff", "#c3c2b7", "#898781"
GRID, AXIS = "#2c2c2a", "#383835"
BLUE, ORANGE, AQUA, RED = "#3987e5", "#d95926", "#199e70", "#e66767"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUE_CMAP = LinearSegmentedColormap.from_list("blues", BLUE_RAMP)
DIVERGING = LinearSegmentedColormap.from_list("bwr", ["#3987e5", "#2c2c2a", "#e66767"])

FIGDIR = "figures"
CACHE = os.path.join(FIGDIR, "cache.npz")

CFG = dict(S0=100.0, K=100.0, r=0.0, sigma=0.2, T=1.0 / 12, n_steps=30,
           kappa_main=0.01, lam=1.0,
           kappas=[0.0, 0.0025, 0.005, 0.01, 0.02])


def set_style():
    plt.rcParams.update({
        "figure.facecolor": PAGE, "savefig.facecolor": PAGE,
        "axes.facecolor": SURFACE, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.grid": True, "axes.axisbelow": True,
        "font.family": "sans-serif", "font.size": 11,
        "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 140,
    })


# --------------------------------------------------------------------------- #
def compute_all(cfg):
    os.makedirs(FIGDIR, exist_ok=True)
    S0, K, r, sig, T, ns = (cfg[k] for k in ("S0", "K", "r", "sigma", "T", "n_steps"))
    lam, km = cfg["lam"], cfg["kappa_main"]
    premium = float(bs_price(S0, K, T, r, sig))

    Str = gbm_paths(S0, r, sig, T, ns, 40_000, seed=1)
    Ste = gbm_paths(S0, r, sig, T, ns, 100_000, seed=99)

    print(f"premium (BS) = {premium:.4f}")
    print(f"training headline hedger (kappa={km}) ...")
    pol = Policy(hidden=32, seed=0)
    hist = train(pol, Str, K, premium, km, T, lam=lam, iters=500, batch=2048, lr=1e-2)

    # eval wealth distributions
    unhedged = premium - np.maximum(Ste[:, -1] - K, 0.0)
    wd, dpos = deep_wealth_np(pol, Ste, K, premium, km, T, record=True)
    wb, bpos = delta_wealth_np(Ste, K, r, sig, premium, km, T, record=True)

    # a representative single path (moderate terminal move)
    order = np.argsort(Ste[:, -1])
    j = order[int(0.7 * len(order))]
    samp_S = Ste[j]; samp_deep = dpos[j]; samp_delta = bpos[j]

    # hedge-ratio surfaces over (moneyness, time-to-maturity), sliced at hprev=delta
    mgrid = np.linspace(0.85, 1.15, 60)       # S/K
    tgrid = np.linspace(T, T / ns, 40)        # time to maturity, expiry at bottom
    Sg = (mgrid[None, :] * K)
    delta_surf = np.zeros((len(tgrid), len(mgrid)))
    deep_surf = np.zeros((len(tgrid), len(mgrid)))
    for i, tau in enumerate(tgrid):
        d = bs_delta(Sg[0], K, tau, r, sig)
        delta_surf[i] = d
        base = np.stack([np.log(Sg[0] / K), np.full(len(mgrid), tau)], axis=1)
        deep_surf[i] = pol.act_np(base, d.reshape(-1, 1))[:, 0]

    # cost frontier: retrain deep at each kappa, record CVaR95 for both
    fr_delta, fr_deep = [], []
    for kap in cfg["kappas"]:
        if abs(kap - km) < 1e-12:
            p = pol
        else:
            print(f"training frontier hedger (kappa={kap}) ...")
            p = Policy(hidden=32, seed=0)
            train(p, Str, K, premium, kap, T, lam=lam, iters=250, batch=2048,
                  lr=1e-2, log_every=0)
        fr_deep.append(metrics(deep_wealth_np(p, Ste, K, premium, kap, T))["cvar95"])
        fr_delta.append(metrics(delta_wealth_np(Ste, K, r, sig, premium, kap, T))["cvar95"])

    res = dict(premium=premium, hist=np.array(hist),
               unhedged=unhedged, wd=wd, wb=wb,
               samp_S=samp_S, samp_deep=samp_deep, samp_delta=samp_delta,
               mgrid=mgrid, tgrid=tgrid, delta_surf=delta_surf, deep_surf=deep_surf,
               kappas=np.array(cfg["kappas"]),
               fr_delta=np.array(fr_delta), fr_deep=np.array(fr_deep),
               paths=Ste[:180], term_spot=Ste[:, -1])
    np.savez(CACHE, **res)
    return res


# --------------------------------------------------------------------------- #
def fig_paths(R, cfg):
    fig, ax = plt.subplots(figsize=(8, 5))
    P = R["paths"]
    xs = np.linspace(0, cfg["T"] * 12, P.shape[1])           # months
    term = P[:, -1]
    norm = (term - term.min()) / (np.ptp(term) + 1e-9)
    for i in range(P.shape[0]):
        ax.plot(xs, P[i], color=BLUE_CMAP(0.25 + 0.6 * norm[i]), lw=0.7, alpha=0.5)
    ax.axhline(cfg["K"], color=ORANGE, lw=1.5, ls="--", label="strike (K=100)")
    ax.set_title("Simulated price paths  ·  GBM, 1-month horizon")
    ax.set_xlabel("time (months)"); ax.set_ylabel("underlying price")
    ax.legend(frameon=False, labelcolor=INK2)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/1_paths.png"); plt.close(fig)


def fig_training(R, cfg):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    h = R["hist"]
    ax.plot(h, color=BLUE, lw=2)
    ax.set_title("Training  ·  entropic risk of terminal P&L")
    ax.set_xlabel("iteration"); ax.set_ylabel("entropic risk  (lower = better)")
    ax.annotate(f"final {h[-1]:.2f}", (len(h) - 1, h[-1]), color=INK2,
                xytext=(-70, 20), textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color=MUTED))
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/2_training.png"); plt.close(fig)


def _cvar(w, a=0.95):
    l = -w; v = np.quantile(l, a); return l[l >= v].mean()


def fig_pnl(R, cfg):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    xs = np.linspace(-8, 4, 500)
    for w, c, name in [(R["unhedged"], MUTED, "unhedged"),
                       (R["wb"], ORANGE, "delta hedge"),
                       (R["wd"], BLUE, "deep hedge")]:
        k = gaussian_kde(np.clip(w, -8, 4)); y = k(xs)
        ax.fill_between(xs, y, color=c, alpha=0.16)
        ax.plot(xs, y, color=c, lw=2, label=name)
    for w, c in [(R["wb"], ORANGE), (R["wd"], BLUE)]:
        ax.axvline(-_cvar(w), color=c, ls=":", lw=1.4, alpha=0.9)
    imp = (_cvar(R["wb"]) - _cvar(R["wd"])) / abs(_cvar(R["wb"])) * 100
    ax.set_title(f"Terminal P&L distribution  ·  {int(cfg['kappa_main']*100)}% cost   "
                 f"(deep hedge cuts 95% tail loss by {imp:.0f}%)")
    ax.set_xlabel("terminal wealth (P&L per option)"); ax.set_ylabel("density")
    ax.set_yticks([])
    ax.legend(frameon=False, labelcolor=INK2, loc="upper left")
    ax.text(0.99, 0.96, "dotted lines: 95% CVaR (tail loss)", transform=ax.transAxes,
            ha="right", va="top", color=MUTED, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/3_pnl_distribution.png"); plt.close(fig)


def fig_surfaces(R, cfg):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    m, t = R["mgrid"], R["tgrid"] * 12
    ext = [m[0], m[-1], t[-1], t[0]]
    for ax, surf, title, cmap in [
        (axes[0], R["delta_surf"], "Black-Scholes delta", BLUE_CMAP),
        (axes[1], R["deep_surf"], "Deep hedge policy", BLUE_CMAP)]:
        im = ax.imshow(surf, aspect="auto", extent=ext, cmap=cmap, vmin=0, vmax=1,
                       origin="upper", interpolation="bilinear")
        ax.set_title(title); ax.set_xlabel("moneyness S/K")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("time to maturity (months)")
    diff = R["deep_surf"] - R["delta_surf"]
    mx = np.abs(diff).max()
    im = axes[2].imshow(diff, aspect="auto", extent=ext, cmap=DIVERGING,
                        vmin=-mx, vmax=mx, origin="upper", interpolation="bilinear")
    axes[2].set_title("deep - delta  (learned adjustment)")
    axes[2].set_xlabel("moneyness S/K"); axes[2].grid(False)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle("Hedge ratio as a function of moneyness and time", y=1.02,
                 color=INK, fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/4_hedge_surface.png", bbox_inches="tight")
    plt.close(fig)


def fig_frontier(R, cfg):
    fig, ax = plt.subplots(figsize=(8, 5))
    k = R["kappas"] * 100
    ax.plot(k, R["fr_delta"], "-o", color=ORANGE, lw=2, ms=6, label="delta hedge")
    ax.plot(k, R["fr_deep"], "-o", color=BLUE, lw=2, ms=6, label="deep hedge")
    ax.fill_between(k, R["fr_deep"], R["fr_delta"], color=BLUE, alpha=0.10)
    ax.set_title("Tail risk vs transaction cost  ·  the deep hedge advantage grows")
    ax.set_xlabel("proportional transaction cost (%)")
    ax.set_ylabel("95% CVaR  (tail loss, lower = better)")
    ax.legend(frameon=False, labelcolor=INK2)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/5_cost_frontier.png"); plt.close(fig)


def fig_single(R, cfg):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.1, 1]))
    xs = np.linspace(0, cfg["T"] * 12, len(R["samp_S"]))
    a1.plot(xs, R["samp_S"], color=BLUE, lw=2)
    a1.axhline(cfg["K"], color=ORANGE, ls="--", lw=1.3, label="strike")
    a1.set_ylabel("underlying"); a1.set_title("One path  ·  spot, and how each strategy holds")
    a1.legend(frameon=False, labelcolor=INK2, loc="upper left")
    xh = xs[:-1]
    a2.step(xh, R["samp_delta"], color=ORANGE, lw=2, where="post", label="delta hedge")
    a2.step(xh, R["samp_deep"], color=BLUE, lw=2, where="post", label="deep hedge")
    a2.set_ylabel("hedge ratio (shares)"); a2.set_xlabel("time (months)")
    a2.legend(frameon=False, labelcolor=INK2, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/6_single_path.png"); plt.close(fig)


def fig_error(R, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, sharex=True)
    ST, K = R["term_spot"], cfg["K"]
    for ax, w, name in [(axes[0], R["wb"], "delta hedge"),
                        (axes[1], R["wd"], "deep hedge")]:
        hb = ax.hexbin(ST, w, gridsize=45, cmap=BLUE_CMAP, mincnt=1,
                       extent=(70, 135, -8, 4))
        ax.axhline(0, color=MUTED, lw=1, ls="--")
        ax.axvline(K, color=ORANGE, lw=1.2, ls="--")
        ax.set_title(name); ax.set_xlabel("terminal spot  $S_T$"); ax.grid(False)
    axes[0].set_ylabel("terminal wealth (P&L)")
    fig.suptitle("Hedging error vs terminal spot  ·  where each strategy wins and loses",
                 y=1.02, color=INK, fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/7_error_vs_spot.png", bbox_inches="tight")
    plt.close(fig)


def make_figures(R, cfg):
    set_style()
    fig_paths(R, cfg)
    fig_training(R, cfg)
    fig_pnl(R, cfg)
    fig_surfaces(R, cfg)
    fig_frontier(R, cfg)
    fig_single(R, cfg)
    fig_error(R, cfg)
    print(f"wrote figures to {FIGDIR}/")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    if "--retrain" in sys.argv or not os.path.exists(CACHE):
        R = compute_all(CFG)
    else:
        R = dict(np.load(CACHE))
        print("loaded cached results")
    make_figures(R, CFG)

    # headline metrics
    md, mb = metrics(R["wd"]), metrics(R["wb"])
    print("\n            mean      std     CVaR95   entropic")
    print(f"delta   {mb['mean']:+.4f}  {mb['std']:.4f}  {mb['cvar95']:+.4f}  {mb['entropic']:+.4f}")
    print(f"deep    {md['mean']:+.4f}  {md['std']:.4f}  {md['cvar95']:+.4f}  {md['entropic']:+.4f}")


if __name__ == "__main__":
    main()
