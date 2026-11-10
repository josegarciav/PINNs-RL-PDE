"""J11 figure: what a spray dryer's instruments determine of the drying-rate closure.

Usage: python experiments/J11_dryer_closure/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run as J  # noqa: E402

results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "j11_dryer_closure.png"
d = json.loads((results / "j11.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN, AMBER, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e5bd9"
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
    }
)
fig, axes = plt.subplots(2, 2, figsize=(17, 11.5), dpi=200, facecolor=SURFACE)
for ax in axes.ravel():
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def lin(kind, camp, tower, known):
    return next(
        r
        for r in d["linear"]
        if (r["kind"], r["campaign"], r["sample_tower"], r["known"]) == (kind, camp, tower, known)
    )


def summ(kind, camp, tower, known):
    return next(
        r
        for r in d["summary"]
        if (r["kind"], r["campaign"], r["sample_tower"], r["known"]) == (kind, camp, tower, known)
    )


# a) the tower
ax = axes[0, 0]
ax2 = ax.twinx()
ax2.spines["top"].set_visible(False)
for name, style, label in (
    ("base", "-", "base point"),
    ("new", "--", "new point (colder, more loaded)"),
):
    p = d["profiles"][name]
    z = np.array(p["z"])
    ax.plot(z, np.array(p["T_a"]) - 273.15, style, color=ORANGE, linewidth=2, label=f"air, {label}")
    ax.plot(
        z,
        np.array(p["T_p"]) - 273.15,
        style,
        color=AMBER,
        linewidth=1.5,
        label=f"droplets, {label}",
    )
    ax2.plot(z, p["X"], style, color=BLUE, linewidth=2)
base = d["profiles"]["base"]
ax.plot(
    d["probes"],
    np.interp(d["probes"], base["z"], base["T_a"]) - 273.15,
    "o",
    color=INK,
    markersize=6,
    label="air thermocouples",
)
ax.set_xlabel("distance below the atomiser (m)")
ax.set_ylabel("temperature (°C)")
ax2.set_ylabel("droplet moisture X, dry basis (blue)", color=BLUE)
ax2.tick_params(axis="y", colors=BLUE)
ax.set_title("a) the tower: air and droplets along the height", loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper right")

# b) the free-form closure
ax = axes[0, 1]
drop = J.tower().droplet
k = J.knots()
grid = np.linspace(drop.X_eq, drop.X0, 300)
truth = drop.drying_curve(grid)
z90 = 1.6448536269514722
for tower, color, label in (
    (False, ORANGE, "plant instruments, 5 operating points"),
    (True, GREEN, "+ powder sampled at 5 heights"),
):
    r = lin("free", "5 points", tower, False)
    std = np.concatenate([[0.0], r["std"][: len(k) - 1]])
    centre = drop.drying_curve(k)
    ax.fill_between(
        k,
        centre - z90 * std,
        centre + z90 * std,
        color=color,
        alpha=0.2,
        linewidth=0,
        label=f"90 % posterior band: {label}",
    )
ax.fill_between(
    k,
    drop.drying_curve(k) - z90 * 0.3,
    drop.drying_curve(k) + z90 * 0.3,
    color=INK2,
    alpha=0.07,
    linewidth=0,
    label="the prior (0.3 per knot)",
)
ax.plot(grid, truth, color=INK, linewidth=2.4, label="the plant's curve")
shown = False
for f in d["fits"]:
    if (
        f["kind"] == "free"
        and f["campaign"] == "5 points"
        and not f["sample_tower"]
        and not f["known"]
        and f["f_hat"]
    ):
        ax.plot(
            grid,
            f["f_hat"],
            color=PURPLE,
            linewidth=1.2,
            label=None if shown else "three fitted closures (plant instruments)",
        )
        shown = True
ax.set_ylim(-0.25, 1.6)
ax.set_xlabel("moisture X (dry basis)")
ax.set_ylabel("drying-rate closure f(X)")
ax.set_title("b) a free-form closure: what the instruments pin down", loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper left")

# c) the two numbers of the characteristic curve
ax = axes[1, 0]
configs = [
    ("5 points\nwall loss and droplet size known", ("cdc", "5 points", False, True)),
    ("5 points\nboth uncertain (30 %, 10 %)", ("cdc", "5 points", False, False)),
    ("1 point\nboth uncertain", ("cdc", "1 point", False, False)),
]
x = np.arange(len(configs))
for j, (pname, color) in enumerate(
    (("critical moisture $X_{cr}$ (truth 1.0)", BLUE), ("exponent n (truth 1.5)", GREEN))
):
    pred = [summ(*c)["param_std_predicted"][j] for _, c in configs]
    found = [summ(*c)["param_std_found"][j] for _, c in configs]
    ax.bar(
        x + (j - 0.5) * 0.36,
        found,
        width=0.32,
        color=color,
        label=f"{pname}: spread of {len([f for f in d['fits'] if f['kind'] == 'cdc']) // 3} fits",
    )
    ax.plot(
        x + (j - 0.5) * 0.36,
        pred,
        "_",
        color=INK,
        markersize=26,
        mew=2.5,
        label="predicted (linear)" if j == 0 else None,
    )
ax.set_xticks(x)
ax.set_xticklabels([c[0] for c in configs], fontsize=9)
ax.set_ylabel("standard deviation of the estimate")
ax.set_title("c) the characteristic curve's two parameters", loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.grid(axis="x", visible=False)

# d) the prediction at a new operating point
ax = axes[1, 1]
rows = [
    ("curve form known, 5 points\nnuisances known", ("cdc", "5 points", False, True), BLUE),
    ("curve form known, 5 points\nnuisances uncertain", ("cdc", "5 points", False, False), BLUE),
    ("curve form known, 1 point\nnuisances uncertain", ("cdc", "1 point", False, False), BLUE),
    ("free form, 5 points\nnuisances known", ("free", "5 points", False, True), PURPLE),
    ("free form, 5 points\nnuisances uncertain", ("free", "5 points", False, False), PURPLE),
    (
        "free form, 5 points + tower samples\nnuisances uncertain",
        ("free", "5 points", True, False),
        PURPLE,
    ),
]
y = np.arange(len(rows))[::-1]
xn = d["new_point_X"]
ax.barh(
    y,
    [100 * summ(*c)["new_point_rmse"] / xn for _, c, _ in rows],
    color=[c for _, _, c in rows],
    height=0.6,
)
ax.plot(
    [100 * summ(*c)["new_point_std_predicted"] / xn for _, c, _ in rows],
    y,
    "|",
    color=INK,
    markersize=16,
    mew=2.5,
    label="predicted (linear)",
)
for yi, (_, c, _) in zip(y, rows, strict=True):
    s = summ(*c)
    ax.text(
        100 * max(s["new_point_rmse"], s["new_point_std_predicted"]) / xn + 0.06,
        yi,
        f"closure off by {100 * s['closure_rel_l2_median']:.0f} %",
        va="center",
        fontsize=9,
        color=INK2,
    )
ax.axvline(
    100 * d["noise"]["X_out"] / xn,
    color=INK,
    linestyle=":",
    linewidth=1.4,
    label="one lab assay of the powder",
)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=9)
ax.set_xlim(0, 3.2)
ax.set_xlabel(f"error of the predicted powder moisture at the new point, % of its value ({xn:.2f})")
ax.set_title("d) what a twin is asked: the powder at a new operating point", loc="left")
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.grid(axis="y", visible=False)

fig.suptitle(
    "J11: the drying-rate closure at dryer scale, from air thermocouples, outlet humidity and powder moisture",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
