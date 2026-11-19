# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pyyaml"]
# ///
"""Hybrid-pipeline figure (paper/generated/body_hybrid.pdf) from results/offline_v4/hybrid.json.

One marker per primary set of jev-opt/experiments/offline_hybrid.py (the StrMatch second reference is left out):
x = share of items answered by Jev at the Learn-then-Test threshold (mean over the cross-fitted halvings),
y = hybrid accuracy minus the paper model's accuracy on the same items (points), with the paired cluster-bootstrap
95% interval. Filled markers: non-inferior at the 0.5-point margin after Holm. Reads only the saved JSON.
Run from the repo root:

    uv run scripts/paper_fig_hybrid.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import paper_figs_body as fb  # noqa: E402,F401  (sets the paper's rcParams)
from paper_fig_hero import JEV  # noqa: E402

SRC = ROOT / "results" / "offline_v4" / "hybrid.json"
OUT = ROOT / "paper" / "generated" / "body_hybrid.pdf"
COLW = 3.03  # ACL column width, inches
NOT_NI = "#c92a2a"  # fb.COLORS["below"]
# set key -> (label, label offset in points, horizontal alignment)
SETS = {
    "xstest": ("XSTest", (-5, 0), "right"),
    "cicc/mtod": ("MTOD", (-4, 7), "right"),
    "cicc/hwu64": ("HWU64", (5, -2), "left"),
    "cicc/acid": ("ACID", (-5, -2), "right"),
    "legalseg": ("LegalSeg", (5, -6), "left"),
    "cicc/c150": ("C150", (-3, 7), "right"),
    "cicc/atis": ("ATIS", (0, 8), "center"),
}


def main() -> int:
    d = json.loads(SRC.read_text())
    margin = 100 * d["margin"]
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig, ax = plt.subplots(figsize=(COLW, 1.3))
    ax.axhline(0, color="#adb5bd", lw=0.6, zorder=1)
    ax.axhline(-margin, color="#495057", lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(-2, -margin - 0.3, f"$-${margin:g} margin", fontsize=7, va="top", ha="left", color="#495057")
    ax.set_yticks([-2, 0, 2, 4, 6, 8])
    for key, (label, off, ha) in SETS.items():
        s = d["sets"][key]
        x = 100 * s["jev_coverage_mean"]
        y = 100 * s["diff_system_minus_paper"]
        lo, hi = (100 * v for v in s["diff_ci"])
        ni = s["p_noninferiority_holm"] < 0.05
        color = JEV if ni else NOT_NI
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="o" if ni else "s", ms=3.6, color=color,
                    mfc=color if ni else "white", mew=0.9, elinewidth=0.8, capsize=0, zorder=3)
        ax.annotate(label, (x, y), xytext=off, textcoords="offset points", fontsize=7, ha=ha, va="center")
    ax.set_xlim(-4, 104)
    ax.set_ylim(-3.0, 8.2)
    ax.set_xlabel("share of items answered by Jev (%)", labelpad=1)
    ax.set_ylabel("hybrid $-$ paper's\nmodel (points)", labelpad=1)
    ax.tick_params(length=2, pad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
