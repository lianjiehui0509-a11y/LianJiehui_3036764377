#!/usr/bin/env python3
"""
Card & Krueger (1994) — "Minimum Wages and Employment"
Replication of Tables 2, 3, 4 and Figure 1, plus an independent extension
(chain heterogeneity in the employment response).

Reference:
    Card, David, and Alan B. Krueger (1994). "Minimum Wages and Employment:
    A Case Study of the Fast-Food Industry in New Jersey and Pennsylvania."
    American Economic Review, 84(4), 772-793.

Data:
    Princeton public replication data for the paper, provided as `public.csv`
    (410 restaurants surveyed in waves 1 and 2). See DATA_DICTIONARY.md.

Run:
    python analysis/ck1994_replication.py

This script is fully self-contained: after the raw data are placed in
`data/public.csv`, it reconstructs all variables and regenerates every table
and figure in `output/` without manual edits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "public.csv"
OUT = ROOT / "output"
OUT_PROCESSED = OUT / "processed"
OUT_TABLES = OUT / "tables"
OUT_FIGURES = OUT / "figures"
for d in (OUT_PROCESSED, OUT_TABLES, OUT_FIGURES):
    d.mkdir(parents=True, exist_ok=True)

# Design-system categorical palette (colorblind-safe). NJ = blue, PA = orange.
NJ_COLOR = "#2a78d6"
PA_COLOR = "#eb6834"
INK = "#0b0b0b"
GRID = "#e1e0d9"

# --------------------------------------------------------------------------- #
# 1. Load data and construct variables
# --------------------------------------------------------------------------- #
df = pd.read_csv(DATA)

# Full-time-equivalent employment: full-time workers (incl. managers) count 1,
# part-time workers count 0.5.  (Paper, footnote in Section II.)
df["fte1"] = df["EMPFT"] + df["NMGRS"] + 0.5 * df["EMPPT"]
df["fte2"] = df["EMPFT2"] + df["NMGRS2"] + 0.5 * df["EMPPT2"]

# Wave-2 employment treatment (paper, Table 3 notes):
#   - permanently closed stores (STATUS2 == 3)      -> employment = 0
#   - temporarily closed / renovated (2, 4, 5) and refusals (0) -> missing
df.loc[df["STATUS2"] == 3, "fte2"] = 0.0
df.loc[df["STATUS2"].isin([0, 2, 4, 5]), "fte2"] = np.nan

# Change in FTE employment
df["dfte"] = df["fte2"] - df["fte1"]

# Initial wage gap: proportional increase in the starting wage needed to reach
# the new NJ minimum ($5.05).  Zero in Pennsylvania and in NJ stores already
# paying at or above $5.05.  (Paper, Section III-B.)
df["gap"] = 0.0
low = (df["STATE"] == 1) & df["WAGE_ST"].notna() & (df["WAGE_ST"] < 5.05)
df.loc[low, "gap"] = (5.05 - df.loc[low, "WAGE_ST"]) / df.loc[low, "WAGE_ST"]

# Starting-wage change (first-stage check)
df["dwage"] = df["WAGE_ST2"] - df["WAGE_ST"]

# Full-meal price (after-tax): medium soda + small fries + entree.
df["price1"] = df["PSODA"] + df["PFRY"] + df["PENTREE"]
df["price2"] = df["PSODA2"] + df["PFRY2"] + df["PENTREE2"]
df["dlogprice"] = np.log(df["price2"]) - np.log(df["price1"])

# Chain dummies (base = Burger King), company-ownership dummy.
chain_dummies = pd.get_dummies(df["CHAIN"], prefix="chain", drop_first=True).astype(float)
df = pd.concat([df, chain_dummies], axis=1)

# Region indicators (North / Central / South New Jersey and the two eastern
# Pennsylvania sub-regions) already exist as columns in `public.csv`
# (NORTHJ, CENTRALJ, SOUTHJ, PA1, PA2); they are used directly below.

df["NJ"] = (df["STATE"] == 1).astype(float)

# --------------------------------------------------------------------------- #
# 2. Sample definitions
# --------------------------------------------------------------------------- #
# "All available observations" (Table 3, rows 1-3): non-missing wave outcome.
# "Balanced sample" (Table 3, row 4): employment in both waves.
# "Table 4 sample": employment in both waves, wave-1 wage, and wave-2 wage
#   (waived for permanently closed stores, which cannot report a wave-2 wage).
t4 = df[
    df["fte1"].notna()
    & df["fte2"].notna()
    & df["WAGE_ST"].notna()
    & (df["WAGE_ST2"].notna() | (df["STATUS2"] == 3))
].copy()

# Save the processed analysis dataset (documented input -> reproducible output).
proc_cols = [
    "SHEET", "CHAIN", "CO_OWNED", "STATE", "NJ", "SOUTHJ", "CENTRALJ", "NORTHJ",
    "PA1", "PA2", "SHORE", "STATUS2", "fte1", "fte2", "dfte", "WAGE_ST",
    "WAGE_ST2", "dwage", "gap", "price1", "price2", "dlogprice",
    "EMPFT", "EMPPT", "NMGRS", "EMPFT2", "EMPPT2", "NMGRS2", "HRSOPEN",
    "HRSOPEN2",
]
df[proc_cols].to_csv(OUT_PROCESSED / "ck1994_analysis.csv", index=False)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def mean_se(series: pd.Series) -> tuple[float, float, int]:
    """Return (mean, standard error, n) over non-missing observations."""
    s = series.dropna()
    return s.mean(), s.std(ddof=1) / np.sqrt(len(s)), len(s)


def did_pair(chg_nj: pd.Series, chg_pa: pd.Series) -> tuple[float, float, float]:
    """Difference-in-differences of two independent group means."""
    m_nj, se_nj, _ = mean_se(chg_nj)
    m_pa, se_pa, _ = mean_se(chg_pa)
    did = m_nj - m_pa
    se = np.sqrt(se_nj**2 + se_pa**2)
    return did, se, did / se


def fmt(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


def ols_row(y: pd.Series, X: pd.DataFrame, label: str) -> dict:
    """Fit an OLS model and return a compact summary dict."""
    X = sm.add_constant(X)
    res = sm.OLS(y, X).fit()
    return {
        "label": label,
        "n": int(res.nobs),
        "r2": res.rsquared,
        "coef": res.params.drop("const"),
        "se": res.bse.drop("const"),
    }


# --------------------------------------------------------------------------- #
# 3. Table 2 — means of key variables (wave 1 and wave 2, by state)
# --------------------------------------------------------------------------- #
def table2():
    rows = []
    for wave, (fte, wage, pct_full) in enumerate(
        [
            ("fte1", "WAGE_ST", lambda x: x["EMPFT"] + x["NMGRS"]),
            ("fte2", "WAGE_ST2", lambda x: x["EMPFT2"] + x["NMGRS2"]),
        ],
        start=1,
    ):
        for st, label in [(1, "NJ"), (0, "PA")]:
            sub = df[df["STATE"] == st]
            fte_m, fte_se, fte_n = mean_se(sub[fte])
            w_m, w_se, w_n = mean_se(sub[wage])
            # percentage full-time = FT / FTE
            ft = pct_full(sub)
            pct = 100 * ft / (sub[fte] + 0.5 * sub[{"fte1": "EMPPT", "fte2": "EMPPT2"}[fte]])
            pct_m, pct_se, _ = mean_se(pct)
            rows.append(
                {
                    "wave": wave,
                    "state": label,
                    "FTE": f"{fmt(fte_m)} ({fmt(fte_se)})",
                    "FTE_n": fte_n,
                    "starting_wage": f"{fmt(w_m)} ({fmt(w_se)})",
                    "wage_n": w_n,
                    "pct_full_time": f"{fmt(pct_m, 1)} ({fmt(pct_se, 1)})",
                }
            )
    return pd.DataFrame(rows)


t2 = table2()
t2.to_csv(OUT_TABLES / "table2_means.csv", index=False)

# --------------------------------------------------------------------------- #
# 4. Table 3 — average employment per store, before and after (DID)
# --------------------------------------------------------------------------- #
def table3():
    def group_stats(mask_fte1, mask_fte2):
        """Return dict of stats for a store group (NJ, PA, or NJ wage band)."""
        fte1_m, fte1_se, fte1_n = mean_se(df.loc[mask_fte1, "fte1"])
        fte2_m, fte2_se, fte2_n = mean_se(df.loc[mask_fte2, "fte2"])
        # row 3: change in means (independent means)
        chg = fte2_m - fte1_m
        chg_se = np.sqrt(fte1_se**2 + fte2_se**2)
        # row 4: balanced sample
        bal = df[mask_fte1 & mask_fte2]
        bm, bse, bn = mean_se(bal["dfte"])
        # row 5: temporarily closed treated as zero
        return {
            "fte1": fte1_m, "fte1_se": fte1_se, "fte1_n": fte1_n,
            "fte2": fte2_m, "fte2_se": fte2_se, "fte2_n": fte2_n,
            "chg": chg, "chg_se": chg_se,
            "bal": bm, "bal_se": bse, "bal_n": bn,
        }

    def bands():
        """NJ stores classified by wave-1 starting wage."""
        nj = df["STATE"] == 1
        w = df["WAGE_ST"]
        return {
            "low": (nj & (w == 4.25)),
            "mid": (nj & (w > 4.25) & (w < 5.00)),
            "high": (nj & (w >= 5.00)),
        }

    b = bands()
    stats = {}
    stats["PA"] = group_stats(df["STATE"] == 0, df["STATE"] == 0)
    stats["NJ"] = group_stats(df["STATE"] == 1, df["STATE"] == 1)
    for k, mask in b.items():
        stats[k] = group_stats(mask, mask)

    # assemble a tidy table
    out = []
    for key, nm in [("PA", "Pennsylvania"), ("NJ", "New Jersey")]:
        s = stats[key]
        out.append(
            {
                "group": nm, "n_before": s["fte1_n"], "n_after": s["fte2_n"],
                "FTE_before": f"{fmt(s['fte1'])} ({fmt(s['fte1_se'])})",
                "FTE_after": f"{fmt(s['fte2'])} ({fmt(s['fte2_se'])})",
                "change_all": f"{fmt(s['chg'])} ({fmt(s['chg_se'])})",
                "change_balanced": f"{fmt(s['bal'])} ({fmt(s['bal_se'])})",
                "n_balanced": s["bal_n"],
            }
        )
    did, did_se, did_t = did_pair(df[df["STATE"] == 1]["dfte"], df[df["STATE"] == 0]["dfte"])
    out.append(
        {
            "group": "Difference-in-differences", "n_before": "", "n_after": "",
            "FTE_before": "", "FTE_after": "",
            "change_all": "",
            "change_balanced": f"{fmt(did)} ({fmt(did_se)})  [t={fmt(did_t)}]",
            "n_balanced": "",
        }
    )
    # within-NJ wage bands
    for k, nm in [("low", "NJ, wage=$4.25"), ("mid", "NJ, $4.26-$4.99"), ("high", "NJ, >=$5.00")]:
        s = stats[k]
        out.append(
            {
                "group": nm, "n_before": s["fte1_n"], "n_after": s["fte2_n"],
                "FTE_before": f"{fmt(s['fte1'])} ({fmt(s['fte1_se'])})",
                "FTE_after": f"{fmt(s['fte2'])} ({fmt(s['fte2_se'])})",
                "change_all": f"{fmt(s['chg'])} ({fmt(s['chg_se'])})",
                "change_balanced": f"{fmt(s['bal'])} ({fmt(s['bal_se'])})",
                "n_balanced": s["bal_n"],
            }
        )
    return pd.DataFrame(out)


t3 = table3()
t3.to_csv(OUT_TABLES / "table3_did.csv", index=False)

# --------------------------------------------------------------------------- #
# 5. Table 4 — reduced-form models for the change in FTE employment
# --------------------------------------------------------------------------- #
y = t4["dfte"]
controls = ["CO_OWNED", "chain_2", "chain_3", "chain_4"]
region = ["NORTHJ", "CENTRALJ", "PA1", "PA2"]

specs = [
    ("(i)", ["NJ"], "New Jersey dummy, no controls"),
    ("(ii)", ["NJ"] + controls, "New Jersey dummy + chain/ownership controls"),
    ("(iii)", ["gap"], "Initial wage gap, no controls"),
    ("(iv)", ["gap"] + controls, "Initial wage gap + chain/ownership controls"),
    ("(v)", ["gap"] + controls + region, "Initial wage gap + controls + region"),
]

t4_rows = []
for tag, cols, desc in specs:
    X = t4[cols].astype(float)
    r = ols_row(y, X, desc)
    coef = r["coef"]
    se = r["se"]
    key = "NJ" if "NJ" in cols else "gap"
    t4_rows.append(
        {
            "column": tag,
            "key_coef": f"{fmt(coef[key])} ({fmt(se[key])})",
            "description": desc,
        }
    )
t4_out = pd.DataFrame(t4_rows)
t4_out.to_csv(OUT_TABLES / "table4_regressions.csv", index=False)

# Full coefficient detail (for the report appendix).
detail_rows = []
for tag, cols, desc in specs:
    X = t4[cols].astype(float)
    r = ols_row(y, X, desc)
    for name in r["coef"].index:
        detail_rows.append(
            {
                "column": tag,
                "variable": name,
                "coef": round(r["coef"][name], 4),
                "se": round(r["se"][name], 4),
                "t": round(r["coef"][name] / r["se"][name], 3),
            }
        )
    detail_rows.append({"column": tag, "variable": "R2", "coef": round(r["r2"], 4), "se": "", "t": ""})
pd.DataFrame(detail_rows).to_csv(OUT_TABLES / "table4_detail.csv", index=False)

# First-stage check: starting-wage DID (validates the treatment).
wage_did, wage_did_se, wage_did_t = did_pair(
    df[df["STATE"] == 1]["dwage"], df[df["STATE"] == 0]["dwage"]
)

# --------------------------------------------------------------------------- #
# 6. Figure 1 — distribution of starting wages, by state and wave
# --------------------------------------------------------------------------- #
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# wage bins in 5-cent increments over the observed support
bins = np.arange(4.20, 5.90, 0.05)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
for ax, wave, wage_col, title in [
    (axes[0], "Feb–Mar 1992", "WAGE_ST", "Before the increase"),
    (axes[1], "Nov–Dec 1992", "WAGE_ST2", "After the increase"),
]:
    nj = df.loc[df["STATE"] == 1, wage_col].dropna()
    pa = df.loc[df["STATE"] == 0, wage_col].dropna()
    ax.hist(nj, bins=bins, color=NJ_COLOR, alpha=0.55, label="New Jersey", edgecolor="white", lw=0.5)
    ax.hist(pa, bins=bins, color=PA_COLOR, alpha=0.55, label="Pennsylvania", edgecolor="white", lw=0.5)
    ax.axvline(4.25, color=INK, lw=1, ls="--", alpha=0.5)
    ax.axvline(5.05, color=NJ_COLOR, lw=1, ls="--", alpha=0.7)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
    ax.set_xlabel("Starting wage (dollars per hour)", fontsize=9, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)

axes[0].set_ylabel("Number of restaurants", fontsize=9, color=INK)
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
# annotate the two policy-relevant wage levels
axes[0].text(4.25, axes[0].get_ylim()[1] * 0.92, "$4.25", fontsize=8, color=INK, ha="center")
axes[1].text(5.05, axes[1].get_ylim()[1] * 0.92, "$5.05", fontsize=8, color=NJ_COLOR, ha="center")

fig.suptitle(
    "Figure 1 — Distribution of starting wages in New Jersey and Pennsylvania",
    fontsize=12, color=INK, x=0.01, ha="left", fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT_FIGURES / "figure1_wage_distribution.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------- #
# 7. Extension — chain heterogeneity in the employment response
# --------------------------------------------------------------------------- #
# Economic question: do the four chains respond differently to the minimum-wage
# increase?  The chains differ in menu, store size, and production technology,
# so a heterogeneous response informs the mechanism behind the average effect.

# (a) Descriptive: chain composition and average initial wage gap.
chain_desc = (
    t4.groupby("CHAIN")
    .agg(
        n=("SHEET", "size"),
        n_NJ=("NJ", "sum"),
        mean_gap=("gap", "mean"),
        mean_fte1=("fte1", "mean"),
    )
    .reset_index()
)
chain_desc["n_NJ"] = chain_desc["n_NJ"].astype(int)

# (b) DID estimated separately within each chain (with company-ownership control).
ext_rows = []
for c in sorted(t4["CHAIN"].unique()):
    sub = t4[t4["CHAIN"] == c]
    cols = ["NJ"] + (["CO_OWNED"] if sub["CO_OWNED"].nunique() > 1 else [])
    r = ols_row(sub["dfte"], sub[cols].astype(float), f"chain {c}")
    ext_rows.append(
        {
            "chain": int(c),
            "n": r["n"],
            "n_NJ": int(sub["NJ"].sum()),
            "DID": round(r["coef"]["NJ"], 3),
            "SE": round(r["se"]["NJ"], 3),
            "t": round(r["coef"]["NJ"] / r["se"]["NJ"], 3),
        }
    )
ext_by_chain = pd.DataFrame(ext_rows).sort_values("chain")

# (c) Pooled interaction model: DID + chain x NJ interactions.
X_int = pd.concat(
    [
        t4[["NJ", "CO_OWNED", "chain_2", "chain_3", "chain_4"]].astype(float),
        pd.DataFrame(
            t4["NJ"].values[:, None] * t4[["chain_2", "chain_3", "chain_4"]].values,
            columns=["NJ_x_chain2", "NJ_x_chain3", "NJ_x_chain4"],
            index=t4.index,
        ),
    ],
    axis=1,
)
res_int = sm.OLS(t4["dfte"], sm.add_constant(X_int)).fit()

# F-test of joint significance of the three interaction terms.
hyp = ["NJ_x_chain2 = 0", "NJ_x_chain3 = 0", "NJ_x_chain4 = 0"]
f_test = res_int.f_test(hyp)
fval = float(f_test.fvalue[0, 0]) if hasattr(f_test.fvalue, "shape") else float(f_test.fvalue)
fpval = float(f_test.pvalue)

# (d) GAP x chain interaction (mechanism: does the wage-gap response vary?).
X_gapint = pd.concat(
    [
        t4[["gap", "CO_OWNED", "chain_2", "chain_3", "chain_4"]].astype(float),
        pd.DataFrame(
            t4["gap"].values[:, None] * t4[["chain_2", "chain_3", "chain_4"]].values,
            columns=["gap_x_chain2", "gap_x_chain3", "gap_x_chain4"],
            index=t4.index,
        ),
    ],
    axis=1,
)
res_gapint = sm.OLS(t4["dfte"], sm.add_constant(X_gapint)).fit()

ext_by_chain.to_csv(OUT_TABLES / "extension_chain_did.csv", index=False)
chain_desc.to_csv(OUT_TABLES / "extension_chain_descriptives.csv", index=False)

# --------------------------------------------------------------------------- #
# 8. Write a consolidated results summary (markdown) for the report
# --------------------------------------------------------------------------- #
CHAIN_NAMES = {1: "Burger King", 2: "KFC", 3: "Roy Rogers", 4: "Wendy's"}

with open(OUT_TABLES / "results_summary.md", "w") as f:
    f.write("# Card & Krueger (1994) — replication results\n\n")
    f.write("Generated by `analysis/ck1994_replication.py`.\n\n")

    f.write("## First stage (starting wage, difference-in-differences)\n\n")
    w_nj, w_nj_se, _ = mean_se(df[df["STATE"] == 1]["dwage"])
    w_pa, w_pa_se, _ = mean_se(df[df["STATE"] == 0]["dwage"])
    f.write(
        f"- NJ starting-wage change = {fmt(w_nj)} (SE {fmt(w_nj_se)}); "
        f"PA = {fmt(w_pa)} (SE {fmt(w_pa_se)}).\n"
    )
    f.write(
        f"- Wage DID = {fmt(wage_did)} (SE {fmt(wage_did_se)}), t = {fmt(wage_did_t)}. "
        f"(Paper: +0.48.)\n\n"
    )

    f.write("## Table 4 — reduced-form models for change in FTE employment\n\n")
    f.write("| Column | Key coefficient (SE) | Specification |\n")
    f.write("|---|---|---|\n")
    for _, r in t4_out.iterrows():
        f.write(f"| {r['column']} | {r['key_coef']} | {r['description']} |\n")
    f.write(
        f"\nSample: {len(t4)} stores; dependent-variable mean = "
        f"{fmt(y.mean())}, SD = {fmt(y.std(), 3)}.\n"
        f"(Paper: 357 stores; mean -0.237, SD 8.825.)\n\n"
    )

    f.write("## Extension — chain heterogeneity\n\n")
    f.write("| Chain | N | N in NJ | DID (SE) | t |\n")
    f.write("|---|---|---|---|---|\n")
    for _, r in ext_by_chain.iterrows():
        f.write(
            f"| {CHAIN_NAMES[r['chain']]} | {r['n']:.0f} | {r['n_NJ']:.0f} "
            f"| {r['DID']:.3f} ({r['SE']:.3f}) | {r['t']:.2f} |\n"
        )
    f.write(
        f"\nInteraction-model F-test (chain x NJ jointly): F = {fval:.2f}, "
        f"p = {fpval:.3f}.\n"
    )
    for v in ["NJ_x_chain2", "NJ_x_chain3", "NJ_x_chain4"]:
        f.write(
            f"- {v}: {res_int.params[v]:.3f} (SE {res_int.bse[v]:.3f}), "
            f"t = {res_int.params[v]/res_int.bse[v]:.2f}\n"
        )
    f.write("\nGAP x chain interaction model:\n")
    for v in ["gap_x_chain2", "gap_x_chain3", "gap_x_chain4"]:
        f.write(
            f"- {v}: {res_gapint.params[v]:.3f} (SE {res_gapint.bse[v]:.3f}), "
            f"t = {res_gapint.params[v]/res_gapint.bse[v]:.2f}\n"
        )

# --------------------------------------------------------------------------- #
# 9. Console summary
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=" * 72)
    print("Card & Krueger (1994) replication — completed")
    print("=" * 72)
    print(f"\nTable 2 (means)          -> {len(t2)} rows  -> output/tables/table2_means.csv")
    print(f"Table 3 (DID)            -> {len(t3)} rows  -> output/tables/table3_did.csv")
    print(f"Table 4 (regressions)    -> {len(t4_out)} cols -> output/tables/table4_regressions.csv")
    print(f"Figure 1 (wages)         -> output/figures/figure1_wage_distribution.png")
    print(f"Extension (chain)        -> output/tables/extension_chain_did.csv")
    print(f"Processed data           -> output/processed/ck1994_analysis.csv")
    print()
    print("Key DID (balanced):")
    did, did_se, did_t = did_pair(df[df["STATE"] == 1]["dfte"], df[df["STATE"] == 0]["dfte"])
    print(f"  FTE change NJ - PA = {did:.3f}  (SE {did_se:.3f}, t = {did_t:.3f})   [paper: 2.76, t = 2.03]")
    print(f"  Wage DID           = {wage_did:.3f}  (SE {wage_did_se:.3f}, t = {wage_did_t:.3f})   [paper: +0.48]")
    print(f"  Table 4 sample n   = {len(t4)}   [paper: 357]")
    print(f"  Chain interaction F = {fval:.2f} (p = {fpval:.3f})")
