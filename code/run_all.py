#!/usr/bin/env python3
"""
Card & Krueger (1994) — "Minimum Wages and Employment"
Single entry point that reproduces the assigned tables and figure, plus the
independent extension and supplementary analyses.

Reference:
    Card, David, and Alan B. Krueger (1994). "Minimum Wages and Employment:
    A Case Study of the Fast-Food Industry in New Jersey and Pennsylvania."
    American Economic Review, 84(4), 772-793.

Run:
    python code/run_all.py

Requires the raw input at `data/raw/public.csv`. After the raw data are in
place, this script regenerates every table and figure without manual edits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (relative to the repository root, so the script is location-independent)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "public.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_FIGURES = ROOT / "outputs" / "figures"
for d in (DATA_PROCESSED, OUT_TABLES, OUT_FIGURES):
    d.mkdir(parents=True, exist_ok=True)

# Design-system categorical palette (colorblind-safe). NJ = blue, PA = orange.
NJ_COLOR = "#2a78d6"
PA_COLOR = "#eb6834"
INK = "#0b0b0b"
GRID = "#e1e0d9"

CHAIN_NAMES = {1: "Burger King", 2: "KFC", 3: "Roy Rogers", 4: "Wendy's"}


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


def ols(y: pd.Series, X: pd.DataFrame):
    """Fit OLS with a constant; return the fitted result."""
    X = sm.add_constant(X)
    return sm.OLS(y, X).fit()


# --------------------------------------------------------------------------- #
# 1. Load data and construct variables
# --------------------------------------------------------------------------- #
df = pd.read_csv(DATA_RAW)


def build_fte(d, suffix=""):
    """Full-time-equivalent employment = FT (incl. managers) + 0.5 * PT."""
    return d[f"EMPFT{suffix}"] + d[f"NMGRS{suffix}"] + 0.5 * d[f"EMPPT{suffix}"]


# Base FTE (managers counted as full-time, part-time weighted 0.5).
df["fte1"] = build_fte(df, "")
df["fte2"] = build_fte(df, "2")

# Wave-2 employment treatment (paper, Table 3 notes):
#   - permanently closed (STATUS2 == 3)      -> employment = 0
#   - temporarily closed / renovated (2,4,5) and refusals (0) -> missing
df.loc[df["STATUS2"] == 3, "fte2"] = 0.0
df.loc[df["STATUS2"].isin([0, 2, 4, 5]), "fte2"] = np.nan

df["dfte"] = df["fte2"] - df["fte1"]

# Initial wage gap: proportional increase to reach the new $5.05 minimum.
df["gap"] = 0.0
low = (df["STATE"] == 1) & df["WAGE_ST"].notna() & (df["WAGE_ST"] < 5.05)
df.loc[low, "gap"] = (5.05 - df.loc[low, "WAGE_ST"]) / df.loc[low, "WAGE_ST"]

# Wage change (level and proportional) and full-meal price.
df["dwage"] = df["WAGE_ST2"] - df["WAGE_ST"]
df["dwage_pct"] = (df["WAGE_ST2"] - df["WAGE_ST"]) / df["WAGE_ST"]
df["price1"] = df["PSODA"] + df["PFRY"] + df["PENTREE"]
df["price2"] = df["PSODA2"] + df["PFRY2"] + df["PENTREE2"]
df["dlogprice"] = np.log(df["price2"]) - np.log(df["price1"])

# Dummies.
df = pd.concat(
    [df, pd.get_dummies(df["CHAIN"], prefix="chain", drop_first=True).astype(float)],
    axis=1,
)
df["NJ"] = (df["STATE"] == 1).astype(float)

# --------------------------------------------------------------------------- #
# 2. Samples
# --------------------------------------------------------------------------- #
# Regression sample (Table 4): employment both waves, wave-1 wage, and wave-2
# wage (waived for permanently closed stores).
t4 = df[
    df["fte1"].notna()
    & df["fte2"].notna()
    & df["WAGE_ST"].notna()
    & (df["WAGE_ST2"].notna() | (df["STATUS2"] == 3))
].copy()

CONTROLS = ["CO_OWNED", "chain_2", "chain_3", "chain_4"]
REGION = ["NORTHJ", "CENTRALJ", "PA1", "PA2"]

# Save the store-level processed dataset (local only; NOT committed — the raw
# data are restricted).
proc_cols = [
    "SHEET", "CHAIN", "CO_OWNED", "STATE", "NJ", "SOUTHJ", "CENTRALJ", "NORTHJ",
    "PA1", "PA2", "SHORE", "STATUS2", "fte1", "fte2", "dfte", "WAGE_ST",
    "WAGE_ST2", "dwage", "dwage_pct", "gap", "price1", "price2", "dlogprice",
    "EMPFT", "EMPPT", "NMGRS", "EMPFT2", "EMPPT2", "NMGRS2", "HRSOPEN", "HRSOPEN2",
]
df[proc_cols].to_csv(DATA_PROCESSED / "ck1994_analysis.csv", index=False)


# --------------------------------------------------------------------------- #
# 3. Table 2 — means of key variables
# --------------------------------------------------------------------------- #
def table2():
    rows = []
    for wave, fte, wage in [(1, "fte1", "WAGE_ST"), (2, "fte2", "WAGE_ST2")]:
        for st, label in [(1, "NJ"), (0, "PA")]:
            sub = df[df["STATE"] == st]
            fte_m, fte_se, fte_n = mean_se(sub[fte])
            w_m, w_se, w_n = mean_se(sub[wage])
            rows.append(
                {
                    "wave": wave, "state": label,
                    "FTE": f"{fmt(fte_m)} ({fmt(fte_se)})", "FTE_n": fte_n,
                    "starting_wage": f"{fmt(w_m)} ({fmt(w_se)})", "wage_n": w_n,
                }
            )
    return pd.DataFrame(rows)


table2().to_csv(OUT_TABLES / "table2_means.csv", index=False)


# --------------------------------------------------------------------------- #
# 4. Table 3 — average employment per store, before and after (DID)
# --------------------------------------------------------------------------- #
def table3():
    def group_stats(mask):
        fte1_m, fte1_se, fte1_n = mean_se(df.loc[mask, "fte1"])
        fte2_m, fte2_se, fte2_n = mean_se(df.loc[mask, "fte2"])
        chg = fte2_m - fte1_m
        chg_se = np.sqrt(fte1_se**2 + fte2_se**2)
        bal = df[mask & df["fte1"].notna() & df["fte2"].notna()]
        bm, bse, bn = mean_se(bal["dfte"])
        return dict(
            fte1=fte1_m, fte1_se=fte1_se, fte1_n=fte1_n,
            fte2=fte2_m, fte2_se=fte2_se, fte2_n=fte2_n,
            chg=chg, chg_se=chg_se, bal=bm, bal_se=bse, bal_n=bn,
        )

    w = df["WAGE_ST"]
    bands = {
        "low": (df["STATE"] == 1) & (w == 4.25),
        "mid": (df["STATE"] == 1) & (w > 4.25) & (w < 5.00),
        "high": (df["STATE"] == 1) & (w >= 5.00),
    }
    stats = {"PA": group_stats(df["STATE"] == 0), "NJ": group_stats(df["STATE"] == 1)}
    for k, m in bands.items():
        stats[k] = group_stats(m)

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
            "FTE_before": "", "FTE_after": "", "change_all": "",
            "change_balanced": f"{fmt(did)} ({fmt(did_se)})  [t={fmt(did_t)}]",
            "n_balanced": "",
        }
    )
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


table3().to_csv(OUT_TABLES / "table3_did.csv", index=False)


# --------------------------------------------------------------------------- #
# 5. Table 4 — reduced-form models for the change in FTE employment
# --------------------------------------------------------------------------- #
def table4_summary(sample: pd.DataFrame):
    y = sample["dfte"]
    specs = [
        ("(i)", ["NJ"], "NJ dummy, no controls"),
        ("(ii)", ["NJ"] + CONTROLS, "NJ dummy + chain/ownership"),
        ("(iii)", ["gap"], "Wage gap, no controls"),
        ("(iv)", ["gap"] + CONTROLS, "Wage gap + chain/ownership"),
        ("(v)", ["gap"] + CONTROLS + REGION, "Wage gap + controls + region"),
    ]
    rows, detail = [], []
    for tag, cols, desc in specs:
        r = ols(y, sample[cols].astype(float))
        key = "NJ" if "NJ" in cols else "gap"
        rows.append({"column": tag, "key_coef": f"{fmt(r.params[key])} ({fmt(r.bse[key])})", "description": desc})
        for name in r.params.index:
            if name != "const":
                detail.append({"column": tag, "variable": name, "coef": round(r.params[name], 4),
                               "se": round(r.bse[name], 4), "t": round(r.params[name] / r.bse[name], 3)})
    return pd.DataFrame(rows), pd.DataFrame(detail), len(sample), y.mean(), y.std()


t4_summary, t4_detail, n4, m4, sd4 = table4_summary(t4)
t4_summary.to_csv(OUT_TABLES / "table4_regressions.csv", index=False)
t4_detail.to_csv(OUT_TABLES / "table4_detail.csv", index=False)

# --------------------------------------------------------------------------- #
# 6. First-stage wage regression (validates the GAP treatment measure)
# --------------------------------------------------------------------------- #
wage_samp = df[df["dwage_pct"].notna() & df["gap"].notna()]
r_wage = ols(wage_samp["dwage_pct"], wage_samp[["gap"]].astype(float))
wage_stage = pd.DataFrame(
    [{"variable": "gap", "coef": round(r_wage.params["gap"], 4),
      "se": round(r_wage.bse["gap"], 4), "R2": round(r_wage.rsquared, 3),
      "n": int(r_wage.nobs)}]
)
wage_stage.to_csv(OUT_TABLES / "first_stage_wage.csv", index=False)

# --------------------------------------------------------------------------- #
# 7. Table 7 — reduced-form models for the change in log price of a full meal
# --------------------------------------------------------------------------- #
price_samp = df[df[["dlogprice", "WAGE_ST", "WAGE_ST2", "fte1", "fte2"]].notna().all(axis=1)]
price_specs = [
    ("(i)", ["NJ"]),
    ("(ii)", ["NJ"] + CONTROLS),
    ("(iii)", ["gap"]),
    ("(iv)", ["gap"] + CONTROLS),
    ("(v)", ["gap"] + CONTROLS + REGION),
]
price_rows = []
for tag, cols in price_specs:
    r = ols(price_samp["dlogprice"], price_samp[cols].astype(float))
    key = "NJ" if "NJ" in cols else "gap"
    price_rows.append({"column": tag, "key_coef": f"{fmt(r.params[key])} ({fmt(r.bse[key])})"})
pd.DataFrame(price_rows).to_csv(OUT_TABLES / "table7_prices.csv", index=False)


# --------------------------------------------------------------------------- #
# 8. Table 5 — robustness of the employment estimates
# --------------------------------------------------------------------------- #
def robustness():
    rows = []

    def add_row(label, sample):
        r_nj = ols(sample["dfte"], sample[["NJ"] + CONTROLS].astype(float))
        r_gap = ols(sample["dfte"], sample[["gap"] + CONTROLS].astype(float))
        rows.append(
            {
                "specification": label, "n": len(sample),
                "NJ": f"{fmt(r_nj.params['NJ'])} ({fmt(r_nj.bse['NJ'])})",
                "gap": f"{fmt(r_gap.params['gap'])} ({fmt(r_gap.bse['gap'])})",
            }
        )

    add_row("1. Base (Table 4 col. ii/iv)", t4)

    # 2. Treat four temporarily closed stores as permanently closed.
    d2 = df.copy()
    d2.loc[d2["STATUS2"].isin([2, 4, 5]), "fte2"] = 0.0
    d2["dfte"] = d2["fte2"] - d2["fte1"]
    s2 = d2[d2["fte1"].notna() & d2["fte2"].notna() & d2["WAGE_ST"].notna()
            & (d2["WAGE_ST2"].notna() | (d2["STATUS2"] == 3))]
    add_row("2. Temporarily closed stores set to zero", s2)

    # 3. Exclude managers from FTE.
    d3 = df.copy()
    d3["fte1"] = d3["EMPFT"] + 0.5 * d3["EMPPT"]
    d3["fte2"] = d3["EMPFT2"] + 0.5 * d3["EMPPT2"]
    d3.loc[d3["STATUS2"] == 3, "fte2"] = 0.0
    d3.loc[d3["STATUS2"].isin([0, 2, 4, 5]), "fte2"] = np.nan
    d3["dfte"] = d3["fte2"] - d3["fte1"]
    s3 = d3[d3["fte1"].notna() & d3["fte2"].notna() & d3["WAGE_ST"].notna()
            & (d3["WAGE_ST2"].notna() | (d3["STATUS2"] == 3))]
    add_row("3. Exclude managers from FTE", s3)

    # 4-5. Alternative part-time weights.
    for wt, lab in [(0.4, "4. Part-time weighted 0.4"), (0.6, "5. Part-time weighted 0.6")]:
        d = df.copy()
        d["fte1"] = d["EMPFT"] + d["NMGRS"] + wt * d["EMPPT"]
        d["fte2"] = d["EMPFT2"] + d["NMGRS2"] + wt * d["EMPPT2"]
        d.loc[d["STATUS2"] == 3, "fte2"] = 0.0
        d.loc[d["STATUS2"].isin([0, 2, 4, 5]), "fte2"] = np.nan
        d["dfte"] = d["fte2"] - d["fte1"]
        s = d[d["fte1"].notna() & d["fte2"].notna() & d["WAGE_ST"].notna()
              & (d["WAGE_ST2"].notna() | (d["STATUS2"] == 3))]
        add_row(lab, s)

    # 6. Exclude NJ shore stores.
    s6 = t4[t4["SHORE"] == 0]
    add_row("6. Exclude NJ shore stores", s6)

    return pd.DataFrame(rows)


robustness().to_csv(OUT_TABLES / "table5_robustness.csv", index=False)


# --------------------------------------------------------------------------- #
# 9. Figure 1 — distribution of starting wages, by state and wave
# --------------------------------------------------------------------------- #
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

bins = np.arange(4.20, 5.90, 0.05)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
for ax, title, wage_col in [
    (axes[0], "Before the increase (Feb–Mar 1992)", "WAGE_ST"),
    (axes[1], "After the increase (Nov–Dec 1992)", "WAGE_ST2"),
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
# 10. Extension — chain heterogeneity in the employment response
# --------------------------------------------------------------------------- #
chain_desc = (
    t4.groupby("CHAIN")
    .agg(n=("SHEET", "size"), n_NJ=("NJ", "sum"), mean_gap=("gap", "mean"), mean_fte1=("fte1", "mean"))
    .reset_index()
)
chain_desc["n_NJ"] = chain_desc["n_NJ"].astype(int)

ext_rows = []
for c in sorted(t4["CHAIN"].unique()):
    sub = t4[t4["CHAIN"] == c]
    cols = ["NJ"] + (["CO_OWNED"] if sub["CO_OWNED"].nunique() > 1 else [])
    r = ols(sub["dfte"], sub[cols].astype(float))
    ext_rows.append(
        {"chain": int(c), "n": len(sub), "n_NJ": int(sub["NJ"].sum()),
         "DID": round(r.params["NJ"], 3), "SE": round(r.bse["NJ"], 3),
         "t": round(r.params["NJ"] / r.bse["NJ"], 3)}
    )
ext_by_chain = pd.DataFrame(ext_rows).sort_values("chain")

# Pooled interaction model: chain x NJ.
X_int = pd.concat(
    [
        t4[["NJ", "CO_OWNED", "chain_2", "chain_3", "chain_4"]].astype(float),
        pd.DataFrame(
            t4["NJ"].values[:, None] * t4[["chain_2", "chain_3", "chain_4"]].values,
            columns=["NJ_x_chain2", "NJ_x_chain3", "NJ_x_chain4"], index=t4.index,
        ),
    ],
    axis=1,
)
res_int = ols(t4["dfte"], X_int)
f_int = res_int.f_test(["NJ_x_chain2 = 0", "NJ_x_chain3 = 0", "NJ_x_chain4 = 0"])

# GAP x chain interaction.
X_gapint = pd.concat(
    [
        t4[["gap", "CO_OWNED", "chain_2", "chain_3", "chain_4"]].astype(float),
        pd.DataFrame(
            t4["gap"].values[:, None] * t4[["chain_2", "chain_3", "chain_4"]].values,
            columns=["gap_x_chain2", "gap_x_chain3", "gap_x_chain4"], index=t4.index,
        ),
    ],
    axis=1,
)
res_gapint = ols(t4["dfte"], X_gapint)

ext_by_chain.to_csv(OUT_TABLES / "extension_chain_did.csv", index=False)
chain_desc.to_csv(OUT_TABLES / "extension_chain_descriptives.csv", index=False)


# --------------------------------------------------------------------------- #
# 11. Results summary (markdown) for the report
# --------------------------------------------------------------------------- #
did, did_se, did_t = did_pair(df[df["STATE"] == 1]["dfte"], df[df["STATE"] == 0]["dfte"])
wage_did, wage_did_se, wage_did_t = did_pair(df[df["STATE"] == 1]["dwage"], df[df["STATE"] == 0]["dwage"])

with open(OUT_TABLES / "results_summary.md", "w") as f:
    f.write("# Card & Krueger (1994) — replication results\n\n")
    f.write("Generated by `code/run_all.py`.\n\n")

    f.write("## First stage: starting wage\n\n")
    w_nj, w_nj_se, _ = mean_se(df[df["STATE"] == 1]["dwage"])
    w_pa, w_pa_se, _ = mean_se(df[df["STATE"] == 0]["dwage"])
    f.write(f"- NJ wage change = {fmt(w_nj)} (SE {fmt(w_nj_se)}); PA = {fmt(w_pa)} (SE {fmt(w_pa_se)}).\n")
    f.write(f"- Wage DID = {fmt(wage_did)} (SE {fmt(wage_did_se)}), t = {fmt(wage_did_t)}. (Paper: +0.48.)\n")
    f.write(
        f"- Proportional wage change on GAP: coefficient {fmt(r_wage.params['gap'], 3)} "
        f"(R2 = {fmt(r_wage.rsquared, 3)}). (Paper: 1.03, R2 = 0.75.)\n\n"
    )

    f.write("## Table 4 — change in FTE employment\n\n")
    f.write("| Column | Key coefficient (SE) | Specification |\n|---|---|---|\n")
    for _, r in t4_summary.iterrows():
        f.write(f"| {r['column']} | {r['key_coef']} | {r['description']} |\n")
    f.write(f"\nSample: {n4} stores; dep. var. mean {fmt(m4)}, SD {fmt(sd4, 3)}. (Paper: 357; -0.237, 8.825.)\n\n")

    f.write("## Table 7 — change in log price of a full meal\n\n")
    f.write("| Column | Key coefficient (SE) |\n|---|---|\n")
    for _, r in pd.DataFrame(price_rows).iterrows():
        f.write(f"| {r['column']} | {r['key_coef']} |\n")

    f.write("\n## Extension — chain heterogeneity\n\n")
    f.write("| Chain | N | N in NJ | DID (SE) | t |\n|---|---|---|---|---|\n")
    for _, r in ext_by_chain.iterrows():
        f.write(f"| {CHAIN_NAMES[r['chain']]} | {r['n']:.0f} | {r['n_NJ']:.0f} | {r['DID']:.3f} ({r['SE']:.3f}) | {r['t']:.2f} |\n")
    f.write(f"\nInteraction-model F-test (chain x NJ): F = {float(f_int.fvalue):.2f}, p = {float(f_int.pvalue):.3f}.\n")
    for v in ["NJ_x_chain2", "NJ_x_chain3", "NJ_x_chain4"]:
        f.write(f"- {v}: {res_int.params[v]:.3f} (SE {res_int.bse[v]:.3f}), t = {res_int.params[v]/res_int.bse[v]:.2f}\n")
    f.write("\nGAP x chain interaction model:\n")
    for v in ["gap_x_chain2", "gap_x_chain3", "gap_x_chain4"]:
        f.write(f"- {v}: {res_gapint.params[v]:.3f} (SE {res_gapint.bse[v]:.3f}), t = {res_gapint.params[v]/res_gapint.bse[v]:.2f}\n")


# --------------------------------------------------------------------------- #
# 12. Console summary
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=" * 72)
    print("Card & Krueger (1994) replication — completed")
    print("=" * 72)
    print(f"Table 2 (means)        -> outputs/tables/table2_means.csv")
    print(f"Table 3 (DID)          -> outputs/tables/table3_did.csv")
    print(f"Table 4 (regressions)  -> outputs/tables/table4_regressions.csv")
    print(f"Table 5 (robustness)   -> outputs/tables/table5_robustness.csv")
    print(f"Table 7 (prices)       -> outputs/tables/table7_prices.csv")
    print(f"Figure 1 (wages)       -> outputs/figures/figure1_wage_distribution.png")
    print(f"Extension (chain)      -> outputs/tables/extension_chain_did.csv")
    print(f"Processed data         -> data/processed/ck1994_analysis.csv (local only)")
    print()
    print(f"Key DID (balanced): {did:.3f} (SE {did_se:.3f}, t = {did_t:.3f})   [paper: 2.76, t = 2.03]")
    print(f"Wage DID:            {wage_did:.3f} (SE {wage_did_se:.3f}, t = {wage_did_t:.3f})   [paper: +0.48]")
    print(f"Table 4 sample n:    {n4}   [paper: 357]")
    print(f"Wage first stage:    gap coef = {r_wage.params['gap']:.3f} (R2 = {r_wage.rsquared:.3f})   [paper: 1.03, R2 = 0.75]")
    print(f"Chain interaction F: {float(f_int.fvalue):.2f} (p = {float(f_int.pvalue):.3f})")
