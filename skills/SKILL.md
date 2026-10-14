---
name: card-krueger-1994-did-replication
description: Reproduce the Card & Krueger (1994) difference-in-differences analysis of the New Jersey minimum-wage increase (Tables 2–4, Figure 1) plus a chain-heterogeneity extension, from the Princeton public replication data.
---

# Card–Krueger (1994) DID Replication

A reusable, end-to-end workflow that reconstructs the Card & Krueger (1994)
analysis sample and regenerates the assigned tables and figure with a single
command.

## Inputs

- Raw data at `data/raw/public.csv` (the Princeton public replication dataset:
  410 fast-food restaurants, 46 columns, two survey waves). Obtained from the
  course Package 01 download; see `data/raw/README.md`.
- Python 3.10+ with the packages in `requirements.txt`.

## Run

```bash
pip install -r requirements.txt
python code/run_all.py
```

No manual edits to the data or code are required. The script reads
`data/raw/public.csv`, reconstructs all variables, and writes every output.

## What the skill does

1. **Constructs variables** — full-time-equivalent employment
   (`FTE = EMPFT + NMGRS + 0.5 * EMPPT`), the initial wage gap
   (`GAP = (5.05 - WAGE_ST)/WAGE_ST` for NJ stores below $5.05, else 0), the
   full-meal price, and chain/ownership/region dummies.
2. **Codes wave-2 employment** — permanently closed stores set to zero;
   temporarily closed/renovated/refused stores set to missing.
3. **Reproduces** Table 2 (means), Table 3 (DID), Table 4 (reduced-form
   regressions), and Figure 1 (wage distributions), plus supplementary
   first-stage, price (Table 7), and robustness (Table 5) tables.
4. **Runs the extension** — chain-heterogeneity DID and interaction models.

## Outputs

| Location | Content |
|---|---|
| `outputs/tables/` | All reproduced tables (CSV) and a results summary (Markdown) |
| `outputs/figures/figure1_wage_distribution.png` | Figure 1 |
| `data/processed/ck1994_analysis.csv` | Store-level processed dataset (local only, not committed) |

## Verification targets

The workflow is verified against the published numbers: DID = +2.75 (paper
+2.76, t = 2.03); Table 4 columns (i)–(iv) match to two decimals (2.33/1.19,
2.30/1.20, 15.65/6.08, 14.92/6.21); regression sample = 357 stores; first-stage
wage regression coefficient = 1.04 with R² = 0.77 (paper 1.03, 0.75).
