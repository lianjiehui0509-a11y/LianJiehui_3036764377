# Minimum Wages and Employment — Replication of Card & Krueger (1994)

A reproducible, AI-assisted replication of

> Card, David, and Alan B. Krueger (1994). "Minimum Wages and Employment: A
> Case Study of the Fast-Food Industry in New Jersey and Pennsylvania."
> *American Economic Review*, 84(4), 772–793.

Prepared for ECON6067 · Quantitative Tools for Economic Analysis (Individual
Project 01 — Minimum Wage / Difference-in-Differences).

## What is reproduced

| Target | Status |
|---|---|
| Table 2 — means of key variables | ✅ |
| Table 3 — average employment per store (DID) | ✅ |
| Table 4 — reduced-form models for the change in FTE employment | ✅ |
| Figure 1 — distribution of starting wages | ✅ |
| Independent extension — chain heterogeneity | ✅ |

The headline result reproduces the paper's: a difference-in-differences estimate
of **+2.75 FTE employees (t = 2.05)** vs the published **+2.76 (t = 2.03)**, and
Table 4 coefficients matching to two decimals (e.g., New Jersey dummy **2.30,
SE 1.20**).

## Repository structure

```
.
├── analysis/
│   └── ck1994_replication.py   # single script that regenerates everything
├── data/
│   └── public.csv              # NOT committed — see "Data" below
├── output/
│   ├── processed/              # constructed analysis dataset
│   ├── tables/                 # reproduced tables (CSV + a results summary)
│   └── figures/                # Figure 1
├── report/
│   ├── report.md               # empirical report
│   └── ai_use_disclosure.md    # AI-use disclosure
├── DATA_DICTIONARY.md          # column definitions
├── requirements.txt
└── README.md
```

## How to reproduce

### 1. Place the input data

The raw data are **restricted teaching data** and are intentionally **not**
committed to this public repository. Obtain `public.csv` from the course
"Project Data Downloads" page (Package **01 — Minimum wage / DID**), then place
it at:

```
data/public.csv
```

No other manual step is needed: after the file is in place, the script
reconstructs every variable and regenerates every table and figure.

### 2. Install dependencies

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

(`pandas`, `numpy`, `scipy`, `statsmodels`, `matplotlib`.)

### 3. Run the workflow

```bash
python analysis/ck1994_replication.py
```

This writes the processed dataset to `output/processed/`, the reproduced tables
to `output/tables/`, and Figure 1 to `output/figures/`, and prints a console
summary of the key estimates.

## Data access

`public.csv` is the Princeton public replication dataset for Card & Krueger
(1994): 410 fast-food restaurants in New Jersey and eastern Pennsylvania,
surveyed in February–March 1992 and November–December 1992. Column definitions
are in [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md). Per the course data-access
terms, the store-level data are **not** redistributed here: neither the raw
`data/public.csv` nor the store-level processed dataset is committed. The
workflow regenerates the processed dataset locally under `output/processed/`;
only aggregate results (the reproduced tables and Figure 1) are committed.

## Report

The full write-up — research question, data and variable construction, empirical
design, core replication results with a side-by-side comparison to the paper, the
independent extension, and validation/limitations — is in
[`report/report.md`](report/report.md). The AI-use disclosure is in
[`report/ai_use_disclosure.md`](report/ai_use_disclosure.md).
