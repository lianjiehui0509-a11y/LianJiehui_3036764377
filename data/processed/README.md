# Processed data

Running `python code/run_all.py` writes the store-level processed dataset here:

```
data/processed/ck1994_analysis.csv
```

It contains the constructed variables used in the final analysis (FTE
employment in both waves, the employment change, the initial wage gap, wage and
price changes, and chain/region indicators).

Because this file reproduces the restricted store-level raw data, it is kept
out of version control (see `.gitignore`) and is generated locally by the
workflow. Only the aggregate results in `outputs/` are committed.
