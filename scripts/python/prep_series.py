# prep_series.py

"""

Auteur: L. Danion — ORCID 0009-0008-8733-8261

"""

import argparse, pandas as pd, numpy as np
from pathlib import Path

def zscore(x):
    mu, sig = np.nanmean(x), np.nanstd(x)
    return (x - mu) / (sig if sig>0 else 1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV d'entrée")
    ap.add_argument("--time-col", default="time", help="colonne temps (datetime ou jours)")
    ap.add_argument("--value-col", required=True, help="colonne valeur Y")
    ap.add_argument("--driver-col", default=None, help="colonne driver S (optionnel)")
    ap.add_argument("--time-as-datetime", action="store_true", help="parsing datetime")
    ap.add_argument("--train-frac", type=float, default=0.7, help="fraction train si pas de date-cut")
    ap.add_argument("--date-cut", default=None, help="YYYY-MM-DD, split temporel (optionnel)")
    ap.add_argument("--out", required=True, help="CSV de sortie prêt pour le fit")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    # temps -> numérique (jours)
    if args.time_as_datetime:
        df[args.time_col] = pd.to_datetime(df[args.time_col])
        t0 = df[args.time_col].iloc[0]
        df["t"] = (df[args.time_col] - t0).dt.total_seconds()/86400.0
    else:
        df["t"] = df[args.time_col].astype(float)

    # valeurs
    df["Y"] = pd.to_numeric(df[args.value_col], errors="coerce")

    # driver optionnel
    if args.driver_col:
        df["S"] = pd.to_numeric(df[args.driver_col], errors="coerce")
    else:
        df["S"] = np.nan

    # tri + dropna partiel
    df = df.sort_values("t").reset_index(drop=True)

    # split
    if args.date_cut and args.time_as_datetime:
        cut = pd.to_datetime(args.date_cut)
        mask_train = df[args.time_col] < cut
    else:
        n = len(df)
        ntr = max(1, int(args.train_frac*n))
        mask_train = np.arange(n) < ntr

    df["split"] = np.where(mask_train, "train", "test")

    # standardisation par split (pour éviter fuite d’info)
    out = []
    for split, g in df.groupby("split"):
        y = zscore(g["Y"].values)
        if g["S"].isna().all():
            s = np.full_like(y, np.nan, dtype=float)
        else:
            s = zscore(g["S"].values)
        gg = pd.DataFrame({"t":g["t"].values, "Y":y, "S":s, "split":split})
        out.append(gg)
    df2 = pd.concat(out, ignore_index=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df2.to_csv(args.out, index=False)
    print(f"✓ préparé → {args.out}  (N={len(df2)}, train/test={dict(df2['split'].value_counts())})")

if __name__ == "__main__":
    main()
