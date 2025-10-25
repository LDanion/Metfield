#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Auteur: L. Danion — ORCID 0009-0008-8733-8261

python detect_ponly_enhanced_v2.py \
  --metrics-csv results_japan_full_v3/jp_full_pq_metrics.csv \
  --events-csv data/usgs_japan/usgs_japan_2010_2024_M5.5.csv \
  --out-prefix results_japan_full_v3/ltpq_runs/jp_ltpq_M5p5 \
  --score-pct 88 --w1 1 --w2 0.8 --w3 0.8 --w4 0.8 \
  --lt-modulate-pvar --plot


"""

import argparse, pandas as pd, numpy as np, os, json, matplotlib.pyplot as plt
from datetime import timedelta

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="ΔP-only precursor detection (enhanced v2, robust)")
    p.add_argument("--metrics-csv", required=True)
    p.add_argument("--events-csv", required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--mag-min", type=float, default=6.0)
    p.add_argument("--lead-days", type=int, default=35)
    p.add_argument("--cooldown-days", type=int, default=10)
    p.add_argument("--dp-win", type=int, default=6)
    p.add_argument("--dp-base", type=int, default=90)
    p.add_argument("--dq-win", type=int, default=6)
    p.add_argument("--dq-base", type=int, default=90)
    p.add_argument("--dp-sigma", type=float, default=0.5)
    p.add_argument("--dp-min", type=float, default=0.008)
    p.add_argument("--dp-min-pct", type=float, default=0.75)
    p.add_argument("--dp-pct-win", type=int, default=60)
    p.add_argument("--min-consecutive", type=int, default=2)
    p.add_argument("--gate-coh-pct", type=float, default=0.45)
    p.add_argument("--gate-vel-pct", type=float, default=0.55)
    p.add_argument("--score-pct", type=float, default=85.0)
    p.add_argument("--w1", type=float, default=1.0)
    p.add_argument("--w2", type=float, default=0.8)
    p.add_argument("--w3", type=float, default=0.8)
    p.add_argument("--w4", type=float, default=0.8)
    p.add_argument("--lt-modulate-pvar", action="store_true")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()

# ---------- Utils ----------
def clean_name(c: str) -> str:
    return str(c).replace("\ufeff", "").strip().lower().replace(" ", "")

def smooth(x, n=7):
    if len(x) < 3 or n <= 1:
        return np.asarray(x)
    k = max(1, int(n))
    return np.convolve(np.asarray(x, float), np.ones(k)/k, mode="same")

def ensure_metrics_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalise les noms
    df.columns = [clean_name(c) for c in df.columns]
    # infer 'time' si besoin
    if "time" not in df.columns:
        # prend la 1re colonne qui ressemble à une date
        for c in df.columns:
            try:
                test = pd.to_datetime(df[c], errors="coerce")
                if test.notna().sum() > 0:
                    df = df.rename(columns={c: "time"})
                    break
            except Exception:
                pass
    if "time" not in df.columns:
        raise KeyError(f"Aucune colonne temporelle trouvée. Colonnes={list(df.columns)}")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).reset_index(drop=True)

    # map des alias
    alias = {
        "p": ["p", "P", "p_field", "phi", "ϕ", "phi_rad", "pvalue"],
        "q": ["q", "Q", "q_field", "qvalue"],
        "coh": ["coh", "coherence", "coh_pct", "cohpct"]
    }
    def first_present(keys):
        for k in keys:
            k2 = clean_name(k)
            if k2 in df.columns:
                return k2
        return None

    pcol = first_present(alias["p"])
    if pcol is None:
        raise KeyError(f"Colonne 'p' introuvable dans {path}. Colonnes lues: {list(df.columns)}")
    if pcol != "p":
        df = df.rename(columns={pcol: "p"})

    qcol = first_present(alias["q"])
    if qcol and qcol != "q":
        df = df.rename(columns={qcol: "q"})
    cohcol = first_present(alias["coh"])
    if cohcol and cohcol != "coh":
        df = df.rename(columns={cohcol: "coh"})

    # coerce numériques
    for c in ["p", "q", "coh"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def compute_score(df, w1, w2, w3, w4):
    # garantit colonnes présentes
    if "q" not in df.columns:
        df["q"] = 0.0
    if "coh" not in df.columns:
        df["coh"] = 0.0
    # gradients (équivalent ΔP/ΔQ lissés)
    dp = np.gradient(df["p"].fillna(method="ffill").fillna(method="bfill").astype(float).values)
    dq = np.gradient(df["q"].fillna(0.0).astype(float).values)
    coh = df["coh"].fillna(0.0).values
    return w1*np.abs(dp) + w2*np.abs(dq) + w3*(1.0 - coh) + w4*(df["p"].fillna(0.0).values**2)

# ---------- Core ----------
def detect_precursors(metrics_csv, events_csv, out_prefix, args):
    print(f"[INFO] Reading metrics: {metrics_csv}")
    df = ensure_metrics_df(metrics_csv)

    score = compute_score(df, args.w1, args.w2, args.w3, args.w4)
    if args.lt_modulate_pvar:
        pstd = pd.Series(df["p"]).rolling(20, min_periods=1).std().fillna(0.0).values
        score = score * (1.0 + 0.5*np.abs(pstd))

    df["score"] = smooth(score, n=7)
    thr = np.nanpercentile(df["score"], args.score_pct)
    df["alert"] = df["score"] > thr

    # cooldown
    alerts, last = [], None
    for _, r in df[df["alert"]].iterrows():
        t = r["time"]
        if last is None or (t - last).days > int(args.cooldown_days):
            alerts.append(t)
            last = t
    A = pd.DataFrame({"time": alerts})
    print(f"[INFO] Alert candidates: {len(A)}")

    # events
    E = pd.read_csv(events_csv)
    E.columns = [clean_name(c) for c in E.columns]
    if "time" not in E.columns:
        raise KeyError(f"Catalogue: colonne 'time' absente. Colonnes={list(E.columns)}")
    E["time"] = pd.to_datetime(E["time"], errors="coerce")
    if "mag" not in E.columns and "magnitude" in E.columns:
        E = E.rename(columns={"magnitude": "mag"})
    if "mag" not in E.columns:
        raise KeyError("Catalogue: colonne 'mag' absente.")
    E["mag"] = pd.to_numeric(E["mag"], errors="coerce")
    E = E[(E["mag"] >= float(args.mag_min)) & E["time"].notna()].reset_index(drop=True)

    # matching
    matches = []
    for _, a in A.iterrows():
        w = (E["time"] > a["time"]) & (E["time"] <= a["time"] + timedelta(days=int(args.lead_days)))
        if w.any():
            ev = E.loc[w].iloc[0]
            matches.append((a["time"], ev["time"], ev["mag"]))
    M = pd.DataFrame(matches, columns=["alert_time", "event_time", "mag"])

    # scores
    precision = len(M) / len(A) if len(A) else 0.0
    recall = len(M) / len(E) if len(E) else 0.0
    f1 = (2*precision*recall/(precision+recall)) if (precision+recall) else 0.0

    # save
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    A.to_csv(f"{out_prefix}_alerts.csv", index=False)
    M.to_csv(f"{out_prefix}_matches.csv", index=False)
    summary = {
        "n_alertes": len(A),
        "n_events": len(E),
        "n_matched_alerts": len(M),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "score_percentile": args.score_pct,
        "cooldown_days": args.cooldown_days,
        "weights": {"w1": args.w1, "w2": args.w2, "w3": args.w3, "w4": args.w4},
    }
    with open(f"{out_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] {len(M)} matches | F1={f1:.3f}")
    print(f"[OUT] {out_prefix}_summary.json")

    if args.plot:
        fig, ax1 = plt.subplots(figsize=(11,5))
        ax1.plot(df["time"], df["p"], alpha=0.6, label="P")
        ax1.set_ylabel("P", color="C0")
        ax2 = ax1.twinx()
        ax2.plot(df["time"], df["score"], alpha=0.4, label="score", color="C3")
        ax2.axhline(thr, ls="--", alpha=0.4)
        for t in A["time"]:
            ax1.axvline(t, color="orange", alpha=0.6, lw=0.8)
        for _, e in E.iterrows():
            ax1.axvline(e["time"], color="k", alpha=0.15, lw=0.5)
        ax1.set_title("ΔP precursor detection timeline")
        fig.tight_layout()
        out_fig = f"{out_prefix}_timeline.png"
        plt.savefig(out_fig, dpi=150)
        plt.close(fig)
        print(f"[OK] Figure saved: {out_fig}")

# ---------- Main ----------
def main():
    args = parse_args()
    detect_precursors(args.metrics_csv, args.events_csv, args.out_prefix, args)

if __name__ == "__main__":
    main()
