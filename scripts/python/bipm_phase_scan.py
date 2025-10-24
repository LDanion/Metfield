#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Auteur: L. Danion — ORCID 0009-0008-8733-8261

python bipm_phase_scan.py \
  --input runs/bipm/utc9098_parsed_long.csv \
  --out-prefix runs/bipm/bipm9098_phase_scan \
  --pmin 1500 --pmax 3000 --step 5 \
  --min-points 20 --min-labs 15 \
  --plot

bipm_phase_scan.py — Scan de cohérence de phase (Rayleigh) sur un réseau d'horloges.

Entrée attendue : un CSV "long" avec au moins ces colonnes :
    - mjd (float) : date en Modified Julian Day
    - lab (str)   : code laboratoire (ex. "UT", "NML", ...)
    - offset_us (float) ou offset_ns (float) : UTC - UTC(k)

Le script :
  1) charge et unifie les unités (μs si possible, sinon ns -> μs),
  2) regroupe par labo, filtre par nombre minimum de points,
  3) standardise chaque série (z-score) pour comparabilité,
  4) pour chaque période du scan (Pmin..Pmax), ajuste y ≈ a cos(ωt)+b sin(ωt)
     et extrait la phase φ_lab = atan2(b, a),
  5) calcule la cohérence Rayleigh R = |(1/N) Σ exp(i φ_lab)| et sa p-value,
  6) sauvegarde :
       - un CSV des phases à la meilleure période,
       - un JSON récapitulatif,
       - des figures (R vs période, et rose des phases) si --plot.

Dépendances : numpy, pandas, matplotlib (optionnel), scipy (optionnel pour stats),
              aucune dépendance Astropy (on fait un fit sinus direct).
"""

import argparse, json, math, os
import numpy as np
import pandas as pd

# ---------- utilitaires ----------

def zscore(x):
    x = np.asarray(x, float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s == 0.0:
        return x * 0.0
    return (x - m) / s

def fit_cos_sin(t, y, w=None, omega=None):
    """
    Ajuste y ~ a cos(ωt) + b sin(ωt) (moindres carrés).
    Retourne (a, b, amp, phase_rad).
    phase = atan2(b, a) (convention cos/sin).
    """
    c = np.cos(omega * t)
    s = np.sin(omega * t)
    if w is None:
        X = np.column_stack([c, s])
        # moindres carrés robustes via np.linalg.lstsq
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    else:
        W = np.sqrt(w)[:, None]
        X = np.column_stack([c, s]) * W
        Y = y * W.ravel()
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    a, b = coef
    amp = math.hypot(a, b)
    phase = math.atan2(b, a)
    return a, b, amp, phase

def rayleigh_R_phases(phases):
    """
    phases : array de phases (radians)
    Retourne (R, pvalue).
    R = |mean(exp(iφ))|
    p-value approx. Rayleigh : exp(-N * R^2) pour N >= ~10.
    """
    phases = np.asarray(phases, float)
    phases = phases[np.isfinite(phases)]
    N = len(phases)
    if N == 0:
        return np.nan, np.nan
    C = np.cos(phases).mean()
    S = np.sin(phases).mean()
    R = math.hypot(C, S)
    # Approximation classique (bonne dès N>10)
    p = math.exp(-N * R * R)
    return R, p

# ---------- pipeline ----------

def load_long_csv(path, min_points=20):
    df = pd.read_csv(path)
    cols = set(df.columns.str.lower())
    # colonnes
    if "mjd" not in cols or "lab" not in cols:
        raise RuntimeError("CSV invalide : il faut des colonnes 'mjd' et 'lab'.")
    # valeur : offset_us prioritaire, sinon offset_ns (converti en μs)
    val_col = None
    for c in df.columns:
        cl = c.lower()
        if cl == "offset_us":
            val_col = c
            break
    if val_col is None:
        for c in df.columns:
            if c.lower() == "offset_ns":
                df["offset_us"] = df[c] / 1000.0
                val_col = "offset_us"
                break
    if val_col is None:
        raise RuntimeError("CSV invalide : ni 'offset_us' ni 'offset_ns' trouvé.")

    # nettoyage
    df = df[["mjd", "lab", val_col]].dropna()
    df = df.rename(columns={val_col: "y_us"})
    df["lab"] = df["lab"].astype(str)

    # split par labo + filtrage
    groups = {}
    for lab, g in df.groupby("lab"):
        g = g.sort_values("mjd")
        if len(g) >= min_points:
            groups[lab] = g
    if not groups:
        raise RuntimeError("Aucun labo ne passe le filtre min_points.")
    return groups

def build_common_time(groups):
    """Retourne t0 et un dict lab -> (t_rel, y_z)"""
    # On met t=0 au premier MJD global
    t0 = min(g["mjd"].iloc[0] for g in groups.values())
    out = {}
    for lab, g in groups.items():
        t = g["mjd"].values - t0
        y = g["y_us"].values
        y = zscore(y)
        # retire NaN / inf
        m = np.isfinite(t) & np.isfinite(y)
        t, y = t[m], y[m]
        if len(t) >= 5:
            out[lab] = (t, y)
    return t0, out

def scan_periods(lab_series, pmin, pmax, step, weights="amp", min_labs=10):
    """
    lab_series : dict lab -> (t, y) (z-score)
    Balaye les périodes et calcule R, p.
    weights = 'amp' pour pondérer les phases par l'amplitude (optionnel léger).
    """
    periods = np.arange(pmin, pmax + 1e-9, step, dtype=float)
    results = []  # tuples (period, R, p, N_used)
    per_period_phases = {}  # period -> dict lab -> (phase, amp)

    for P in periods:
        w = 2.0 * math.pi / P
        phases = []
        amps = []
        per_lab = {}
        for lab, (t, y) in lab_series.items():
            if len(t) < 5:
                continue
            try:
                _, _, amp, phi = fit_cos_sin(t, y, omega=w)
            except Exception:
                continue
            per_lab[lab] = (phi, amp)
            phases.append(phi)
            amps.append(amp)

        N = len(phases)
        if N < min_labs:
            results.append((P, np.nan, np.nan, N))
            continue

        phases = np.array(phases)
        amps = np.array(amps)

        if weights == "amp":
            # pondération simple : répéter chaque phase ~amp (limité)
            # mais ici on garde Rayleigh non pondéré pour rester standard.
            pass

        R, p = rayleigh_R_phases(phases)
        results.append((P, R, p, N))
        per_period_phases[P] = per_lab

    res_df = pd.DataFrame(results, columns=["period_days", "rayleigh_R", "rayleigh_p", "N_labs"])
    return res_df, per_period_phases

def save_outputs(out_prefix, res_df, per_period_phases, bestP, lab_series, plot):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True) if os.path.dirname(out_prefix) else None

    # 1) CSV du scan
    scan_csv = f"{out_prefix}_phase_scan.csv"
    res_df.to_csv(scan_csv, index=False)

    # 2) phases à la meilleure période
    phase_rows = []
    for lab, (t, y) in lab_series.items():
        phi, amp = (np.nan, np.nan)
        pl = per_period_phases.get(bestP, {}).get(lab, None)
        if pl is not None:
            phi, amp = pl
        phase_rows.append({"lab": lab, "period_days": bestP, "phase_rad": float(phi), "amplitude_z": float(amp)})
    phase_csv = f"{out_prefix}_phases_best.csv"
    pd.DataFrame(phase_rows).to_csv(phase_csv, index=False)

    # 3) résumé
    best_row = res_df.loc[res_df["period_days"].sub(bestP).abs().idxmin()]
    summary = {
        "best_period_days": float(bestP),
        "best_rayleigh_R": float(best_row["rayleigh_R"]),
        "best_rayleigh_p": float(best_row["rayleigh_p"]),
        "N_labs_at_best": int(best_row["N_labs"]),
        "scan_csv": os.path.basename(scan_csv),
        "phases_csv": os.path.basename(phase_csv),
    }
    with open(f"{out_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    if plot:
        import matplotlib.pyplot as plt

        # (a) R vs période
        plt.figure(figsize=(7,4))
        plt.plot(res_df["period_days"], res_df["rayleigh_R"], lw=2)
        plt.axvline(bestP, ls="--", color="r", label=f"best ~ {bestP:.0f} j")
        plt.xlabel("Période (jours)")
        plt.ylabel("Rayleigh R")
        plt.title("Cohérence de phase (Rayleigh) vs période")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_R_vs_period.png", dpi=160)

        # (b) Rose des phases à la meilleure période
        phases = []
        for lab, (phi, amp) in per_period_phases[bestP].items():
            if np.isfinite(phi):
                phases.append(phi)
        phases = np.array(phases)
        if len(phases) >= 3:
            R, _ = rayleigh_R_phases(phases)
            theta = np.linspace(0, 2*np.pi, 361)
            plt.figure(figsize=(5,5))
            ax = plt.subplot(111, polar=True)
            ax.scatter((phases % (2*np.pi)), np.ones_like(phases), s=20)
            ax.set_title(f"Phases @ P≈{bestP:.0f} j  —  R={R:.2f}", va='bottom')
            # vecteur moyen
            C = np.cos(phases).mean(); S = np.sin(phases).mean()
            mean_phase = math.atan2(S, C); mean_r = math.hypot(C, S)
            ax.plot([mean_phase, mean_phase], [0, 1.05], lw=3, color="r")
            plt.tight_layout()
            plt.savefig(f"{out_prefix}_phases_rose.png", dpi=160)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV 'long' (utc9098_parsed_long.csv, etc.)")
    ap.add_argument("--out-prefix", default="bipm_phase_scan")
    ap.add_argument("--pmin", type=float, default=1500.0)
    ap.add_argument("--pmax", type=float, default=3000.0)
    ap.add_argument("--step", type=float, default=5.0)
    ap.add_argument("--min-points", type=int, default=20, help="points min par labo")
    ap.add_argument("--min-labs", type=int, default=10, help="#labos min pour calculer R/p")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    groups = load_long_csv(args.input, min_points=args.min_points)
    t0, lab_series = build_common_time(groups)
    if len(lab_series) < args.min_labs:
        raise RuntimeError(f"Trop peu de labos après filtrage : {len(lab_series)} < {args.min_labs}")

    res_df, per_period_phases = scan_periods(
        lab_series,
        pmin=args.pmin, pmax=args.pmax, step=args.step,
        min_labs=args.min_labs
    )

    # meilleure période = max R (ignorer NaN)
    valid = res_df.dropna(subset=["rayleigh_R"])
    if valid.empty:
        raise RuntimeError("Aucune période valide n'a produit un R calculable.")
    best_idx = valid["rayleigh_R"].idxmax()
    bestP = float(valid.loc[best_idx, "period_days"])

    save_outputs(args.out_prefix, res_df, per_period_phases, bestP, lab_series, plot=args.plot)

if __name__ == "__main__":
    main()
