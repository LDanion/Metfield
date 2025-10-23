#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Auteur: L. Danion — ORCID 0009-0008-8733-8261

python bipm_pq_common_modulation.py \
  --input runs/bipm/utc9098_parsed_long.csv \
  --time-col mjd \
  --value-col offset_us \
  --lab-col lab \
  --min-n 20 \
  --band 1e-4 1e-1 \
  --out-prefix bipm9098 \
  --plot


bipm_pq_common_modulation.py
--------------------------------
But:
  - Charger des séries UTC-UTC(lab) (BIPM) depuis CSV long/large ou texte brut.
  - Nettoyer, centrer/standardiser, et estimer un spectre par labo (Lomb-Scargle).
  - Agréger un spectre "commun" (médiane robuste à travers les labos).
  - Détecter un pic étroit, mesurer f0, Δf(FWHM), Q = f0/Δf.
  - Sauvegarder: PNG (PSD + zoom), CSV/JSON de synthèse.

Entrées acceptées:
  (A) CSV "long"    : colonnes ≥ [mjd, lab, offset_us|offset_ns]
  (B) CSV "large"   : colonnes [mjd, LAB1, LAB2, ...]
  (C) Texte "brut"  : fichiers BIPM 'utc.9x', 'utcgps9x', 'ta.9x' (tableaux paginés).
"""

import argparse, json, math, re, io, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

# Lomb-Scargle pour échantillonnage irrégulier
try:
    from astropy.timeseries import LombScargle
except Exception as e:
    LombScargle = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# -------- utilitaires --------
MONTHS = {m.lower(): i+1 for i, m in enumerate(
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
)}

def _z(x):
    x = np.asarray(x, float)
    m = np.nanmean(x); s = np.nanstd(x) + 1e-12
    return (x - m) / s

def is_regular(t, tol=1e-6):
    t = np.asarray(t, float)
    dt = np.diff(np.sort(t))
    return np.isfinite(dt).all() and (np.nanstd(dt) / (np.nanmean(dt) + 1e-12) < tol)

def fwhm_from_peak(f, p, ipk):
    """FWHM simple autour d’un pic ipk sur (f, p)."""
    half = p[ipk] / 2.0
    # gauche
    i = ipk
    while i > 0 and p[i] > half:
        i -= 1
    f1 = np.interp(half, [p[i], p[i+1]], [f[i], f[i+1]]) if i < ipk else f[i]
    # droite
    j = ipk
    while j < len(p) - 1 and p[j] > half:
        j += 1
    f2 = np.interp(half, [p[j-1], p[j]], [f[j-1], f[j]]) if j > ipk else f[j]
    return abs(f2 - f1), f1, f2

# -------- chargeurs --------
def load_csv_long(path, time_col, value_col, lab_col):
    df = pd.read_csv(path)
    for c in (time_col, value_col, lab_col):
        if c not in df.columns:
            raise RuntimeError(f"Colonne manquante '{c}' dans {path}")
    df = df[[time_col, value_col, lab_col]].dropna().copy()
    df[time_col]  = pd.to_numeric(df[time_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna()
    series = {}
    for lab, g in df.groupby(lab_col):
        t = g[time_col].values
        y = g[value_col].values
        if len(t) >= 2:
            series[str(lab)] = (t, y)
    return series

def load_csv_wide(path, time_col):
    df = pd.read_csv(path)
    if time_col not in df.columns:
        raise RuntimeError(f"Colonne temps '{time_col}' absente de {path}")
    labs = [c for c in df.columns if c != time_col]
    series = {}
    for lab in labs:
        g = df[[time_col, lab]].dropna()
        t = pd.to_numeric(g[time_col], errors="coerce").values
        y = pd.to_numeric(g[lab], errors="coerce").values
        m = np.isfinite(t) & np.isfinite(y)
        if m.sum() >= 2:
            series[str(lab)] = (t[m], y[m])
    return series

def _parse_table_block(lines, header_idx, labs):
    """
    Parsage d’un bloc tabulaire BIPM:
    - Cherche MJD en 2e colonne.
    - Colonnes suivantes: valeurs pour chaque lab de `labs`.
    """
    data = []
    for k in range(header_idx+1, len(lines)):
        ln = lines[k].rstrip()
        if not ln.strip():
            break  # fin de bloc
        # split tolérant: espaces multiples
        parts = re.split(r"\s+", ln.strip())
        if len(parts) < 2:
            continue
        # MJD en deuxième colonne typiquement
        # Ex: "Jan  6 50819  ...", "Feb 15 50859 ..."
        # On essaye: si parts[1] ou parts[2] est un MJD
        mjd = None
        for j in range(1, min(3, len(parts))):
            if re.fullmatch(r"\d{5}", parts[j]):
                mjd = int(parts[j]); mjd_pos = j; break
        if mjd is None:
            continue
        # Les valeurs suivent le MJD, dans l'ordre des labs
        vals_str = parts[mjd_pos+1:]
        # On tolère qu'il y ait moins/plus de colonnes -> tronque/pad
        vals = []
        for j in range(len(labs)):
            v = np.nan
            if j < len(vals_str):
                try:
                    v = float(vals_str[j])
                except Exception:
                    v = np.nan
            vals.append(v)
        data.append((mjd, vals))
    if not data:
        return None
    mjd_arr = np.array([d[0] for d in data], float)
    vals_m = np.array([d[1] for d in data], float)  # shape: (nrow, nlab)
    return mjd_arr, vals_m

def load_bipm_text(path):
    """
    Parser minimal pour les tables 'UTC-UTC(k)' dans les fichiers BIPM (paginés).
    Retourne un dict {lab: (mjd[], offset_ns[])}.
    """
    text = Path(path).read_text(errors="ignore")
    lines = text.splitlines()
    # 1) détecter la ligne d’entête listant les labos:
    #    'UTC − UTC(k)' suivi de colonnes 'AOS  APL  AUS ...'
    header_idx = None
    labs = None
    for i, ln in enumerate(lines):
        if re.search(r"UTC\s*-\s*UTC\(k\)", ln):
            # les noms de labos sont généralement sur la/les lignes suivantes
            # on scanne quelques lignes pour extraire des sigles (AOS, APL, ...)
            lab_line = ""
            for j in range(i+1, min(i+6, len(lines))):
                lab_line += " " + lines[j]
            # sigles = mots majuscules de 2–6 lettres
            labs = re.findall(r"\b[A-Z]{2,6}\b", lab_line)
            # nettoyer artifices évidents
            labs = [x for x in labs if x not in {"UTC","MJD","UNIT","DATE"}]
            if len(labs) >= 3:
                header_idx = i+1
                break
    if header_idx is None or labs is None:
        raise RuntimeError("Impossible de trouver l'en-tête 'UTC-UTC(k)' et la liste des labos.")
    # 2) parser bloc(s) successifs
    mjd_all = []
    val_all = []
    i = header_idx
    while i < len(lines):
        parsed = _parse_table_block(lines, i, labs)
        if parsed is None:
            i += 1
            continue
        mjd_arr, vals_m = parsed
        mjd_all.append(mjd_arr)
        val_all.append(vals_m)
        # avancer jusqu'à fin de bloc détectée dans _parse_table_block
        # ici: on incrémente prudemment d'une dizaine de lignes
        i += 10
    if not mjd_all:
        raise RuntimeError("Aucun tableau 'UTC-UTC(k)' exploitable trouvé.")

    mjd_cat = np.concatenate(mjd_all)
    vals_cat = np.vstack(val_all)  # (nrow_total, nlab)

    series = {}
    for j, lab in enumerate(labs):
        y = vals_cat[:, j].astype(float)  # ns (d'après les tables)
        m = np.isfinite(mjd_cat) & np.isfinite(y)
        if m.sum() >= 2:
            series[lab] = (mjd_cat[m], y[m])  # temps: MJD, valeur: ns
    return series

# -------- spectre & détection --------
def spectrum_ls(t_days, y_value, fmin=None, fmax=None, nfreq=5000):
    if LombScargle is None:
        raise RuntimeError("Astropy requis: pip install astropy")
    t = np.asarray(t_days, float)
    y = _z(y_value)  # normalise
    # grille de fréquences en 1/jour
    T = t.max() - t.min()
    if T <= 0:
        raise RuntimeError("Fenêtre temporelle nulle.")
    if fmin is None: fmin = 1.0/(T*10.0)
    if fmax is None:
        dt_min = np.median(np.diff(np.unique(t)))
        fmax = 0.5 / max(dt_min, 1e-3)
    f = np.logspace(np.log10(max(fmin, 1e-6)), np.log10(fmax), nfreq)
    ls = LombScargle(t, y, normalization="psd")
    P = ls.power(f)
    return f, P

def detect_peak(f, P, smooth=5, band=None):
    P_s = gaussian_filter1d(P, sigma=smooth) if smooth>0 else P.copy()
    if band is not None:
        fmin, fmax = band
        m = (f>=fmin) & (f<=fmax)
        f_b, P_b = f[m], P_s[m]
    else:
        f_b, P_b = f[1:-1], P_s[1:-1]
    if len(f_b) < 10:
        raise RuntimeError("Bande trop étroite/vidée.")
    med = np.median(P_b); mad = np.median(np.abs(P_b-med)) + 1e-15
    peaks, info = find_peaks(P_b, height=med+5*mad, distance=5)
    if len(peaks)==0:
        peaks, info = find_peaks(P_b, height=med+3*mad, distance=5)
    if len(peaks)==0:
        ipk_b = int(np.argmax(P_b))
        f0 = float(f_b[ipk_b]); note = "Pic faible; pris au max(P)."
    else:
        ipk_b = int(peaks[np.argmax(P_b[peaks])])
        f0 = float(f_b[ipk_b]); note = "Pic détecté."
    # FWHM
    try:
        df, f1, f2 = fwhm_from_peak(f_b, P_b, ipk_b)
        Q = float(f0 / max(df, 1e-12))
    except Exception:
        df, f1, f2, Q = float("nan"), float("nan"), float("nan"), float("nan")
        note += " FWHM non résolue."
    return dict(f=f, P=P, P_s=P_s, f0=f0, df=df, Q=Q, f1=f1, f2=f2, note=note)

# -------- pipeline principal --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Fichier CSV long/large ou texte BIPM")
    ap.add_argument("--time-col", default="mjd", help="Nom colonne temps (CSV)")
    ap.add_argument("--value-col", default=None, help="Nom colonne valeur (CSV long)")
    ap.add_argument("--lab-col", default="lab", help="Nom colonne labo (CSV long)")
    ap.add_argument("--min-n", type=int, default=20, help="Min points/lab")
    ap.add_argument("--band", nargs=2, type=float, default=None,
                    help="Bande de recherche [fmin fmax] en 1/jour")
    ap.add_argument("--smooth", type=int, default=5, help="Lissage gaussien (points)")
    ap.add_argument("--out-prefix", default="bipm_out")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(path)

    # 1) lecture
    series = {}
    try:
        # tentons CSV long
        if args.value_col is None:
            raise RuntimeError  # force test suivant si valeur non donnée
        series = load_csv_long(path, args.time_col, args.value_col, args.lab_col)
        src_kind = "csv_long"
    except Exception:
        try:
            # CSV large
            series = load_csv_wide(path, args.time_col)
            src_kind = "csv_wide"
        except Exception:
            # texte brut BIPM
            series = load_bipm_text(path)
            src_kind = "bipm_text"

    # option: conversion d’unités -> secondes (pour l’échelle PSD, optionnel)
    # Ici on laisse les offsets en ns/us, car le LS normalise.

    # 2) filtrage par densité
    series = {lab:(t,y) for lab,(t,y) in series.items() if len(t)>=args.min_n}
    if not series:
        print("[Diagnostique] Aucun laboratoire n'a passé les filtres.")
        return

    labs = sorted(series.keys())
    print(pd.Series({lab: len(series[lab][0]) for lab in labs}, name="npts"))

    # 3) spectres individuels (Lomb-Scargle) puis médiane robuste
    if LombScargle is None:
        raise RuntimeError("Astropy requis: pip install astropy")
    # grille commune (en 1/jour)
    # On la construit depuis la première série
    t0, _ = next(iter(series.values()))
    T = max(t0) - min(t0)
    fmin = args.band[0] if args.band else 1.0/(T*10.0)
    # fmax heuristique depuis une médiane de pas min sur tous labos
    dt_all = []
    for lab in labs:
        t,_ = series[lab]
        uu = np.diff(np.unique(np.sort(t)))
        if len(uu)>0: dt_all.append(np.median(uu))
    dt_med = np.median(dt_all) if dt_all else 1.0
    fmax = args.band[1] if args.band else 0.5/max(dt_med, 1.0)

    f = np.logspace(np.log10(max(fmin,1e-6)), np.log10(fmax), 5000)
    P_list = []
    for lab in labs:
        t,y = series[lab]
        y = _z(y)
        ls = LombScargle(t, y, normalization="psd")
        P = ls.power(f)
        P_list.append(P)
    P_stack = np.vstack(P_list)  # (nlab, nfreq)
    P_med = np.median(P_stack, axis=0)

    # 4) détection du pic commun
    det = detect_peak(f, P_med, smooth=args.smooth, band=args.band)

    # 5) sorties
    out = {
        "source_kind": src_kind,
        "input": str(path),
        "n_labs": len(labs),
        "labs": labs,
        "band": list(args.band) if args.band else None,
        "f0_per_day": det["f0"],                    # fréquence en 1/jour
        "period_days": (1.0/det["f0"]) if det["f0"]>0 else float("nan"),
        "df_fwhm": det["df"],
        "Q": det["Q"],
        "f1": det["f1"], "f2": det["f2"],
        "note": det["note"]
    }
    # JSON
    with open(f"{args.out_prefix}_summary.json","w") as fjs:
        json.dump(out, fjs, indent=2)
    # CSV synthèse
    pd.DataFrame([out]).to_csv(f"{args.out_prefix}_summary.csv", index=False)

    print(json.dumps(out, indent=2))

    # 6) figures optionnelles
    if args.plot:
        plt.figure(figsize=(8,4))
        for k in range(min(10, len(labs))):  # surcharger l'affichage si beaucoup de labos
            plt.loglog(f, P_stack[k], alpha=0.15)
        plt.loglog(f, P_med, lw=2, label="médiane (tous labos)")
        plt.axvline(det["f0"], color="r", ls="--",
                    label=f"f0={det['f0']:.4g} (P≈{1/det['f0']:.2f} j)")
        if np.isfinite(det["df"]):
            plt.axvline(det["f1"], color="k", ls=":")
            plt.axvline(det["f2"], color="k", ls=":")
        if args.band:
            plt.axvspan(args.band[0], args.band[1], color="grey", alpha=0.08, label="bande")
        plt.xlabel("fréquence (1/jour)")
        plt.ylabel("puissance (LS normalisée)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{args.out_prefix}_psd.png", dpi=160)

        # zoom linéaire
        f0 = det["f0"]
        if np.isfinite(f0) and f0>0:
            win = (f>=0.7*f0) & (f<=1.3*f0)
            if win.sum()>10:
                Pm_s = gaussian_filter1d(P_med, sigma=args.smooth) if args.smooth>0 else P_med
                plt.figure(figsize=(8,3))
                plt.plot(f[win], Pm_s[win], lw=2)
                plt.axvline(f0, color="r", ls="--", label="f0")
                if np.isfinite(det["df"]):
                    plt.axvline(det["f1"], color="k", ls=":")
                    plt.axvline(det["f2"], color="k", ls=":")
                    plt.title(f"Pic commun: f0={f0:.4g} (P≈{1/f0:.2f} j), Δf={det['df']:.2g} ⇒ Q={det['Q']:.1f}")
                else:
                    plt.title(f"Pic commun: f0={f0:.4g} (P≈{1/f0:.2f} j)")
                plt.xlabel("fréquence (1/jour)")
                plt.ylabel("puissance lissée")
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{args.out_prefix}_peak_zoom.png", dpi=160)

if __name__ == "__main__":
    main()
