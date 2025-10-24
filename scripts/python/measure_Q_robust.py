#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Auteur: L. Danion — ORCID 0009-0008-8733-8261
measure_Q_robust.py — Estime f0, Δf (FWHM) et Q = f0/Δf sur séries bruitées.
Fonctionne avec séries à pas régulier (Welch) ou irrégulier (Lomb–Scargle).
Ajoute un lissage spectral (gaussien) et une détection de pics robuste.

Bash :
  python measure_Q_robust.py --input f107_prep.csv --time-col t --value-col Y \
         --out-prefix f107_Q --plot
"""

import argparse, json, math
import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks
from scipy.signal.windows import hann
from scipy.ndimage import gaussian_filter1d
from numpy.polynomial.polynomial import Polynomial

try:
    from astropy.timeseries import LombScargle
    HAS_ASTROPY = True
except Exception:
    HAS_ASTROPY = False


def is_regular(t, tol=1e-6):
    dt = np.diff(t)
    return np.all(np.isfinite(dt)) and (np.nanstd(dt) / (np.nanmean(dt) + 1e-15) < tol)


def detrend_zscore(y):
    # petit polynôme (ordre 2) pour ôter la dérive lente
    x = np.arange(len(y), dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 10:
        return (y - np.nanmean(y)) / (np.nanstd(y) + 1e-12)
    coefs = Polynomial.fit(x[mask], y[mask], deg=2).convert().coef
    trend = coefs[0] + coefs[1]*x + (coefs[2] if len(coefs) > 2 else 0.0) * x * x
    r = y - trend
    return (r - np.nanmean(r)) / (np.nanstd(r) + 1e-12)


def fwhm_from_peak(f, p, ipk):
    """Largeur à mi-hauteur autour du pic ipk (avec interpolation linéaire)."""
    half = p[ipk] / 2.0
    # gauche
    i = int(ipk)
    while i > 0 and p[i] > half:
        i -= 1
    f1 = f[i]
    if i < ipk and p[i+1] != p[i]:
        f1 = np.interp(half, [p[i], p[i+1]], [f[i], f[i+1]])
    # droite
    j = int(ipk)
    n = len(p) - 1
    while j < n and p[j] > half:
        j += 1
    f2 = f[j]
    if j > ipk and p[j] != p[j-1]:
        f2 = np.interp(half, [p[j-1], p[j]], [f[j-1], f[j]])
    return abs(f2 - f1), f1, f2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--time-col", dest="time_col", default="t")
    ap.add_argument("--value-col", dest="value_col", default="Y")
    ap.add_argument("--band", nargs=2, type=float, default=None,
                    help="bande [fmin fmax] pour la recherche de pic")
    ap.add_argument("--smooth", type=int, default=5, help="lissage gaussien (points)")
    ap.add_argument("--use-lomb", dest="use_lomb", action="store_true",
                    help="force Lomb–Scargle")
    ap.add_argument("--out-prefix", dest="out_prefix", default="Q_out")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    # Vérifie la présence des colonnes demandées
    if args.time_col not in df or args.value_col not in df:
        raise ValueError(f"Colonnes requises absentes : {args.time_col}, {args.value_col}")

# --- Conversion de la colonne de temps ---
    if np.issubdtype(df[args.time_col].dtype, np.number):
    # déjà numérique
        t = df[args.time_col].astype(float).values
    else:
    # convertit automatiquement les dates en secondes depuis le début
        t = pd.to_datetime(df[args.time_col], errors="coerce")
        t = (t - t.min()).dt.total_seconds().values

# --- Extraction de la série de valeurs ---
    y = df[args.value_col].astype(float).values

    # nettoyage
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    # ordre croissant en temps
    idx = np.argsort(t)
    t, y = t[idx], y[idx]
    # standardisation douce
    y = detrend_zscore(y)

    # spectre
    regular = is_regular(t) and (not args.use_lomb)
    if regular:
        dt = np.median(np.diff(t))
        fs = 1.0 / max(dt, 1e-12)
        nperseg = max(64, int(len(y)//8))
        f, Pxx = welch(y, fs=fs, window=hann(nperseg), nperseg=nperseg,
                       noverlap=nperseg//2, detrend=False, scaling="density")
    else:
        if not HAS_ASTROPY:
            raise RuntimeError("Astropy requis pour Lomb–Scargle (pip install astropy).")
        # grille de fréquences raisonnable
        T = max(t.max() - t.min(), 1e-9)
        fmin = 1.0 / (T * 10.0)
        # pas « Nyquist » effectif basé sur les intervalles uniques
        du = np.diff(np.unique(t))
        fmax = 0.5 / max(np.median(du) if du.size else 1.0, 1e-9)
        f = np.logspace(np.log10(max(fmin, 1e-6)), np.log10(fmax), 6000)
        ls = LombScargle(t, y, normalization="psd")
        Pxx = ls.power(f)

    # lissage gaussien
    Pxx_s = gaussian_filter1d(Pxx, sigma=max(args.smooth, 0)) if args.smooth > 0 else Pxx.copy()

    # bande de recherche
    if args.band is not None:
        fmin, fmax = args.band
        band = (f >= fmin) & (f <= fmax)
    else:
        # par défaut: ignore f très basses et la dernière valeur
        band = (f >= f[1]) & (f <= f[-2])

    fb, Pb = f[band], Pxx_s[band]
    if fb.size < 10:
        raise RuntimeError("Bande trop étroite/vidée après filtrage.")

    # détection robuste du pic principal
    med = np.median(Pb)
    mad = np.median(np.abs(Pb - med)) + 1e-15
    peaks, _ = find_peaks(Pb, height=med + 5*mad, distance=5)
    if peaks.size == 0:
        peaks, _ = find_peaks(Pb, height=med + 3*mad, distance=5)

    if peaks.size == 0:
        ipk = int(np.argmax(Pb))
        f0 = float(fb[ipk])
        note = "Pic faible (seuil relâché); f0 pris au max(P)."
    else:
        ipk = int(peaks[np.argmax(Pb[peaks])])
        f0 = float(fb[ipk])
        note = "Pic détecté (robuste)."

    # FWHM & Q
    try:
        df_fwhm, f1, f2 = fwhm_from_peak(fb, Pb, ipk)
        Q = float(f0 / max(df_fwhm, 1e-12))
    except Exception:
        df_fwhm, f1, f2, Q = float("nan"), float("nan"), float("nan"), float("nan")
        note += " FWHM non résolue."

    out = {
        "input": args.input,
        "regular_sampling": bool(regular),
        "N": int(len(y)),
        "f0": f0,
        "df_fwhm": df_fwhm,
        "Q": Q,
        "f1_half": f1,
        "f2_half": f2,
        "band": list(args.band) if args.band else None,
        "note": note
    }
    with open(f"{args.out_prefix}_Q.json", "w") as fjs:
        json.dump(out, fjs, indent=2)
    print(json.dumps(out, indent=2))

    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7,4))
        plt.loglog(f, Pxx, alpha=0.3, label="raw PSD")
        plt.loglog(f, Pxx_s, lw=2, label="smoothed PSD")
        plt.axvline(f0, color="r", ls="--", label=f"f0={f0:.4g}")
        if np.isfinite(df_fwhm):
            plt.axvline(f1, color="k", ls=":")
            plt.axvline(f2, color="k", ls=":")
        if args.band:
            plt.axvspan(args.band[0], args.band[1], color="grey", alpha=0.08, label="search band")
        plt.xlabel("frequency")
        plt.ylabel("power")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{args.out_prefix}_PSD.png", dpi=160)

        # zoom linéaire autour du pic
        iwin = (f >= 0.7*f0) & (f <= 1.3*f0)
        if iwin.sum() > 10:
            plt.figure(figsize=(7,3))
            plt.plot(f[iwin], Pxx_s[iwin], lw=2)
            plt.axvline(f0, color="r", ls="--", label="f0")
            if np.isfinite(df_fwhm):
                plt.axvline(f1, color="k", ls=":")
                plt.axvline(f2, color="k", ls=":")
                plt.title(f"Peak @ f0={f0:.4g}; Δf(FWHM)={df_fwhm:.2g} ⇒ Q={Q:.1f}")
            else:
                plt.title(f"Peak @ f0={f0:.4g} (FWHM non résolue)")
            plt.xlabel("frequency")
            plt.ylabel("smoothed power")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{args.out_prefix}_peak_zoom.png", dpi=160)


if __name__ == "__main__":
    main()
