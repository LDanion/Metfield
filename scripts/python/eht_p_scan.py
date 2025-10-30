#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Auteur: L. Danion — ORCID 0009-0008-8733-8261

eht_p_scan.py

Scan de périodes sur bandes LO/HI (FITS ou CSV), avec détrend/apodisation,
Lomb–Scargle, top-peaks, timeslides, et jackknife.

Exemples :
  # toutes les LO vs toutes les HI
  python eht_p_scan.py \
    --lo "SGRA_low_fits/*.FITS" \
    --hi "SGRA_high_fits/*.FITS" \
    --label SgrA_ALL_detr \
    --period-min 60 --period-max 3800 \
    --nfreq 8000 \
    --detrend poly:3 \
    --apod cos:0.1 \
    --topk 20 \
    --timeslides 5000 \
    --outdir scan_sgra_all4000_detr

    python eht_p_scan.py \
    --lo "CENA_lo_fits/*.FITS" \
    --hi "CENA_hi_fits/*.FITS" \
    --label CENA_detr \
    --period-min 60 --period-max 5000 \
    --nfreq 8000 \
    --detrend poly:3 \
    --apod cos:0.1 \
    --topk 20 \
    --timeslides 5000 \
    --outdir scan_CENA


"""

import argparse, json, os, sys, glob, math, random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

# SciPy est utile (Savitzky-Golay, find_peaks, lombscargle fallback)
from scipy.signal import savgol_filter, find_peaks
from scipy import signal

# Astropy (optionnel mais préférable pour Lomb–Scargle)
try:
    from astropy.io import fits as afits
    from astropy.timeseries import LombScargle as A_LS
    HAS_ASTROPY = True
except Exception:
    HAS_ASTROPY = False

# Matplotlib optionnel
try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False


# --------------------- Utils ---------------------

def log(msg: str):
    print(msg, file=sys.stderr)

def expand_paths(pattern: str) -> List[str]:
    # accepte motif glob (entre guillemets dans le shell) et listes séparées par ','
    paths = []
    for tok in pattern.split(","):
        tok = tok.strip()
        paths.extend(glob.glob(tok))
    return sorted(list(dict.fromkeys(paths)))  # unique & stable

def robust_nanstd(x):
    x = np.asarray(x, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return 1.4826 * mad

def cosine_taper(n: int, frac: float) -> np.ndarray:
    """Cosine taper de fraction 'frac' aux bords (0..1)."""
    frac = max(0.0, min(0.49, float(frac)))
    w = np.ones(n, float)
    m = int(frac * n)
    if m <= 0:
        return w
    x = np.linspace(0, np.pi/2, m)
    w[:m] = np.sin(x)**2
    w[-m:] = w[:m][::-1]
    return w


# --------------------- I/O séries ---------------------

def read_csv_series(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lit un CSV avec colonnes header au moins 'time' et 'value'.
    """
    import pandas as pd
    df = pd.read_csv(path)
    # colonnes les plus probables
    tcol = None
    for c in df.columns:
        if str(c).lower() in ("time", "t", "timestamp", "mjd", "jd", "secs", "seconds"):
            tcol = c; break
    vcol = None
    for c in df.columns:
        if str(c).lower() in ("value", "y", "amp", "amplitude", "flux", "signal", "phase"):
            vcol = c; break
    if tcol is None or vcol is None:
        raise ValueError(f"{path}: columns 'time'/'value' not found")
    t = df[tcol].to_numpy(dtype=float)
    y = df[vcol].to_numpy(dtype=float)
    return t, y

def read_fits_series(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extraction robuste (t, y) depuis des FITS EHT-like.
    - Cherche des colonnes temps parmi: TIME,T,TS,UTC,MJD,JD,SECONDS,SEC
    - Cherche des colonnes signal parmi: AMP,AMP_VIS,AMPL,VALUE,FLUX,SIGNAL,Y,
      REAL/RE, IMAG/IM -> amplitude = sqrt(RE^2+IM^2)
    - Fallback image/PRIMARY: t = index, y = pixels
    """
    if not HAS_ASTROPY:
        raise RuntimeError("astropy non disponible pour lire FITS")
    cand_t = {"time","t","ts","utc","mjd","jd","seconds","sec"}
    cand_y = {"amp","amp_vis","ampl","value","flux","signal","y"}
    cand_re = {"real","re","real_vis","re_vis"}
    cand_im = {"imag","im","imag_vis","im_vis"}

    with afits.open(path, memmap=False) as hdul:
        # 1) tables binaires
        for h in hdul:
            if isinstance(h, afits.BinTableHDU):
                cols = [c.lower() for c in h.columns.names]
                # temps
                tname = next((h.columns.names[i] for i,c in enumerate(cols) if c in cand_t), None)
                # amplitude directe
                yname = next((h.columns.names[i] for i,c in enumerate(cols) if c in cand_y), None)
                # sinon amplitude à partir de Re/Im
                rname = next((h.columns.names[i] for i,c in enumerate(cols) if c in cand_re), None)
                iname = next((h.columns.names[i] for i,c in enumerate(cols) if c in cand_im), None)

                if tname and (yname or (rname and iname)):
                    t = np.asarray(h.data[tname], dtype=float).ravel()
                    if yname:
                        y = np.asarray(h.data[yname], dtype=float).ravel()
                    else:
                        re = np.asarray(h.data[rname], dtype=float).ravel()
                        im = np.asarray(h.data[iname], dtype=float).ravel()
                        y = np.sqrt(re*re + im*im)
                    # sécurité basique
                    if t.size >= 8 and y.size == t.size:
                        return t, y

        # 2) images (PRIMARY/ImageHDU)
        for h in hdul:
            if isinstance(h, (afits.ImageHDU, afits.PrimaryHDU)):
                arr = np.asarray(h.data, dtype=float).ravel()
                if arr.size >= 8:
                    t = np.arange(arr.size, dtype=float)
                    return t, arr

    raise ValueError(f"{path}: structure FITS non reconnue (ni TIME/AMP ni image).")


def load_series(paths: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Charge et concatène plusieurs fichiers en (t,y)."""
    T, Y, used = [], [], []
    for p in paths:
        try:
            if p.lower().endswith(".csv"):
                t,y = read_csv_series(p)
            else:
                t,y = read_fits_series(p)
            if len(t) != len(y) or len(t) < 8:
                continue
            # normalisation par fichier pour éviter domination
            y = (y - np.nanmedian(y))
            std = robust_nanstd(y)
            y = y / (std if std>0 else (np.nanstd(y)+1e-12))
            # concat
            T.append(np.asarray(t, float))
            Y.append(np.asarray(y, float))
            used.append(p)
        except Exception as e:
            log(f"[WARN] skip {p}: {type(e)._name_}: {e}")
            continue
    if not used:
        return np.array([], float), np.array([], float), used
    # concatène en une seule série (temps relatif)
    t = np.concatenate(T)
    y = np.concatenate(Y)
    # remettre t au repère 0 et trier
    t = t - np.nanmin(t)
    idx = np.argsort(t)
    return t[idx], y[idx], used


# --------------------- Prétraitements ---------------------

def apply_detrend(t: np.ndarray, y: np.ndarray, spec: Optional[str]) -> np.ndarray:
    """
    'poly:k' -> fit poly ordre k sur t et soustrait.
    'hp:tau' -> high-pass doux via Savitzky-Golay avec fenêtre ~ tau (s).
    """
    if spec is None:
        return y
    try:
        kind, val = spec.split(":")
    except ValueError:
        kind, val = spec, ""
    kind = kind.strip().lower()
    if kind == "poly":
        k = max(1, int(val)) if val else 2
        # centre t pour éviter mauvaise condition
        tc = t - np.mean(t)
        coefs = np.polyfit(tc, y, k)
        trend = np.polyval(coefs, tc)
        return y - trend
    elif kind == "hp":
        # val = tau (s) ~ échelle lente à enlever
        tau = float(val) if val else (0.2 * (np.max(t)-np.min(t)+1e-9))
        dt = np.median(np.diff(np.sort(t)))
        win = max(7, int(round(tau / max(dt, 1e-6))))
        if win % 2 == 0:
            win += 1
        # low-pass SG puis soustraction
        try:
            trend = savgol_filter(y, window_length=win, polyorder=3, mode="interp")
        except Exception:
            # fallback : moyenne glissante simple
            k = min(win, max(3, (len(y)//51)*2+1))
            kernel = np.ones(k)/k
            trend = np.convolve(y, kernel, mode="same")
        return y - trend
    else:
        return y

def apply_apod(y: np.ndarray, spec: Optional[str]) -> np.ndarray:
    """
    'cos:frac' -> cosine taper de fraction 'frac' aux bords.
    """
    if not spec:
        return y
    try:
        kind, val = spec.split(":")
    except ValueError:
        kind, val = spec, ""
    if kind.strip().lower() != "cos":
        return y
    frac = float(val) if val else 0.1
    w = cosine_taper(len(y), frac)
    return y * w


# --------------------- Lomb–Scargle ---------------------

def lomb_scargle(t: np.ndarray, y: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
    """
    Renvoie la puissance normalisée sur la grille freqs_hz.
    Essaye Astropy, sinon SciPy (ang freq).
    """
    # enlever NaN
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if t.size < 16:
        return np.zeros_like(freqs_hz)
    # centrer y (floating mean approx.)
    y = y - np.mean(y)

    if HAS_ASTROPY:
        ls = A_LS(t, y)
        power = ls.power(freqs_hz, normalization='psd')
        return np.asarray(power, float)
    else:
        # SciPy lombscargle attend des ω (rad/s)
        omega = 2*np.pi*freqs_hz
        # standardise à la variance
        ystd = y / (np.std(y)+1e-12)
        p = signal.lombscargle(t, ystd, omega, normalize=True)
        return np.asarray(p, float)


# --------------------- Détection de pics ---------------------

@dataclass
class Peak:
    period_s: float
    freq_hz: float
    power: float
    index: int

def pick_top_peaks(freqs, power, topk=20, guard_bins=2) -> List[Peak]:
    """Trouve des pics locaux et renvoie les topk triés par puissance."""
    # ignore bords
    mask = np.ones_like(power, dtype=bool)
    mask[:guard_bins] = False
    mask[-guard_bins:] = False
    # pics locaux
    peaks, _ = find_peaks(power[mask])
    if peaks.size == 0:
        return []
    # reindex dans le tableau complet
    idxs = np.arange(len(power))[mask][peaks]
    vals = power[idxs]
    order = np.argsort(vals)[::-1]
    idxs = idxs[order][:topk]
    out = []
    for i in idxs:
        f = float(freqs[i])
        p = float(power[i])
        out.append(Peak(period_s=(1.0/f if f>0 else float('nan')),
                        freq_hz=f, power=p, index=int(i)))
    # trie final par période croissante
    out.sort(key=lambda z: z.period_s)
    return out


# --------------------- Timeslides ---------------------

def timeslides_pvalue(t: np.ndarray, y: np.ndarray, freqs_hz: np.ndarray,
                      observed_stat: float, nslides: int = 2000, rng: int = 42) -> float:
    """
    Estime p-val en décalant circulairement y d'un offset aléatoire à chaque slide
    et en recalculant la stat (ici max(power)).
    """
    if nslides <= 0:
        return float('nan')
    rnd = np.random.default_rng(rng)
    N = len(y)
    better = 0
    for _ in range(nslides):
        shift = int(rnd.integers(0, N))
        y_sh = np.roll(y, shift)
        p = lomb_scargle(t, y_sh, freqs_hz)
        stat = float(np.max(p))
        if stat >= observed_stat:
            better += 1
    return (better + 1.0) / (nslides + 1.0)


# --------------------- Jackknife ---------------------

def jackknife_maxpower(files: List[str], period_grid_s: np.ndarray,
                       detrend_spec: Optional[str], apod_spec: Optional[str]) -> List[float]:
    """
    Pour chaque leave-one-out, renvoie le max(power) (stat simple mais robuste).
    """
    if len(files) < 2:
        return []
    vals = []
    for i in range(len(files)):
        subset = files[:i] + files[i+1:]
        t, y, used = load_series(subset)
        if len(used) == 0:
            continue
        y = apply_detrend(t, y, detrend_spec)
        y = apply_apod(y, apod_spec)
        freqs = 1.0 / period_grid_s
        p = lomb_scargle(t, y, freqs)
        vals.append(float(np.max(p)))
    return vals


# --------------------- Orchestration ---------------------

def run(args):
    os.makedirs(args.outdir, exist_ok=True)

    lo_files = expand_paths(args.lo)
    hi_files = expand_paths(args.hi)

    if not lo_files and not hi_files:
        log("[ERR] Aucun fichier trouvé (LO/HI). Vérifie les motifs et les guillemets.")
        sys.exit(2)

    # si on force une seule bande, on duplique l'autre
    if not lo_files and hi_files:
        lo_files = hi_files
    if not hi_files and lo_files:
        hi_files = lo_files

    # charge & fusionne LO+HI
    t_lo, y_lo, used_lo = load_series(lo_files)
    t_hi, y_hi, used_hi = load_series(hi_files)

    if len(used_lo)==0 or len(used_hi)==0:
        log("[ERR] Impossible de charger LO et/ou HI.")
        sys.exit(2)

    # concatène LO+HI (z-score déjà appliqué fichier-par-fichier)
    t = np.concatenate([t_lo, t_hi])
    y = np.concatenate([y_lo, y_hi])
    # re-tri par temps
    idx = np.argsort(t)
    t, y = t[idx], y[idx]

    # prétraitements
    y = apply_detrend(t, y, args.detrend)
    y = apply_apod(y, args.apod)

    # grille de fréquences via périodes
    periods = np.linspace(args.period_min, args.period_max, args.nfreq)
    freqs = 1.0 / periods

    power = lomb_scargle(t, y, freqs)
    top_peaks = pick_top_peaks(freqs, power, topk=args.topk, guard_bins=2)

    observed_max = float(np.max(power))
    p_timeslides = timeslides_pvalue(t, y, freqs, observed_max, nslides=args.timeslides, rng=42)

    # jackknife (si plusieurs fichiers)
    jk_stats = jackknife_maxpower(list(set(used_lo + used_hi)), periods,
                                  args.detrend, args.apod)

    # Sauvegardes
    label = args.label or "scan"
    out_json = os.path.join(args.outdir, "report.json")
    result = dict(
        label=label,
        params=dict(
            lo_files=used_lo,
            hi_files=used_hi,
            period_min_s=args.period_min,
            period_max_s=args.period_max,
            nfreq=args.nfreq,
            detrend=args.detrend,
            apod=args.apod,
            topk=args.topk,
            timeslides=args.timeslides,
        ),
        peaks=[dict(period_s=p.period_s, freq_hz=p.freq_hz, power=p.power, index=p.index)
               for p in top_peaks],
        max_power=float(observed_max),
        p_timeslides=float(p_timeslides) if not math.isnan(p_timeslides) else None,
        jackknife_maxpower=jk_stats if jk_stats else None,
    )
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # PNG optionnels
    if HAS_PLOT:
        figpath = os.path.join(args.outdir, "periodogram.png")
        plt.figure(figsize=(8,4))
        plt.plot(periods, power)
        for pk in top_peaks:
            plt.axvline(pk.period_s, ls='--', alpha=0.4)
        plt.gca().invert_xaxis()  # visuel : petites périodes à droite
        plt.xlabel("Period [s]")
        plt.ylabel("Power (LS)")
        plt.title(f"{label}: LS periodogram")
        plt.tight_layout()
        plt.savefig(figpath, dpi=140)
        plt.close()

        tsfig = os.path.join(args.outdir, "timeseries.png")
        plt.figure(figsize=(8,3))
        plt.plot(t, y, lw=0.6)
        plt.xlabel("Time [s] (relative)")
        plt.ylabel("Amplitude (z-score)")
        plt.title(f"{label}: preprocessed time series")
        plt.tight_layout()
        plt.savefig(tsfig, dpi=140)
        plt.close()

    print(f"[OK] report -> {out_json}")
    if HAS_PLOT:
        print(f"[OK] figures -> {os.path.join(args.outdir, 'periodogram.png')}, {os.path.join(args.outdir, 'timeseries.png')}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", required=True, help="Motif fichiers LO (ex: 'SGRA_low_fits/*.FITS' ou CSV).")
    ap.add_argument("--hi", required=True, help="Motif fichiers HI (ex: 'SGRA_high_fits/*.FITS' ou CSV).")
    ap.add_argument("--label", type=str, default="", help="Étiquette pour les sorties.")
    ap.add_argument("--period-min", type=float, required=True, dest="period_min",
                    help="Période minimale (s).")
    ap.add_argument("--period-max", type=float, required=True, dest="period_max",
                    help="Période maximale (s).")
    ap.add_argument("--nfreq", type=int, default=8000, help="Nombre de points de période/frequence.")
    ap.add_argument("--detrend", type=str, default=None,
                    help="Ex: 'poly:2' ou 'hp:6000' (échelle lente à supprimer).")
    ap.add_argument("--apod", type=str, default=None,
                    help="Ex: 'cos:0.1' (taper fraction).")
    ap.add_argument("--topk", type=int, default=20, help="Nombre de pics à rapporter.")
    ap.add_argument("--timeslides", type=int, default=0,
                    help="Nombre de décalages circulaires pour p-val empirique (0 pour désactiver).")
    ap.add_argument("--outdir", type=str, required=True, help="Dossier de sortie.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
