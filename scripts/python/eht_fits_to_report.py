#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Auteur: L. Danion — ORCID 0009-0008-8733-8261

python eht_fits_to_report.py \
  --fits E17A10.0.bin0000.source0000-1923lo.FITS E17A10.1.bin0000.source0000-1923lo.FITS \
  --label 1921-293_LO --dt 0.4 --period-min 60 --period-max 5000 --nfreq 6000 \
  --topk 12 --outdir scan_1921-293_lo

eht_fits_to_report.py  —  Robust extractor + periodogram for EHT FITS-IDI

Usage (examples):
  python eht_fits_to_report.py \
    --fits E17A10.0.bin0000.source0000-1923lo.FITS E17A10.1.bin0000.source0000-1923lo.FITS \
    --label 1921-293_LO --dt 0.4 --period-min 60 --period-max 5000 --nfreq 6000 \
    --topk 12 --outdir scan_1921-293_lo

  python eht_fits_to_report.py \
    --fits E17A10.0.bin0000.source0000.FITS E17A10.1.bin0000.source0000.FITS \
    --label 1921-293_HI --dt 0.4 --period-min 60 --period-max 5000 --nfreq 6000 \
    --topk 12 --outdir scan_1921-293_hi
"""
import argparse, json, os, math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# ----------------------------- helpers -----------------------------
def find_uv_data_hdu(hdul):
    """Return index of UV_DATA table in a FITS-IDI file."""
    for i, h in enumerate(hdul):
        # FITS-IDI stores a BINTABLE with EXTNAME 'UV_DATA' typically
        extname = (h.header.get("EXTNAME") or "").strip().upper()
        if "UV_DATA" in extname or ("UU--SIN" in h.columns.names if hasattr(h, "columns") else False):
            return i
    # fallback: first BinTable that has columns TIME + FLUX
    for i, h in enumerate(hdul):
        if hasattr(h, "columns"):
            cols = set([c.upper() for c in h.columns.names])
            if {"TIME", "FLUX"}.issubset(cols):
                return i
    raise RuntimeError("No UV_DATA-like BINTABLE found.")

def read_one_fits_ts(path):
    """Read one FITS-IDI file and return (t_sec, flux) 1D arrays."""
    with fits.open(path, memmap=True) as hdul:
        idx = find_uv_data_hdu(hdul)
        tab = hdul[idx].data

        # columns are vectorized per row for baselines; we average per integration
        # TIME + DATE may be present; INT TIM often in seconds
        time_col = np.array(tab["TIME"], dtype="float64")
        if time_col.ndim > 1:
            time_col = np.mean(time_col, axis=1)

        # Some sets have DATE (MJD integer) + TIME (days) or TIME (seconds)
        # Heuristic: if the spread of TIME is < 5e4, assume TIME in days and convert to seconds
        # If there is DATE, include it to get monotonic absolute seconds, then re-zero later.
        date_col = None
        if "DATE" in tab.columns.names:
            date_col = np.array(tab["DATE"], dtype="float64")
            if date_col.ndim > 1:
                date_col = np.mean(date_col, axis=1)

        # Integration time
        if "INTTIM" in tab.columns.names:
            inttim = np.array(tab["INTTIM"], dtype="float64")
            if inttim.ndim > 1: inttim = np.mean(inttim, axis=1)
        else:
            inttim = None

        # Convert TIME units -> seconds
        t = time_col.copy()
        span = np.nanmax(t) - np.nanmin(t)
        if span < 5e4:  # very likely "days"
            t = t * 86400.0
        # include DATE if present (days)
        if date_col is not None:
            t = (date_col * 86400.0) + t

        # Re-zero and sort
        t = t - np.nanmin(t)
        order = np.argsort(t)
        t = t[order]

        # Build scalar flux: mean across baselines per integration
        flux = np.array(tab["FLUX"], dtype="float64")
        if flux.ndim > 1:
            flux = np.nanmean(flux, axis=1)
        flux = flux[order]

        # Guard against NaNs/Infs
        mask = np.isfinite(t) & np.isfinite(flux)
        t = t[mask]
        flux = flux[mask]

        # If there are duplicated times, average them
        if len(t) == 0:
            raise RuntimeError(f"{path}: empty time series after cleaning.")
        u, inv = np.unique(t, return_inverse=True)
        if len(u) < len(t):
            acc = np.zeros_like(u, dtype=float)
            cnt = np.zeros_like(u, dtype=float)
            np.add.at(acc, inv, flux)
            np.add.at(cnt, inv, 1.0)
            flux = acc / np.maximum(cnt, 1.0)
            t = u

        return t, flux

def resample_uniform(t, y, dt=0.4):
    """Uniform resampling by linear interpolation onto 0..T with step dt."""
    if len(t) < 4:
        raise RuntimeError("Not enough points to resample.")
    T = t[-1]
    n = int(math.floor(T / dt)) + 1
    tu = np.linspace(0.0, n * dt, n, endpoint=False)
    yu = np.interp(tu, t, y)
    return tu, yu

def detrend_normalize(y, poly=1):
    """Remove low-order polynomial and normalize to unit variance."""
    x = np.arange(len(y), dtype=float)
    if poly >= 0:
        coeff = np.polyfit(x, y, deg=poly)
        trend = np.polyval(coeff, x)
        y = y - trend
    y = y - np.nanmean(y)
    std = np.nanstd(y)
    if std > 0:
        y = y / std
    return y

def log_period_grid(pmin, pmax, nfreq):
    freqs = 1.0 / np.geomspace(pmax, pmin, num=nfreq)  # ascending freq
    return freqs

def periodogram_fft(y, fs, taper=True):
    """Simple FFT periodogram in (freq, power)."""
    n = len(y)
    if taper:
        w = np.hanning(n)
        yw = y * w
    else:
        yw = y
    Y = np.fft.rfft(yw)
    P = (np.abs(Y) ** 2) / n
    f_full = np.fft.rfftfreq(n, d=1.0 / fs)  # CAREFUL: numpy uses d=sample spacing
    # We want f in Hz; sample spacing = 1/fs -> OK
    return f_full, P

def regrid_spectrum(f_full, P_full, f_grid):
    """Interpolate |P| onto a target frequency grid."""
    # Ensure monotonic increasing f_full
    idx = np.argsort(f_full)
    f_full = f_full[idx]; P_full = P_full[idx]
    # Interpolate in log(P) for dynamic range stability
    P_pos = np.maximum(P_full, np.max(P_full)*1e-16)
    Pg = np.exp(np.interp(f_grid, f_full, np.log(P_pos), left=np.nan, right=np.nan))
    # replace nans by 0
    Pg[np.isnan(Pg)] = 0.0
    return Pg

def find_top_peaks(f, P, topk=12, min_separation_bins=3):
    """Return list of top peaks (period_s, f_Hz, power)."""
    idx = np.argsort(P)[::-1]
    keep = []
    for i in idx:
        if len(keep) >= topk: break
        if all(abs(i - j) >= min_separation_bins for j in keep):
            keep.append(i)
    keep = sorted(keep)
    out = []
    for i in keep:
        out.append({
            "period_s": float(1.0 / f[i] if f[i] > 0 else np.inf),
            "f_Hz": float(f[i]),
            "power": float(P[i])
        })
    # sort by descending power
    out = sorted(out, key=lambda r: r["power"], reverse=True)
    return out

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", nargs="+", required=True, help="One or more FITS-IDI files")
    ap.add_argument("--label", default="EHT", help="Label in report")
    ap.add_argument("--dt", type=float, default=0.4, help="Uniform resample step [s]")
    ap.add_argument("--period-min", type=float, default=60.0)
    ap.add_argument("--period-max", type=float, default=5000.0)
    ap.add_argument("--nfreq", type=int, default=6000)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--poly", type=int, default=1, help="Detrend polynomial order")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # -- read & merge
    all_t, all_y = [], []
    for fpath in args.fits:
        try:
            t, y = read_one_fits_ts(fpath)
            all_t.append(t); all_y.append(y)
            print(f"[OK] {fpath}: {len(t)} points, span={t[-1]:.1f}s")
        except Exception as e:
            print(f"[WARN] {fpath}: {e}")

    if not all_t:
        raise RuntimeError("No usable time series read from the provided FITS.")

    # Concatenate and sort (if multiple files)
    t = np.concatenate(all_t); y = np.concatenate(all_y)
    ord_ = np.argsort(t); t = t[ord_]; y = y[ord_]

    # Uniform resample
    tu, yu = resample_uniform(t, y, dt=args.dt)
    yu = detrend_normalize(yu, poly=args.poly)
    fs = 1.0 / args.dt

    # FFT spectrum then regrid to requested log-period grid
    f_full, P_full = periodogram_fft(yu, fs=fs, taper=True)
    f_grid = log_period_grid(args.period_min, args.period_max, args.nfreq)
    P_grid = regrid_spectrum(f_full, P_full, f_grid)

    # Peaks
    peaks = find_top_peaks(f_grid, P_grid, topk=args.topk)

    # Save report JSON compatible with our other tools
    report = {
        "root": {
            "input_fits": args.fits,
            "label": args.label,
            "bin_sec": args.dt,
            "period_min_s": args.period_min,
            "period_max_s": args.period_max,
            "n_points": int(len(f_grid)),
            "top_peaks": peaks,
            # arrays for downstream comb/timeslides:
            "f": f_grid.tolist(),
            "P_sig": P_grid.tolist()
        }
    }
    with open(os.path.join(args.outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[DONE] JSON → {args.outdir}/report.json")

    # Plots
    # 1) time series
    plt.figure(figsize=(9,3.5))
    plt.plot(tu, yu, lw=0.8)
    plt.xlabel("Time [s]"); plt.ylabel("Flux (detrended, norm.)")
    plt.title(f"{args.label} — uniform series (dt={args.dt}s)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "timeseries.png"), dpi=150)
    plt.close()

    # 2) periodogram (period on log x)
    periods = 1.0 / np.maximum(f_grid, 1e-12)
    plt.figure(figsize=(8.5,4.2))
    plt.semilogx(periods, P_grid, lw=1.2, label="spectrum")
    for pk in peaks:
        plt.axvline(pk["period_s"], ls="--", alpha=0.4)
    plt.gca().invert_xaxis()  # long periods to the left like your previous plots
    plt.xlabel("Period [s] (log)"); plt.ylabel("Power (arb.)")
    plt.title(f"{args.label} — periodogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "periodogram.png"), dpi=150)
    plt.close()
    print(f"[DONE] PNGs → timeseries.png, periodogram.png")

if __name__ == "__main__":
    main()
