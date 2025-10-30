#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Auteur: L. Danion — ORCID 0009-0008-8733-8261

python comb_timeslides.py \
  --report m87_scan_full/report.json \
  --p0 238 --nharm 6 --nperm 1000 \
  --out m87_comb_p238

Compute a harmonic comb score for period P0 and assess significance
via time-slides simulated on the stacked spectrum from a report.json.

Inputs
------
--report : path to JSON with frequency grid and stacked power spectrum.
           Expected keys (any of): 
             - {"spectrum": {"f": [...], "P": [...]} }
             - {"f": [...], "P": [...]}   (flat)
             - a top-level dict with the first two equal-length numeric arrays
--p0     : fundamental period [s]
--nharm  : number of harmonics to include (k=1..nharm)
--nperm  : number of circular-shift permutations for null
--band   : optional "fmin,fmax" (Hz) to restrict the analysis band
--width  : optional half-window around each harmonic (Hz). If omitted,
           uses 2 * median df of the frequency grid.
--title  : optional figure title
--out    : output prefix (PNG + JSON). If ends with '/', a prefix is auto-chosen.

Outputs
-------
<out>.png  : histogram of permuted comb scores with observed vertical line
<out>.json : parameters, observed comb, null summary, p-value

"""

import argparse, json, os, sys
import numpy as np
import matplotlib.pyplot as plt

def load_spectrum(path):
    with open(path, "r") as f:
        d = json.load(f)

    # Try usual nesting
    if isinstance(d, dict):
        # common: {"spectrum":{"f":[...],"P":[...]}}
        if "spectrum" in d and isinstance(d["spectrum"], dict):
            s = d["spectrum"]
            f = np.asarray(s.get("f", []), float)
            P = np.asarray(s.get("P", []), float)
            if f.size and P.size and f.size == P.size:
                return f, P

        # flat: {"f":[...],"P":[...]}
        if "f" in d and "P" in d:
            f = np.asarray(d["f"], float)
            P = np.asarray(d["P"], float)
            if f.size and P.size and f.size == P.size:
                return f, P

        # fallback: pick first two same-length numeric arrays
        candidates = []
        for k, v in d.items():
            if isinstance(v, (list, tuple)) and len(v) > 3:
                try:
                    arr = np.asarray(v, float)
                    if np.all(np.isfinite(arr)):
                        candidates.append((k, arr))
                except Exception:
                    pass
        if len(candidates) >= 2:
            # choose two with same length
            for i in range(len(candidates)):
                for j in range(i+1, len(candidates)):
                    if candidates[i][1].size == candidates[j][1].size:
                        # guess which is frequency by monotonicity
                        a = candidates[i][1]
                        b = candidates[j][1]
                        def is_monotone(x):
                            return np.all(np.diff(x) > 0) or np.all(np.diff(x) < 0)
                        if is_monotone(a):
                            return a, b
                        if is_monotone(b):
                            return b, a

    raise RuntimeError("Could not find matching 'f' and 'P' arrays in the report JSON.")

def restrict_band(f, P, fmin=None, fmax=None):
    if fmin is None and fmax is None:
        return f, P
    m = np.ones_like(f, dtype=bool)
    if fmin is not None:
        m &= (f >= fmin)
    if fmax is not None:
        m &= (f <= fmax)
    return f[m], P[m]

def local_power(f, P, f0, width):
    # Average power in |f - f0| <= width
    if not np.isfinite(f0) or f0 <= 0:
        return 0.0
    m = (f >= (f0 - width)) & (f <= (f0 + width))
    if not np.any(m):
        # fallback to nearest-neighbor
        idx = np.argmin(np.abs(f - f0))
        return float(P[idx])
    return float(np.mean(P[m]))

def comb_score(f, P, P0, nharm, width):
    f0 = 1.0 / float(P0)
    score = 0.0
    used = 0
    for k in range(1, nharm+1):
        fk = k * f0
        if fk < f[0] or fk > f[-1]:
            continue
        score += local_power(f, P, fk, width)
        used += 1
    return score, used

def circular_roll(arr, shift):
    # Positive shift rolls to the right
    return np.roll(arr, int(shift))

def auto_out_prefix(out_arg):
    if out_arg.endswith(os.sep):
        os.makedirs(out_arg, exist_ok=True)
        return os.path.join(out_arg, "comb_timeslides")
    # if parent dir is a folder, make sure it exists
    d = os.path.dirname(out_arg)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    return out_arg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="Path to report.json with stacked spectrum.")
    ap.add_argument("--p0", type=float, required=True, help="Fundamental period [s].")
    ap.add_argument("--nharm", type=int, default=6, help="Number of harmonics to include.")
    ap.add_argument("--nperm", type=int, default=1000, help="Number of permutations for null.")
    ap.add_argument("--band", type=str, default=None, help="Optional fmin,fmax in Hz (e.g. 2e-4,5e-3).")
    ap.add_argument("--width", type=float, default=None, help="Half-window around each harmonic [Hz]. Default=2*median(df).")
    ap.add_argument("--title", type=str, default=None, help="Optional title for the figure.")
    ap.add_argument("--out", type=str, required=True, help="Output prefix (PNG+JSON). If ends with '/', acts as directory.")
    args = ap.parse_args()

    f, P = load_spectrum(args.report)

    # Sort by freq if needed
    idx = np.argsort(f)
    f = f[idx]
    P = P[idx]

    # Restrict band if asked
    fmin = fmax = None
    if args.band:
        try:
            parts = [float(x) for x in args.band.split(",")]
            if len(parts) == 2:
                fmin, fmax = parts
        except Exception:
            pass
    f, P = restrict_band(f, P, fmin, fmax)
    if f.size < 10:
        raise RuntimeError("Frequency grid too small after band restriction.")

    # Default width: 2 * median(df)
    if args.width is None:
        df = np.median(np.diff(f))
        width = 2.0 * df
    else:
        width = float(args.width)

    # Observed comb score
    comb_obs, nh_used = comb_score(f, P, args.p0, args.nharm, width)

    # Null via circular time-slides (random rolls)
    rng = np.random.default_rng(12345)
    n = P.size
    comb_null = np.empty(args.nperm, dtype=float)
    for i in range(args.nperm):
        shift = rng.integers(0, n)
        P_perm = circular_roll(P, shift)
        comb_null[i], _ = comb_score(f, P_perm, args.p0, args.nharm, width)

    # p-value (upper tail)
    pval = (1.0 + np.sum(comb_null >= comb_obs)) / (args.nperm + 1.0)

    # Save outputs
    out_prefix = auto_out_prefix(args.out)
    # Figure
    plt.figure(figsize=(7.2, 3.6))
    plt.hist(comb_null, bins=50, density=True, alpha=0.9, label="time-slides")
    plt.axvline(comb_obs, linestyle="--", label=f"zero-lag (comb={comb_obs:.3g})")
    ttl = args.title or f"Comb P0≈{args.p0:.1f}s ; nharm={args.nharm} ; p≈{pval:.3f}"
    plt.title(ttl)
    plt.xlabel("Comb power")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + ".png", dpi=150)
    plt.close()

    # JSON
    out = dict(
        params=dict(
            report=args.report,
            P0_s=args.p0,
            nharm=args.nharm,
            nperm=args.nperm,
            band=[fmin, fmax],
            width_Hz=width,
            n_points=int(f.size),
            df_median=np.median(np.diff(f)),
        ),
        observed=dict(
            comb=float(comb_obs),
            harmonics_used=int(nh_used)
        ),
        null=dict(
            mean=float(np.mean(comb_null)),
            std=float(np.std(comb_null)),
            min=float(np.min(comb_null)),
            max=float(np.max(comb_null))
        ),
        p_value=float(pval)
    )
    with open(out_prefix + ".json", "w") as fo:
        json.dump(out, fo, indent=2)
    print(f"[DONE] Saved → {out_prefix+'.png'} & {out_prefix+'.json'}")
    print(f"p-value ≈ {pval:.4f} (comb={comb_obs:.3g}, nharm_used={nh_used})")

if __name__ == "__main__":
    main()
