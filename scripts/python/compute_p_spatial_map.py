#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auteur: L. Danion — ORCID 0009-0008-8733-8261

# 1) Joindre les coordonnées aux matches
python join_matches_with_coords.py \
  --events-csv data/japan/usgs_japan_2010_2024_M5.5.csv \
  --matches-csv results_eval/japan/20251024_144957/matches.csv \
  --out-csv results_eval/japan/20251024_144957/matches_with_coords.csv

# 2) Générer cartes P_spatial(t) = P(t)*KDE(x,y) pour chaque alerte
python compute_p_spatial_map.py \
  --region-name japan \
  --events-csv data/japan/usgs_japan_2010_2024_M5.5.csv \
  --alerts-csv results_eval/japan/20251024_144957/alerts.csv \
  --outdir results_eval/japan/20251024_144957/p_spatial \
  --grid-res 300 --bw-scale 1.0 --top-percent 10 --dpi 150

"""

import argparse, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path
# ... imports
import argparse
from bias import load_bias, apply_bias

def parse_time(s):
    return pd.to_datetime(str(s), utc=True, errors="coerce")

def make_kde(lat, lon, grid_res=300, bw_scale=1.0, bbox=None):
    xy = np.vstack([lon, lat])  # KDE en (x=lon, y=lat)
    kde = gaussian_kde(xy)
    kde.set_bandwidth(bw_method=kde.factor * bw_scale)

    if bbox is None:
        minlon, maxlon = np.quantile(lon, [0.01, 0.99])
        minlat, maxlat = np.quantile(lat, [0.01, 0.99])
    else:
        minlon, maxlon, minlat, maxlat = bbox

    xs = np.linspace(minlon, maxlon, grid_res)
    ys = np.linspace(minlat, maxlat, grid_res)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)

    # normalisation [0,1]
    ZZ = (ZZ - ZZ.min()) / (ZZ.max() - ZZ.min() + 1e-12)
    return XX, YY, ZZ

def main():
    ap = argparse.ArgumentParser()
    # (tes args existants)
    ap.add_argument("--bias-json", default=None, help="JSON avec bias_deg {dlon,dlat} à appliquer à la carte")
    args = ap.parse_args()
    
    # création des grilles (exemple)
    # LON, LAT = np.meshgrid(lon_vec, lat_vec)
    
  
    if args.bias_json:
        dlon, dlat = load_bias(args.bias_json)
        LON, LAT = apply_bias(LON, LAT, dlon, dlat, wrap=True)

    ap = argparse.ArgumentParser(description="Carte spatio-temporelle P(t)*KDE(x,y) par alerte.")
    ap.add_argument("--region-name", required=True)
    ap.add_argument("--events-csv", required=True, help="Catalogue pour KDE (lat/lon requis)")
    ap.add_argument("--alerts-csv", required=True, help="Alerts avec colonnes: alert_time, score (ou zscore/raw)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--amp-col", default="score", help="Colonne amplitude temporelle à utiliser (def: score)")
    ap.add_argument("--grid-res", type=int, default=300)
    ap.add_argument("--bw-scale", type=float, default=1.0)
    ap.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("minlon","maxlon","minlat","maxlat"))
    ap.add_argument("--top-percent", type=float, default=10.0, help="Isoligne sur les X%% plus élevés")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    ev = pd.read_csv(args.events_csv)
    latcol = next((c for c in ev.columns if c.lower() in ("latitude","lat")), None)
    loncol = next((c for c in ev.columns if c.lower() in ("longitude","lon")), None)
    if latcol is None or loncol is None:
        print("[ERR] Catalogue sans colonnes lat/lon.", file=sys.stderr); sys.exit(2)

    lat = ev[latcol].astype(float).values
    lon = ev[loncol].astype(float).values
    XX, YY, KDE = make_kde(lat, lon, grid_res=args.grid_res, bw_scale=args.bw_scale, bbox=args.bbox)

    alerts = pd.read_csv(args.alerts_csv)
    if "alert_time" not in alerts.columns:
        print("[ERR] alerts.csv doit contenir 'alert_time'.", file=sys.stderr); sys.exit(2)
    if args.amp_col not in alerts.columns:
        # fallback sur colonnes possibles
        for c in ("zscore","raw"):
            if c in alerts.columns:
                args.amp_col = c; break
        else:
            print("[ERR] Aucune colonne amplitude trouvée (score/zscore/raw).", file=sys.stderr); sys.exit(2)

    alerts["alert_time"] = alerts["alert_time"].apply(parse_time)
    a = alerts.dropna(subset=["alert_time"]).copy()

    # normalise amplitude temporelle sur [0,1] par quantiles robustes
    x = a[args.amp_col].astype(float).values
    lo, hi = np.nanpercentile(x, [5,95])
    a["amp_norm"] = np.clip((x - lo)/(hi - lo + 1e-12), 0, 1)

    figs = []
    for i, row in a.iterrows():
        amp = float(row["amp_norm"])
        if not np.isfinite(amp): 
            continue
        Z = amp * KDE  # produit spatio-temporel

        # figure
        fig, ax = plt.subplots(figsize=(6,6))
        im = ax.pcolormesh(XX, YY, Z, shading="auto")
        plt.colorbar(im, ax=ax, label="Probabilité (norm.)")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title(f"{args.region_name} — carte P_spatial\n{row['alert_time']:%Y-%m-%d %H:%M UTC}")

        # isoligne top X%
        thr = np.nanpercentile(Z, 100 - args.top_percent)
        cs = ax.contour(XX, YY, Z, levels=[thr], linewidths=1.2, colors="white")
        # Compatibilité matplotlib 3.7+ / 3.9+
        if hasattr(cs, "legend_elements"):
            label = f"top {args.top_percent:.0f}%"
            ax.legend([plt.Line2D([0], [0], color="white", lw=1.2)], [label], loc="upper left")


        ax.legend(loc="upper left")
        fpath = outdir / f"pspatial_{row['alert_time']:%Y%m%d_%H%M%S}.png"
        plt.savefig(fpath, dpi=args.dpi, bbox_inches="tight"); plt.close(fig)
        figs.append(str(fpath))

    # résumé JSON
    summary = {
        "region": args.region_name,
        "inputs": {
            "events_csv": args.events_csv,
            "alerts_csv": args.alerts_csv,
            "amp_col": args.amp_col
        },
        "kde": {
            "grid_res": args.grid_res,
            "bw_scale": args.bw_scale,
            "bbox": args.bbox
        },
        "n_alerts": int(len(a)),
        "figures": figs[:50]  # liste tronquée
    }
    with open(outdir / "pspatial_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] {len(figs)} cartes écrites dans {outdir}")
    print(f"[OK] Résumé : {outdir/'pspatial_summary.json'}")

if __name__ == "__main__":
    main()
