#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Auteur: L. Danion — ORCID 0009-0008-8733-8261

python evaluate_spatial_accuracy.py \
  --pred-csv     results_eval/japan/20251024_144957/p_spatial/epicenters_pred_geo.csv \
  --real-csv     data/japan/usgs_japan_2010_2024_M5.5.csv \
  --matches-csv  results_eval/japan/20251024_144957/matches.csv \
  --outdir       results_eval/japan/20251024_144957/spatial_accuracy \
  --time-window  30


evaluate_spatial_accuracy.py
Calcule la précision spatiale (km) des épicentres prédits vs épicentres réels.

Entrées:
  - CSV prédictions: colonnes ~ ["alert_time", "lat", "lon"] (noms tolérants)
  - CSV catalogue réel: colonnes ~ ["time", "latitude", "longitude"] (+ mag optionnel)
  - (optionnel) matches.csv pour contraindre l’appariement temporel

Sorties:
  - <outdir>/paired_distances.csv
  - <outdir>/spatial_accuracy_summary.json
  - <outdir>/distance_hist.png
  - <outdir>/distance_cdf.png
"""

import argparse
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- utils robustes ----------
CAND_LAT = ["lat", "latitude", "y", "Lat", "Latitude"]
CAND_LON = ["lon", "longitude", "x", "Lon", "Longitude"]
CAND_TIME_PRED = ["alert_time", "time", "date", "timestamp", "AlertTime"]
CAND_TIME_REAL = ["time", "event_time", "origin_time", "Date", "Time"]
CAND_MAG = ["mag", "magnitude", "Mw", "ML", "Mb"]

def find_col(df, candidates, required=True):
    low = {c.lower().strip(): c for c in df.columns}
    for name in candidates:
        if name.lower() in low:
            return low[name.lower()]
    if required:
        raise ValueError(f"Colonnes attendues parmi {candidates}, trouvé: {list(df.columns)}")
    return None

def parse_time(series):
    return pd.to_datetime(series, errors="coerce", utc=True)

def haversine_km(lat1, lon1, lat2, lon2):
    # toutes entrées en degrés -> km
    R = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R*c

def percent_within(distances, km):
    return float((distances <= km).mean()) if len(distances) else 0.0

# ---------- cœur ----------
def load_predictions(path_pred):
    df = pd.read_csv(path_pred)
    lat = find_col(df, CAND_LAT)
    lon = find_col(df, CAND_LON)
    tcol = find_col(df, CAND_TIME_PRED)
    out = df[[tcol, lat, lon]].copy()
    out.columns = ["alert_time", "lat", "lon"]
    out["alert_time"] = parse_time(out["alert_time"])
    out = out.dropna(subset=["alert_time", "lat", "lon"])
    return out.sort_values("alert_time")

def load_reals(path_real):
    df = pd.read_csv(path_real)
    lat = find_col(df, CAND_LAT)
    lon = find_col(df, CAND_LON)
    tcol = find_col(df, CAND_TIME_REAL)
    cols = [tcol, lat, lon]
    magcol = find_col(df, CAND_MAG, required=False)
    if magcol: cols.append(magcol)
    out = df[cols].copy()
    out.columns = ["time", "lat", "lon"] if magcol is None else ["time", "lat", "lon", "mag"]
    out["time"] = parse_time(out["time"])
    out = out.dropna(subset=["time", "lat", "lon"])
    return out.sort_values("time")

def load_matches_optional(path_matches):
    if not path_matches:
        return None
    m = pd.read_csv(path_matches)
    # tolère noms
    at = find_col(m, ["alert_time", "time", "alertTime"])
    et = find_col(m, ["event_time", "time", "origin_time"])
    keep = [at, et]
    if set(["mag","magnitude"]).intersection(m.columns):
        keep.append(find_col(m, CAND_MAG, required=False))
    out = m[keep].copy()
    # standardise
    out.columns = ["alert_time", "event_time"] if len(keep)==2 else ["alert_time","event_time","mag"]
    out["alert_time"] = parse_time(out["alert_time"])
    out["event_time"] = parse_time(out["event_time"])
    return out.dropna(subset=["alert_time", "event_time"]).sort_values("alert_time")

def pair_by_nearest_time(preds, reals, time_window_days, matches=None):
    """
    Si matches est fourni, on associe chaque alert_time à l'event_time (avec merge_asof).
    Sinon: on associe chaque alert_time au real 'time' le plus proche, avec tolérance.
    """
    tol = pd.Timedelta(days=time_window_days)

    if matches is not None:
        # associe event_time -> réel le plus proche
        # 1) rattache event_time aux preds par alert_time
        pm = pd.merge_asof(
            preds.sort_values("alert_time"),
            matches.sort_values("alert_time"),
            on="alert_time", direction="nearest", tolerance=tol
        )
        pm = pm.dropna(subset=["event_time"])
        # 2) rattache coordonnées réelles en cherchant time proche de event_time
        paired = pd.merge_asof(
            pm.sort_values("event_time"),
            reals.sort_values("time"),
            left_on="event_time", right_on="time",
            direction="nearest", tolerance=tol, suffixes=("","_real")
        )
        paired = paired.dropna(subset=["time"])  # ceux qui n'ont rien trouvé
        # renomme clair
        paired = paired.rename(columns={
            "lat": "pred_lat", "lon": "pred_lon",
            "time": "real_time", "lat_real":"real_lat", "lon_real":"real_lon"
        })
        if "real_lat" not in paired.columns:
            # cas où pandas n'a pas créé _real (si noms différents déjà gérés)
            paired = paired.rename(columns={"lat":"real_lat", "lon":"real_lon"})
        return paired
    else:
        # associe alerte -> événement réel le plus proche
        paired = pd.merge_asof(
            preds.sort_values("alert_time"),
            reals.sort_values("time"),
            left_on="alert_time", right_on="time",
            direction="nearest", tolerance=tol, suffixes=("","_real")
        ).dropna(subset=["time"])
        paired = paired.rename(columns={
            "lat":"pred_lat", "lon":"pred_lon",
            "time":"real_time", "lat_real":"real_lat", "lon_real":"real_lon"
        })
        if "real_lat" not in paired.columns:
            paired = paired.rename(columns={"lat":"real_lat", "lon":"real_lon"})
        return paired

def summarize_and_plot(paired, outdir):
    outdir.mkdir(parents=True, exist_ok=True)

    # distances
    dkm = haversine_km(paired["pred_lat"].values, paired["pred_lon"].values,
                       paired["real_lat"].values, paired["real_lon"].values)
    paired = paired.copy()
    paired["dist_km"] = dkm

    # export pairs
    cols = ["alert_time","pred_lat","pred_lon","real_time","real_lat","real_lon","dist_km"]
    if "mag" in paired.columns:
        # si ‘mag’ réel a survécu
        # (selon les merges il peut se nommer différemment)
        cols.insert(4, "mag")
    paired[cols].to_csv(outdir / "paired_distances.csv", index=False)

    # stats
    stats = {
        "n_pred": int(paired.shape[0]),
        "mean_km": float(np.mean(dkm)) if len(dkm) else None,
        "median_km": float(np.median(dkm)) if len(dkm) else None,
        "p25_km": float(np.percentile(dkm, 25)) if len(dkm) else None,
        "p75_km": float(np.percentile(dkm, 75)) if len(dkm) else None,
        "p90_km": float(np.percentile(dkm, 90)) if len(dkm) else None,
        "within_50km": percent_within(dkm, 50),
        "within_100km": percent_within(dkm, 100),
        "within_200km": percent_within(dkm, 200),
    }

    # histogramme
    plt.figure(figsize=(7,5))
    plt.hist(dkm, bins=40)
    plt.xlabel("Erreur spatiale (km)")
    plt.ylabel("Nombre de prédictions")
    plt.title("Histogramme des distances prédiction↔réel (km)")
    plt.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig(outdir / "distance_hist.png", dpi=150)
    plt.close()

    # CDF
    xs = np.sort(dkm)
    ys = np.arange(1, len(xs)+1) / len(xs)
    plt.figure(figsize=(7,5))
    plt.plot(xs, ys)
    for thr in [50, 100, 200]:
        yv = percent_within(dkm, thr)
        plt.axvline(thr, ls="--", alpha=.4)
        plt.text(thr, yv, f" {int(round(yv*100))}% ≤ {thr} km", va="bottom")
    plt.xlabel("Erreur spatiale (km)")
    plt.ylabel("CDF")
    plt.title("CDF des distances prédiction↔réel")
    plt.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig(outdir / "distance_cdf.png", dpi=150)
    plt.close()

    # résumé JSON
    (outdir / "spatial_accuracy_summary.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False)
    )

    return stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", required=True, help="CSV épicentres prédits (alert_time, lat, lon)")
    ap.add_argument("--real-csv", required=True, help="CSV catalogue réel (time, latitude, longitude[, mag])")
    ap.add_argument("--outdir", required=True, help="Dossier de sortie")
    ap.add_argument("--matches-csv", default=None, help="(Option) matches pour contraindre l’appariement")
    ap.add_argument("--time-window", type=int, default=30, help="Tolérance d’appariement en jours (def=30)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    preds = load_predictions(args.pred_csv)
    reals = load_reals(args.real_csv)
    matches = load_matches_optional(args.matches_csv)

    paired = pair_by_nearest_time(preds, reals, args.time_window, matches=matches)

    stats = summarize_and_plot(paired, outdir)

    print("[OK] Résumé:", json.dumps(stats, ensure_ascii=False))
    print(f"[OUT] {outdir/'paired_distances.csv'}")
    print(f"[OUT] {outdir/'spatial_accuracy_summary.json'}")
    print(f"[OUT] {outdir/'distance_hist.png'}")
    print(f"[OUT] {outdir/'distance_cdf.png'}")

if __name__ == "__main__":
    main()
