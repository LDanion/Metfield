#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auteur: L. Danion — ORCID 0009-0008-8733-8261

python calibrate_spatial_bias.py \
  --pairs-csv results_eval/japan/20251024_144957/spatial_accuracy_local/paired_distances.csv \
  --outdir results_eval/japan/20251024_144957/spatial_bias_calib \
  --region-name "Japon"


Calibrage du biais spatial moyen entre épicentres prédits et réels.

Entrée :
  paired_distances.csv (issu de evaluate_spatial_accuracy.py)
Sorties :
  - CSV avec colonnes corrigées lon_corr, lat_corr
  - JSON résumé avec Δlon, Δlat, gain de précision
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CAND_P_LON = ["lon_pred", "pred_lon", "plon", "lon_p", "lon"]
CAND_P_LAT = ["lat_pred", "pred_lat", "plat", "lat_p", "lat"]
CAND_R_LON = ["lon_real", "real_lon", "rlon", "lon_r"]
CAND_R_LAT = ["lat_real", "real_lat", "rlat", "lat_r"]
CAND_DIST  = ["dist_km", "distance_km", "km"]

def pick(df, candidates, role):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Colonne manquante pour {role} dans {df.columns.tolist()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-csv", required=True, help="CSV des paires prédiction↔réel")
    ap.add_argument("--outdir", default="spatial_bias_calib", help="Répertoire de sortie")
    ap.add_argument("--region-name", default="region", help="Nom de la région pour les figures")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.pairs_csv)
    col_plon = pick(df, CAND_P_LON, "lon prédite")
    col_plat = pick(df, CAND_P_LAT, "lat prédite")
    col_rlon = pick(df, CAND_R_LON, "lon réelle")
    col_rlat = pick(df, CAND_R_LAT, "lat réelle")
    col_dist = pick(df, CAND_DIST, "distance (km)")

    m = df[[col_plon, col_plat, col_rlon, col_rlat, col_dist]].dropna().copy()
    m.rename(columns={col_plon:"plon", col_plat:"plat",
                      col_rlon:"rlon", col_rlat:"rlat",
                      col_dist:"dist_km"}, inplace=True)

    # Calcul du biais moyen
    dlon = (m["rlon"] - m["plon"]).mean()
    dlat = (m["rlat"] - m["plat"]).mean()

    # Application de la correction
    m["lon_corr"] = m["plon"] + dlon
    m["lat_corr"] = m["plat"] + dlat

    # Distance recalculée (approx locale simple, sphère moyenne)
    R = 6371.0  # km
    def haversine(lat1, lon1, lat2, lon2):
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
        return 2*R*np.arcsin(np.sqrt(a))

    m["dist_corr_km"] = haversine(m["lat_corr"], m["lon_corr"], m["rlat"], m["rlon"])

    # Stats avant/après
    stats_before = {
        "mean": float(m["dist_km"].mean()),
        "median": float(m["dist_km"].median()),
        "p75": float(m["dist_km"].quantile(0.75)),
    }
    stats_after = {
        "mean": float(m["dist_corr_km"].mean()),
        "median": float(m["dist_corr_km"].median()),
        "p75": float(m["dist_corr_km"].quantile(0.75)),
    }

    # Graphique comparatif
    fig, ax = plt.subplots()
    ax.hist(m["dist_km"], bins=20, alpha=0.6, label="Avant calibration")
    ax.hist(m["dist_corr_km"], bins=20, alpha=0.6, label="Après calibration")
    ax.set_xlabel("Erreur (km)")
    ax.set_ylabel("Nombre de prédictions")
    ax.set_title(f"Effet du calibrage spatial — {args.region_name}")
    ax.legend()
    fig.savefig(outdir / "bias_correction_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Export CSV et JSON
    out_csv = outdir / "epicenters_corrected.csv"
    m.to_csv(out_csv, index=False)
    summary = {
        "region": args.region_name,
        "n": len(m),
        "bias_deg": {"dlon": float(dlon), "dlat": float(dlat)},
        "stats_before": stats_before,
        "stats_after": stats_after,
        "gain_mean_%": 100*(1 - stats_after["mean"]/stats_before["mean"]),
        "gain_median_%": 100*(1 - stats_after["median"]/stats_before["median"]),
        "figure": str(outdir / "bias_correction_hist.png"),
        "output_csv": str(out_csv),
    }
    with open(outdir / "bias_correction_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] Correction enregistrée :", out_csv)
    print("[OK] Δlon =", round(dlon, 3), "Δlat =", round(dlat, 3))
    print("[OK] Gain médian ≈", round(summary["gain_median_%"], 1), "%")

if __name__ == "__main__":
    main()
