#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""

Auteur: L. Danion — ORCID 0009-0008-8733-8261

python compare_spatial_gain.py \
  --before results_eval/japan/20251024_144957/spatial_accuracy_local/spatial_accuracy_summary.json \
  --after  results_eval/japan/20251024_144957/spatial_accuracy_calib/spatial_accuracy_summary.json \
  --outdir results_eval/japan/20251024_144957/spatial_gain \
  --region-name Japon

"""

import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import dedent

METRICS_DIST = ["mean_km", "median_km", "p25_km", "p75_km", "p90_km"]
METRICS_WITHIN = ["within_50km", "within_100km", "within_200km"]

def load_summary(path: Path) -> dict:
    """
    Charge un summary.json. Accepte soit :
      - le fichier JSON lui-même
      - un dossier contenant 'spatial_accuracy_summary.json'
    Gère les JSON qui ont un champ racine 'root'.
    """
    p = Path(path)
    if p.is_dir():
        # Cherche un summary standard dans le dossier
        candidates = [
            p / "spatial_accuracy_summary.json",
            p / "summary.json",
        ]
        for c in candidates:
            if c.exists():
                p = c
                break
    if not p.exists():
        raise FileNotFoundError(f"Summary introuvable : {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "root" in data and isinstance(data["root"], dict):
        data = data["root"]
    return data, p

def to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def build_rowmap(d: dict) -> dict:
    """
    Normalise les champs utiles depuis le summary.
    Ignore silencieusement ceux manquants.
    """
    out = {}
    for k in ["n_pred", *METRICS_DIST, *METRICS_WITHIN]:
        if k in d:
            out[k] = to_float(d[k])
    return out

def format_pct(x):
    if pd.isna(x):
        return ""
    s = f"{x:.1f}%"
    if x >= 0:
        s = "+" + s
    return s

def main():
    ap = argparse.ArgumentParser(
        description="Compare précision spatiale avant/après calibration (résumé + graphes)."
    )
    ap.add_argument("--before", required=True,
                    help="Chemin vers le summary 'AVANT' (fichier JSON ou dossier).")
    ap.add_argument("--after", required=True,
                    help="Chemin vers le summary 'APRES' (fichier JSON ou dossier).")
    ap.add_argument("--outdir", default="results_eval/spatial_gain",
                    help="Répertoire de sortie (CSV + figures).")
    ap.add_argument("--region-name", default="",
                    help="Nom de la région (pour les titres des figures).")
    args = ap.parse_args()

    # Chargement
    before_dict, before_path = load_summary(Path(args.before))
    after_dict,  after_path  = load_summary(Path(args.after))

    before = build_rowmap(before_dict)
    after  = build_rowmap(after_dict)

    # Tableau de comparaison
    all_keys = ["n_pred", *METRICS_DIST, *METRICS_WITHIN]
    df = pd.DataFrame({
        "metric": all_keys,
        "before": [before.get(k, float("nan")) for k in all_keys],
        "after":  [after.get(k, float("nan"))  for k in all_keys],
    }).set_index("metric")

    # Gains : pour les distances, plus petit = mieux (gain = 1 - after/before)
    gains = {}
    for k in METRICS_DIST:
        b, a = df.at[k, "before"], df.at[k, "after"]
        gains[k] = (1.0 - a / b) * 100.0 if (pd.notna(b) and pd.notna(a) and b > 0) else float("nan")

    # Gains : pour les “within”, plus grand = mieux (gain = after - before en points de pourcentage)
    for k in METRICS_WITHIN:
        b, a = df.at[k, "before"], df.at[k, "after"]
        if pd.notna(b) and pd.notna(a):
            # Si valeurs dans [0,1], on convertit en % pour lisibilité
            if (0 <= b <= 1) and (0 <= a <= 1):
                gains[k] = (a - b) * 100.0
            else:
                gains[k] = (a - b)
        else:
            gains[k] = float("nan")

    df["gain_pct"] = pd.Series(gains)

    # Sortie
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "spatial_gain_summary.csv"
    df.to_csv(csv_path)
    # JSON récapitulatif
    recap = {
        "region": args.region_name or "",
        "before_summary_path": str(before_path),
        "after_summary_path": str(after_path),
        "n_pred_before": int(before.get("n_pred", 0)) if not pd.isna(before.get("n_pred", float("nan"))) else None,
        "n_pred_after":  int(after.get("n_pred", 0))  if not pd.isna(after.get("n_pred", float("nan"))) else None,
        "gains": gains,
    }
    with open(outdir / "spatial_gain_summary.json", "w", encoding="utf-8") as f:
        json.dump(recap, f, ensure_ascii=False, indent=2)

    # --- Figures ---
    title_suffix = f" — {args.region_name}" if args.region_name else ""

    # 1) Barres pour les distances
    fig1 = plt.figure(figsize=(7, 4.5))
    ax1 = fig1.add_subplot(111)
    x = range(len(METRICS_DIST))
    before_vals = [df.at[k, "before"] for k in METRICS_DIST]
    after_vals  = [df.at[k, "after"]  for k in METRICS_DIST]
    width = 0.38
    ax1.bar([i - width/2 for i in x], before_vals, width, label="Before")
    ax1.bar([i + width/2 for i in x], after_vals,  width, label="After")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([k.replace("_km", "").upper() for k in METRICS_DIST])
    ax1.set_ylabel("Error (km)")
    ax1.set_title(f"Spatial Accuracy (distances){title_suffix}")
    ax1.legend()
    # Ajoute les gains en annotations
    for i, k in enumerate(METRICS_DIST):
        g = gains.get(k, float("nan"))
        if pd.notna(g):
            ax1.text(i, max(before_vals[i], after_vals[i]) * 1.02, format_pct(g),
                     ha="center", va="bottom", fontsize=9)
    fig1.tight_layout()
    fig1.savefig(outdir / "spatial_gain_distances.png", dpi=150)

    # 2) Barres pour les within
    fig2 = plt.figure(figsize=(7, 4.5))
    ax2 = fig2.add_subplot(111)
    x2 = range(len(METRICS_WITHIN))
    b2 = [df.at[k, "before"] for k in METRICS_WITHIN]
    a2 = [df.at[k, "after"]  for k in METRICS_WITHIN]

    # Convertit en pourcentage si nécessaire
    def as_percent(arr):
        if all((0 <= v <= 1) for v in arr if pd.notna(v)):
            return [v * 100 if pd.notna(v) else v for v in arr], True
        return arr, False

    b2p, scaled = as_percent(b2)
    a2p, _ = as_percent(a2)

    ax2.bar([i - width/2 for i in x2], b2p, width, label="Avant")
    ax2.bar([i + width/2 for i in x2], a2p, width, label="Après")
    ax2.set_xticks(list(x2))
    ax2.set_xticklabels([k.replace("within_", "≤ ").replace("km", " km") for k in METRICS_WITHIN])
    ax2.set_ylabel("Part (%)" if scaled else "Part")
    ax2.set_title(f"Couverture (rayons){title_suffix}")
    ax2.legend()
    # Annotations des gains
    for i, k in enumerate(METRICS_WITHIN):
        g = gains.get(k, float("nan"))
        if pd.notna(g):
            ax2.text(i, max(b2p[i], a2p[i]) * 1.02, format_pct(g),
                     ha="center", va="bottom", fontsize=9)
    fig2.tight_layout()
    fig2.savefig(outdir / "spatial_gain_within.png", dpi=150)

    # Résumé console
    print(dedent(f"""
    [OK] Résumé comparatif enregistré :
        - CSV : {csv_path}
        - FIG : {outdir / "spatial_gain_distances.png"}
        - FIG : {outdir / "spatial_gain_within.png"}
        - JSON: {outdir / "spatial_gain_summary.json"}

    Distances (km) — gain = 1 - après/avant :
        MEAN   : {format_pct(gains.get("mean_km", float("nan")))}
        MEDIAN : {format_pct(gains.get("median_km", float("nan")))}
        P25    : {format_pct(gains.get("p25_km", float("nan")))}
        P75    : {format_pct(gains.get("p75_km", float("nan")))}
        P90    : {format_pct(gains.get("p90_km", float("nan")))}

    Couverture — gain = (après - avant) en points :
        ≤ 50km : {format_pct(gains.get("within_50km", float("nan")))}
        ≤ 100km: {format_pct(gains.get("within_100km", float("nan")))}
        ≤ 200km: {format_pct(gains.get("within_200km", float("nan")))}
    """).strip())

if __name__ == "__main__":
    main()
