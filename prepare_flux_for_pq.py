#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Auteur: L. Danion — ORCID 0009-0008-8733-8261

python prepare_flux_for_pq.py \
  --input solar_radio_flux.csv \
  --output solar_flux_prepared.csv \
  --date_col fluxdate \
  --flux_col fluxadjflux


Prépare un CSV propre pour les scripts P–Q.
Entrée prévue : solar_radio_flux.csv avec colonnes
fluxdate, fluxadjflux (Penticton F10.7). Fonctionne aussi en auto.
"""

import pandas as pd
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Fichier d'entrée (.csv ou .txt)")
    ap.add_argument("--output", default="solar_flux_prepared.csv", help="CSV de sortie")
    ap.add_argument("--date_col", default=None, help="Nom de la colonne date (optionnel)")
    ap.add_argument("--flux_col", default=None, help="Nom de la colonne flux (optionnel)")
    ap.add_argument("--sep", default=None, help="Séparateur (auto si None)")
    args = ap.parse_args()

    # Lecture (laisser pandas auto-détecter le séparateur si None)
    df = pd.read_csv(args.input, sep=args.sep, dtype=str, comment="#", engine="python")

    # Normalisation des noms de colonnes
    df.columns = [c.strip().lower() for c in df.columns]

    # Choix des colonnes
    date_col = (args.date_col.lower() if args.date_col
                else ("fluxdate" if "fluxdate" in df.columns
                      else ("date" if "date" in df.columns else None)))
    if date_col is None:
        raise ValueError("Impossible de trouver une colonne date (ex: fluxdate, date).")

    flux_col = (args.flux_col.lower() if args.flux_col
                else ("fluxadjflux" if "fluxadjflux" in df.columns
                      else ("flux" if "flux" in df.columns else None)))
    if flux_col is None:
        raise ValueError("Impossible de trouver une colonne flux (ex: fluxadjflux, flux).")

    print(f"✓ Date: {date_col} | ✓ Flux: {flux_col}")

    # Parsing des dates
    s = df[date_col].astype(str).str.strip()
    if s.str.match(r"^\d{8}$").all():  # format YYYYMMDD
        date = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    else:
        date = pd.to_datetime(s, infer_datetime_format=True, errors="coerce")

    yraw = pd.to_numeric(df[flux_col], errors="coerce")

    # Nettoyage + tri
    d = pd.DataFrame({"date": date, "flux": yraw}).dropna().sort_values("date")

    # Échantillonnage quotidien + interpolation légère (jusqu’à 7 j)
    d = (d.set_index("date").resample("1D").mean()
           .interpolate(limit=7, limit_direction="both")
           .reset_index())

    # Variables pour P–Q
    d["Y"] = (d["flux"] - d["flux"].mean()) / d["flux"].std(ddof=0)
    d["S"] = d["Y"].diff().fillna(0.0)
    d["time"] = (d["date"] - d["date"].iloc[0]).dt.total_seconds() / 86400.0

    # Export
    d[["time", "Y", "S", "date"]].to_csv(args.output, index=False)
    print(f"✅ Écrit: {args.output} ({len(d)} lignes)")

if __name__ == "__main__":
    main()
