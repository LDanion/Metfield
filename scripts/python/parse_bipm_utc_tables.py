#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Auteur: L. Danion — ORCID 0009-0008-8733-8261

python parse_bipm_utc_tables.py \
  --inputs runs/bipm/utc9098_utc.* \
  --out-prefix runs/bipm/utc9098

python bipm_pq_common_modulation.py \
  --input runs/bipm/utc9098_parsed_long.csv \
  --out-prefix bipm9098 \
  --min-n 20 \
  --band 1e-9 1e-4 \
  --plot


parse_bipm_utc_tables.py
Parsage robuste des fichiers BIPM 'utc.90'...'utc.98' (et similaires).
Extrait les tableaux "Local representations of UTC: values of [UTC - UTC(k)]"
et les empile en un seul jeu de données.

Sorties:
  <out_prefix>_parsed_long.csv  # colonnes: source, year, mjd, date_str, lab, offset_ns, offset_us
  <out_prefix>_parsed_wide.csv  # colonnes: source, year, mjd, date_str, <LAB1>, <LAB2>, ...

Usage:
  python parse_bipm_utc_tables.py --inputs runs/bipm/utc9098_utc.* --out-prefix runs/bipm/utc9098
"""

import re
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

MONTHS = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
          "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}

# Lignes repères
RE_HEADER_BLOCK = re.compile(r"UTC\s*-\s*UTC\(k\)", re.I)
RE_YEAR_LINE    = re.compile(r"^\s*(\d{4})\s*$")
RE_DATE_ROW     = re.compile(r"^\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})\s+(\d{5})\s+(.*)$")
RE_LABS_LINE    = re.compile(r"(?:\b[A-Z]{2,6}\b)(?:\s+\b[A-Z]{2,6}\b){4,}")  # >=5 codes labo sur la ligne

def parse_one_file(path: Path):
    txt = path.read_text(errors="ignore")
    lines = txt.splitlines()

    records_long = []
    records_wide = []

    i = 0
    current_labs = None
    current_year = None

    while i < len(lines):
        line = lines[i]

        # Début d'un bloc "UTC - UTC(k)"
        if RE_HEADER_BLOCK.search(line):
            current_labs = None
            current_year = None

            # Cherche une ligne année proche (souvent la ligne "1998")
            for j in range(i, min(i+15, len(lines))):
                m = RE_YEAR_LINE.match(lines[j].strip())
                if m:
                    current_year = int(m.group(1))
                    break

            # Cherche la ligne avec la liste de labos (juste après)
            for j in range(i, min(i+40, len(lines))):
                l = lines[j].rstrip()
                if RE_LABS_LINE.search(l) and ("Date" not in l) and ("MJD" not in l):
                    # Nettoie les éventuelles annotations (1), (2)
                    labs = [tok for tok in re.split(r"\s+", re.sub(r"\(\d+\)", "", l).strip()) if tok]
                    # Filtre quelques mots parasites éventuels
                    labs = [lab for lab in labs if lab not in ("UTC","Date","MJD","UTC(k)","0h","0hUTC","0h-UTC")]
                    current_labs = labs
                    break

            # Avance jusqu'aux lignes de données
            i += 1
            continue

        # Si nous avons une définition complète (labs) : tenter de lire des lignes données
        if current_labs:
            mrow = RE_DATE_ROW.match(line)
            if mrow:
                month, day, mjd, tail = mrow.groups()
                mjd = int(mjd)
                # Pour la date lisible
                y = current_year if current_year else 0
                date_str = f"{y:04d}-{MONTHS[month]:02d}-{int(day):02d}"

                # Découpe les colonnes: certaines pages ont des colonnes alignées avec espaces multiples.
                cols = re.split(r"\s+", tail.strip())
                # On s'attend à len(cols) == len(labs) ; si pas le cas, on tente un rattrapage
                if len(cols) != len(current_labs):
                    # Rattacher d'éventuels '-' manquants en fin/ligne vide
                    if len(cols) < len(current_labs):
                        cols = cols + ["-"]*(len(current_labs)-len(cols))
                    else:
                        cols = cols[:len(current_labs)]

                vals_ns = []
                for c in cols:
                    if c in ("-", "", "—"):
                        vals_ns.append(np.nan)
                    else:
                        try:
                            vals_ns.append(float(c))
                        except ValueError:
                            # cas exotiques: nombres collés; tente un split sur , ou ;
                            c2 = re.split(r"[;,]", c)[0]
                            try:
                                vals_ns.append(float(c2))
                            except Exception:
                                vals_ns.append(np.nan)

                # Enregistre format wide
                rec_w = {"source": path.name, "year": current_year, "mjd": mjd, "date_str": date_str}
                for lab, v in zip(current_labs, vals_ns):
                    rec_w[lab] = v
                records_wide.append(rec_w)

                # Enregistre format long
                for lab, v in zip(current_labs, vals_ns):
                    records_long.append({
                        "source": path.name,
                        "year": current_year,
                        "mjd": mjd,
                        "date_str": date_str,
                        "lab": lab,
                        "offset_ns": v,
                        "offset_us": (v/1000.0) if pd.notna(v) else np.nan
                    })

        i += 1

    df_long = pd.DataFrame.from_records(records_long)
    df_wide = pd.DataFrame.from_records(records_wide)

    return df_long, df_wide

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Fichiers à parser (glob développé par le shell)")
    ap.add_argument("--out-prefix", required=True, help="Préfixe des CSV de sortie")
    args = ap.parse_args()

    all_long = []
    all_wide = []

    for pat in args.inputs:
        for p in sorted(Path().glob(pat) if any(ch in pat for ch in "*?[]") else [Path(pat)]):
            if not p.exists():
                print(f"[skip] {p} (absent)")
                continue
            try:
                dfL, dfW = parse_one_file(p)
                print(f"[ok] {p.name}: {len(dfW)} lignes (wide), {len(dfL)} (long)")
                all_long.append(dfL)
                all_wide.append(dfW)
            except Exception as e:
                print(f"[warn] {p.name}: échec du parse: {e}")

    if not all_long:
        raise SystemExit("Aucun tableau parsé. Vérifie le format des fichiers.")

    out_long = pd.concat(all_long, ignore_index=True).sort_values(["mjd","lab"])
    out_wide = pd.concat(all_wide, ignore_index=True).sort_values(["mjd"])

    out_long.to_csv(f"{args.out_prefix}_parsed_long.csv", index=False)
    out_wide.to_csv(f"{args.out_prefix}_parsed_wide.csv", index=False)

    # Petit diagnostic: nombre de points par labo
    counts = out_long.groupby("lab")["offset_ns"].count().sort_values(ascending=False)
    print("\n[Labs & #points]\n", counts)
    print(f"\n✅ Fichiers écrits:\n - {args.out_prefix}_parsed_long.csv\n - {args.out_prefix}_parsed_wide.csv")

if __name__ == "__main__":
    main()
