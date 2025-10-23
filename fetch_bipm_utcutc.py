#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Auteur: L. Danion — ORCID 0009-0008-8733-8261

python fetch_bipm_utcutc.py --from-year 1990 --to-year 1998 \
  --outdir runs/bipm --prefix utc9098 --parse --plot

fetch_bipm_utcutc_v2.py
-----------------------
Télécharge automatiquement les tableaux historiques UTC–UTC(lab) du BIPM
en testant plusieurs répertoires (UTCXYZ, UTC, TAIXY(Z), TAI, archive),
avec bascule ftp->https, puis (optionnel) PARSE les fichiers de type 'utc.90'
au format texte en CSV.

Sorties (si --parse est activé) :
- <outdir>/<prefix>_parsed_long.csv   (colonnes: date, mjd, lab, offset_us)
- <outdir>/<prefix>_parsed_wide.csv   (une colonne par labo)

Exemples :
  python fetch_bipm_utcutc_v2.py --from-year 1990 --to-year 1998 \
         --outdir runs/bipm --prefix utc9098 --parse --plot

  python fetch_bipm_utcutc_v2.py --from-year 1993 --to-year 1998 \
         --outdir runs/bipm --prefix tai9398

"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
import math
import json
import urllib.request
from urllib.error import URLError, HTTPError
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict

try:
    import pandas as pd
    import numpy as np
except Exception:
    pd = None
    np = None


BASES = [
    # ordre de préférence
    "https://webtai.bipm.org/ftp/pub/tai/scale/UTCXYZ/utc.{yy}",
    "https://webtai.bipm.org/ftp/pub/tai/scale/UTC/utc{yy}.ar",
    "https://webtai.bipm.org/ftp/pub/tai/scale/TAIXYZ/utc.{yy}",
    "https://webtai.bipm.org/ftp/pub/tai/scale/TAI/tai{yy}.ar",
    # (on évite UTCGPS qui n’est pas UTC-UTC(lab))
]

MJD0 = datetime(1858, 11, 17)  # origine MJD


def mjd_to_date(mjd: int) -> datetime:
    return MJD0 + timedelta(days=int(mjd))


def safe_mkdir(d: str):
    os.makedirs(d, exist_ok=True)


def try_download(url: str, timeout: float = 20.0) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except (HTTPError, URLError, TimeoutError):
        return None


def fetch_one_year(yy: int, outdir: str, prefix: str) -> dict:
    """
    Essaie de récupérer l’année yy (2 chiffres) via les différentes bases.
    Sauve le fichier binaire/txt tel quel dans <outdir>.
    Retourne un dict { 'year': YYYY, 'url':..., 'saved_as':..., 'ok':bool }
    """
    year_full = 1900 + yy if yy >= 70 else 2000 + yy if yy < 70 else 1900 + yy

    tried = []
    payload = None
    used_url = None
    # construit les URLs candidates
    for pattern in BASES:
        url = pattern.format(yy=f"{yy:02d}")
        data = try_download(url)
        tried.append(url)
        if data:
            payload = data
            used_url = url
            break

    result = {"year": year_full, "url": used_url, "tried": tried, "ok": bool(payload), "saved_as": None}
    if not payload:
        return result

    # nom de fichier local
    name = os.path.basename(used_url)
    if not name or "." not in name:
        name = f"utc{yy:02d}.dat"
    local = os.path.join(outdir, f"{prefix}_{name}")
    with open(local, "wb") as f:
        f.write(payload)
    result["saved_as"] = local
    return result


# ---------- PARSEUR POUR utc.90-like (texte) ----------

LAB_TOKEN = re.compile(r"\b[A-Z0-9]{1,5}\b")

def parse_utcxyz_text(raw: str) -> tuple[list[str], list[tuple[int, list[float]]]]:
    """
    Parse un fichier UTCXYZ 'utc.90' (texte).
    Retourne (labs, rows) où rows = list[(mjd, values)], values alignées sur labs.
    La table est parfois sur plusieurs lignes par MJD (blocs de 8 labos).
    0.0 => valeur manquante (on convertit en NaN plus tard).
    """
    lines = [ln.rstrip() for ln in raw.splitlines() if ln.strip()]
    # 1) récupérer la liste complète des labos en lisant les lignes d'en-tête
    labs = []
    header_done = False
    data_start_idx = None

    # Heuristique : on accumule tous les tokens MAJUSCULES courts rencontrés avant la 1re ligne commençant par un entier (MJD)
    for i, ln in enumerate(lines):
        if re.match(r"^\d{5,6}\b", ln.strip()):
            data_start_idx = i
            break
        # repérer des lignes listant des labos :
        toks = [t for t in LAB_TOKEN.findall(ln) if not t.isdigit()]
        # on ignore les mots typiques de phrases, ne garder que codes courts (AOS, PTB, OP, ... )
        for t in toks:
            if len(t) <= 5 and not t.startswith(("Unit", "MJD")):
                # éviter doublons
                if t not in labs:
                    labs.append(t)

    if data_start_idx is None:
        raise ValueError("Impossible de localiser le début des données (MJD).")

    # 2) lire les blocs de données
    # stratégie : chaque fois qu’on rencontre un MJD, on remplit les valeurs dans l’ordre des labs
    rows = []
    current_mjd = None
    current_vals = []
    expect_total = len(labs)

    def flush_row():
        # complète ou tronque à la longueur labs
        if current_mjd is None:
            return
        vals = current_vals[:expect_total]
        if len(vals) < expect_total:
            vals = vals + [math.nan] * (expect_total - len(vals))
        rows.append((current_mjd, vals.copy()))

    for ln in lines[data_start_idx:]:
        s = ln.strip()
        if not s:
            continue
        # si ligne commence par MJD -> nouvelle ligne
        m = re.match(r"^(\d{5,6})\b(.*)$", s)
        if m:
            # flush la précédente
            flush_row()
            current_mjd = int(m.group(1))
            current_vals = []
            rest = m.group(2).strip()
            parts = rest.split()
        else:
            parts = s.split()

        # accumulateur de nombres (certains fichiers ont des '0.0' à interpréter comme NaN optionnellement)
        for p in parts:
            # certains tokens sont non-numériques (artefacts) : ignorer
            try:
                v = float(p)
            except ValueError:
                continue
            # Conserver la convention : 0.0 => valeur disponible mais nulle ; si tu veux la traiter comme “manquante”, dé-commente la ligne suivante :
            # if abs(v) < 1e-15: v = float("nan")
            current_vals.append(v)

    # flush final
    flush_row()

    return labs, rows


def parse_files_if_possible(results: list[dict], outdir: str, prefix: str, plot: bool = False):
    """
    Essaie de parser tous les fichiers texte 'utc.**' récupérés.
    Concatène en deux CSV (long et wide).
    Ignore les fichiers binaires .ar ou TAI.
    """
    if pd is None or np is None:
        print("[parse] pandas/numpy non disponibles. Parsing sauté.")
        return

    all_long = []
    wide_map = OrderedDict()  # (mjd -> dict{lab: value})

    for rec in results:
        fp = rec.get("saved_as")
        if not rec.get("ok") or not fp:
            continue
        base = os.path.basename(fp).lower()
        # on ne parse que les noms de type 'utc.xx' sans extension .ar
        if not base.startswith("utc.") or base.endswith(".ar"):
            continue

        with open(fp, "r", encoding="latin-1", errors="ignore") as f:
            raw = f.read()

        try:
            labs, rows = parse_utcxyz_text(raw)
        except Exception as e:
            print(f"[parse] Échec de parsing {base}: {e}")
            continue

        # accumule
        for mjd, vals in rows:
            d = {lab: (vals[i] if i < len(vals) else float("nan")) for i, lab in enumerate(labs)}
            # "long"
            for lab, v in d.items():
                all_long.append(
                    {
                        "date": mjd_to_date(mjd).strftime("%Y-%m-%d"),
                        "mjd": mjd,
                        "lab": lab,
                        "offset_us": v,
                    }
                )
            # "wide"
            if mjd not in wide_map:
                wide_map[mjd] = d
            else:
                # complète les trous
                wide_map[mjd].update({k: wide_map[mjd].get(k, v) if not math.isnan(wide_map[mjd].get(k, math.nan)) else v
                                      for k, v in d.items()})

    if not all_long:
        print("[parse] Aucun tableau parsé. Vérifie que certains fichiers sont au format 'utc.xx' texte (pas '.ar').")
        return

    # DataFrames
    dfl = pd.DataFrame(all_long).sort_values(["mjd", "lab"])
    # wide
    records = []
    for mjd, dd in wide_map.items():
        row = {"mjd": mjd, "date": mjd_to_date(mjd).strftime("%Y-%m-%d")}
        row.update(dd)
        records.append(row)
    dfw = pd.DataFrame.from_records(records).sort_values("mjd")

    # 0.0 peut être considéré comme “pas de valeur” dans certains fichiers — si tu veux les mettre à NaN :
    # dfl.loc[np.isclose(dfl["offset_us"].values, 0.0), "offset_us"] = np.nan
    # for c in dfw.columns:
    #     if c not in ("mjd", "date"):
    #         dfw.loc[np.isclose(dfw[c].values, 0.0), c] = np.nan

    safe_mkdir(outdir)
    p_long = os.path.join(outdir, f"{prefix}_parsed_long.csv")
    p_wide = os.path.join(outdir, f"{prefix}_parsed_wide.csv")
    dfl.to_csv(p_long, index=False)
    dfw.to_csv(p_wide, index=False)
    print(f"[OK] CSV écrits :\n  - {p_long}\n  - {p_wide}")

    if plot:
        try:
            import matplotlib.pyplot as plt
            # petite vérif visuelle : variance/périodogramme sur quelques labos présents
            candidates = [c for c in dfw.columns if c not in ("mjd", "date")]
            keep = candidates[:6]
            if keep:
                plt.figure(figsize=(9, 4))
                for c in keep:
                    x = pd.to_numeric(dfw[c], errors="coerce")
                    x = x - np.nanmean(x)
                    plt.plot(dfw["mjd"], x, alpha=0.7, label=c)
                plt.xlabel("MJD")
                plt.ylabel("UTC - UTC(lab) [µs] (centré)")
                plt.title("Aperçu multi-lab (cru, centré)")
                plt.legend(ncol=3, fontsize=8)
                plt.tight_layout()
                outpng = os.path.join(outdir, f"{prefix}_quicklook.png")
                plt.savefig(outpng, dpi=150)
                print(f"[OK] Aperçu enregistré : {outpng}")
        except Exception as e:
            print(f"[plot] Skippé ({e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-year", type=int, required=True, help="année de début (ex. 1990)")
    ap.add_argument("--to-year", type=int, required=True, help="année de fin incluse (ex. 1998)")
    ap.add_argument("--outdir", default="runs/bipm")
    ap.add_argument("--prefix", default="bipm")
    ap.add_argument("--parse", action="store_true", help="parse les fichiers 'utc.xx' texte en CSV")
    ap.add_argument("--plot", action="store_true", help="fait un petit quicklook PNG si --parse")
    args = ap.parse_args()

    safe_mkdir(args.outdir)
    # années -> 2 chiffres (90..99) & (00..69) mappés intelligemment plus haut
    years = list(range(args.from_year, args.to_year + 1))
    res = []
    for y in years:
        if y < 1900:
            raise ValueError("Utilise des années à 4 chiffres (ex. 1990).")
        yy = y % 100
        r = fetch_one_year(yy, args.outdir, args.prefix)
        if r["ok"]:
            print(f"[OK] {y} -> {r['saved_as']}")
        else:
            print(f"[!!] {y} : aucun fichier disponible dans {len(r['tried'])} emplacements.")
        res.append(r)

    # Sauve un petit JSON récapitulatif
    with open(os.path.join(args.outdir, f"{args.prefix}_fetch_log.json"), "w") as f:
        json.dump(res, f, indent=2)

    # Parse optionnel
    if args.parse:
        parse_files_if_possible(res, args.outdir, args.prefix, plot=args.plot)


if __name__ == "__main__":
    main()
