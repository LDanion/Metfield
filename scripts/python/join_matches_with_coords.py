#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auteur: L. Danion — ORCID 0009-0008-8733-8261

python join_matches_with_coords.py \
  --events-csv data/japan/usgs_japan_2010_2024_M5.5.csv \
  --matches-csv results_eval/japan/20251024_144957/matches.csv \
  --out-csv results_eval/japan/20251024_144957/matches_with_coords.csv


"""

import argparse, sys
import pandas as pd
import numpy as np

def parse_time(s):
    if pd.isna(s):
        return pd.NaT
    return pd.to_datetime(str(s), utc=True, errors="coerce")

def main():
    ap = argparse.ArgumentParser(description="Joindre coordonnées (lat/lon/depth) aux matches par temps d'événement.")
    ap.add_argument("--events-csv", required=True, help="Catalogue événements (USGS/EMSC...)")
    ap.add_argument("--matches-csv", required=True, help="Fichier matches.csv (alert_time,event_time,mag,lead_days,...)")
    ap.add_argument("--out-csv", required=True, help="Sortie matches_with_coords.csv")
    ap.add_argument("--time-col-events", default="time", help="Nom de la colonne temps dans events (def: time)")
    ap.add_argument("--event-id-col", default=None, help="Colonne ID évènement si dispo (def: None)")
    ap.add_argument("--tol-sec", type=int, default=3600, help="Tolérance d'association par temps (sec) si pas d'ID exact (def: 3600)")
    args = ap.parse_args()

    ev = pd.read_csv(args.events_csv)
    m  = pd.read_csv(args.matches_csv)

    # Harmonise colonnes standard
    # temps
    te = args.time_col_events if args.time_col_events in ev.columns else "time"
    if te not in ev.columns:
        print(f"[ERR] Colonne temps '{te}' absente du catalogue.", file=sys.stderr); sys.exit(2)
    ev["_t"] = ev[te].apply(parse_time)

    tm = "event_time" if "event_time" in m.columns else "time"
    if tm not in m.columns:
        print(f"[ERR] matches.csv doit contenir 'event_time' (ou 'time').", file=sys.stderr); sys.exit(2)
    m["_t"] = m[tm].apply(parse_time)

    # lat/lon/depth/mag
    latcol = next((c for c in ev.columns if c.lower() in ("latitude","lat")), None)
    loncol = next((c for c in ev.columns if c.lower() in ("longitude","lon")), None)
    depcol = next((c for c in ev.columns if c.lower() in ("depth","dep")), None)
    magcol = next((c for c in ev.columns if c.lower() in ("mag","magnitude")), None)
    for c,name in [(latcol,"lat"),(loncol,"lon")]:
        if c is None:
            print(f"[ERR] Colonne {name} absente du catalogue.", file=sys.stderr); sys.exit(2)

    keep = ["_t"]
    rename = {}
    keep.append(latcol); rename[latcol]="lat"
    keep.append(loncol); rename[loncol]="lon"
    if depcol: keep.append(depcol); rename[depcol]="depth"
    if magcol: keep.append(magcol); rename[magcol]="mag_cat"
    if args.event_id_col and args.event_id_col in ev.columns:
        keep.append(args.event_id_col); rename[args.event_id_col]="event_id"

    ev2 = ev[keep].rename(columns=rename).dropna(subset=["_t"]).sort_values("_t").reset_index(drop=True)
    m2  = m.copy()

    # Cas 1 : on a un ID évènement dans matches + events → jointure directe
    if "event_id" in ev2.columns and "event_id" in m2.columns:
        out = m2.merge(ev2.drop_duplicates("event_id"), on="event_id", how="left")
    else:
        # Cas 2 : appariement par temps le plus proche (tolérance)
        # index temps pour recherche rapide
        ev2 = ev2.set_index("_t").sort_index()
        m2["_t"]=pd.to_datetime(m2["_t"], utc=True, errors="coerce")
        nearest_idx = ev2.index.searchsorted(m2["_t"])
        candidates = []
        for i, t in enumerate(m2["_t"]):
            j = nearest_idx[i]
            best = None; best_dt = None
            for k in (j-1,j):
                if 0 <= k < len(ev2.index):
                    tt = ev2.index[k]
                    dt = abs((tt - t).total_seconds()) if (pd.notna(t) and pd.notna(tt)) else np.inf
                    if best is None or dt < best_dt:
                        best = ev2.iloc[k]
                        best_dt = dt
            if best is not None and best_dt is not None and best_dt <= args.tol_sec:
                rec = dict(best)
                rec["_t_match"]=best.name
            else:
                rec = {"lat":np.nan,"lon":np.nan,"depth":np.nan if "depth" in ev2.columns else np.nan,
                       "mag_cat":np.nan, "_t_match":pd.NaT}
            candidates.append(rec)
        add = pd.DataFrame(candidates)
        out = pd.concat([m2.reset_index(drop=True), add.reset_index(drop=True)], axis=1)

    out.to_csv(args.out_csv, index=False)
    print(f"[OK] Écrit : {args.out_csv}")

if __name__ == "__main__":
    main()
