#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Auteur: L. Danion — ORCID 0009-0008-8733-8261

# Pγ = 5760 s ± 60 s (exemple), 2×P0 = 2×2548 s = 5096 s ± 20 s
python commensurability_test.py \
  --Pa 5760 --Pa_err 60 \
  --Pb 5096 --Pb_err 20 \
  --Nmax 6 --Nsamp 20000 --mode gauss \
  --tol_ppm 3000 --Nsurr 5000


python commensurability_test.py \
  --Pa 5000 --Pa_err 10 \
  --Pb 5096 --Pb_err 20 \
  --Nmax 6 --Nsamp 20000 --mode gauss \
  --tol_ppm 3000 --Nsurr 5000

# P0 ≈ 2548 ± 5 s et P2 ≈ 5092 ± 10 s
python commensurability_test.py \
  --Pa 5092 --Pa_err 10 \
  --Pb 2548 --Pb_err 5 \
  --Nmax 6 --Nsamp 20000 --mode gauss \
  --tol_ppm 3000 --Nsurr 5000


Test de commensurabilité rationnelle entre périodes P_a et P_b avec incertitudes.
- Monte-Carlo (Nsamp tirages) des deux périodes (gaussien ou uniforme),
- pour chaque tirage, on cherche le meilleur ratio m/n (1≤m,n≤Nmax)
  qui minimise | m*P_a - n*P_b | / (n*P_b),
- on accumule l'erreur relative (ppm) et le couple (m,n) optimal,
- on calcule une p-value "chance d'obtenir <= tol_ppm sans relation" via permutation.

Sorties : résumé console + CSV optionnel des meilleures réalisations.

"""

import argparse, math, itertools, numpy as np
import pandas as pd

def draw_vals(mu, err, mode, rng):
    if err <= 0:
        return np.array([mu])
    if mode == "gauss":
        return rng.normal(mu, err)
    elif mode == "uniform":
        return rng.uniform(mu-err, mu+err)
    else:
        raise ValueError("mode must be 'gauss' or 'uniform'")

def best_ratio(Pa, Pb, Nmax):
    # minimise |m*Pa - n*Pb| / (n*Pb)
    best = None
    best_ppm = np.inf
    best_mn = (None, None)
    for m in range(1, Nmax+1):
        for n in range(1, Nmax+1):
            num = abs(m*Pa - n*Pb)
            den = abs(n*Pb) if n*Pb != 0 else 1.0
            rel = (num/den)*1e6  # ppm
            if rel < best_ppm:
                best_ppm = rel
                best_mn = (m, n)
                best = (m*Pa, n*Pb)
    return best_ppm, best_mn, best

def run_once(mu_a, sig_a, mu_b, sig_b, Nmax, rng, mode):
    Pa = float(draw_vals(mu_a, sig_a, mode, rng))
    Pb = float(draw_vals(mu_b, sig_b, mode, rng))
    ppm, (m,n), _ = best_ratio(Pa, Pb, Nmax)
    return ppm, m, n, Pa, Pb

def pvalue_by_permutation(samples_ppm, Nsurr=10000, rng=None):
    # "p au hasard" ~ on remélange Pa et Pb indépendamment
    # Construisons un surrogate en mélangeant les tirages entre A et B.
    rng = np.random.default_rng(None if rng is None else rng)
    A = samples_ppm["Pa"].to_numpy()
    B = samples_ppm["Pb"].to_numpy()
    K = len(A)
    obs = np.nanmedian(samples_ppm["ppm"].to_numpy())
    better = 0
    for _ in range(Nsurr):
        idx = rng.integers(0, K, size=K)
        A_s = A[idx]  # A mélangé
        ppm_surr = []
        for a, b in zip(A_s, B):
            ppm,_,_ = best_ratio(a, b, int(samples_ppm.attrs.get("Nmax", 6)))
            ppm_surr.append(ppm)
        if np.nanmedian(ppm_surr) <= obs:
            better += 1
    return (better+1)/(Nsurr+1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Pa", type=float, required=True, help="Période A (s), ex: 2548")
    ap.add_argument("--Pa_err", type=float, default=0.0, help="incertitude (1σ si gauss)")
    ap.add_argument("--Pb", type=float, required=True, help="Période B (s), ex: 5760")
    ap.add_argument("--Pb_err", type=float, default=0.0, help="incertitude (1σ si gauss)")
    ap.add_argument("--Nmax", type=int, default=6, help="bornes m,n ∈ [1..Nmax]")
    ap.add_argument("--Nsamp", type=int, default=20000, help="tirages Monte-Carlo")
    ap.add_argument("--mode", choices=["gauss","uniform"], default="gauss")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default="", help="chemin CSV pour enregistrer les meilleurs m/n")
    ap.add_argument("--tol_ppm", type=float, default=3000.0, help="tolérance pour %hits")
    ap.add_argument("--Nsurr", type=int, default=5000, help="surrogates pour p-value permutation")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Monte-Carlo
    recs = []
    for _ in range(args.Nsamp):
        ppm, m, n, Pa, Pb = run_once(args.Pa, args.Pa_err, args.Pb, args.Pb_err,
                                     args.Nmax, rng, args.mode)
        recs.append((ppm, m, n, Pa, Pb))
    df = pd.DataFrame(recs, columns=["ppm","m","n","Pa","Pb"])
    df.attrs["Nmax"] = args.Nmax

    # Statistiques
    med_ppm = float(np.nanmedian(df["ppm"]))
    p05, p95 = np.nanpercentile(df["ppm"], [5,95])
    # mode (m,n)
    mn_counts = df.groupby(["m","n"]).size().sort_values(ascending=False)
    top_mn = mn_counts.index[0]
    top_frac = mn_counts.iloc[0] / len(df)

    # hits sous tolérance
    hits = float((df["ppm"] <= args.tol_ppm).mean())

    # p-value par permutation/mélange
    p_perm = pvalue_by_permutation(df, Nsurr=args.Nsurr, rng=args.seed)

    print("\n=== Test de commensurabilité m/n ===")
    print(f"P_a = {args.Pa} ± {args.Pa_err} s   |   P_b = {args.Pb} ± {args.Pb_err} s")
    print(f"Nmax = {args.Nmax}   Nsamp = {args.Nsamp}   mode = {args.mode}")
    print(f"ppm (median) = {med_ppm:.1f}  [5–95%] = [{p05:.1f}, {p95:.1f}]")
    print(f"meilleur (m,n) le plus fréquent = {top_mn}  (fraction {top_frac:.3f})")
    print(f"fraction ppm ≤ {args.tol_ppm:.0f} = {hits:.3f}")
    print(f"p_perm (surrogate) ≈ {p_perm:.4g}")

    if args.csv:
        df.to_csv(args.csv, index=False)
        print("→ CSV écrit :", args.csv)

if __name__ == "__main__":
    main()
