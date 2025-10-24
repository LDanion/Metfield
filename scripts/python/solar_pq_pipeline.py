#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auteur: L. Danion — ORCID 0009-0008-8733-8261

python solar_pq_pipeline.py --csv solar_radio_flux.csv --out penticton_pq

"""

import argparse, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import welch, csd, detrend

# --------- Modèle P–Q (cadence + mémoire) ---------
def pq_model(t, theta):
    gamma, lam, phi, a, b = theta
    # pilote (cadence) : P(t)
    P = np.sin(2*np.pi*gamma*t + phi)
    # mémoire (réponse lente) : Q(t) = (1 - e^{-λt}) * P(t)
    Q = (1.0 - np.exp(-lam*np.maximum(t - t.min(), 0.0))) * P
    return a*P + b*Q, P, Q

def residuals(theta, t, y):
    yhat, _, _ = pq_model(t, theta)
    return yhat - y

# --------- Lecture & préparation ---------
def load_and_prepare(path):
    # Essaie d’abord une lecture simple CSV virgule
    df = pd.read_csv(path, dtype=str)

    # Noms de colonnes tolérés
    # date/temps
    date_cols = [c for c in df.columns if c.lower() in ("date","time","datetime","fluxdate")]
    if not date_cols:
        raise ValueError("Aucune colonne date/time trouvée (attendues: date, time, datetime, fluxdate).")
    date_col = date_cols[0]

    # flux ajusté (recommandé), sinon flux obs
    flux_cols_pref = [c for c in df.columns if c.lower() in ("fluxadjflux","adjflux","adjustedflux")]
    flux_cols_alt  = [c for c in df.columns if c.lower() in ("fluxobsflux","flux","obsflux")]
    if flux_cols_pref:
        flux_col = flux_cols_pref[0]
    elif flux_cols_alt:
        flux_col = flux_cols_alt[0]
    else:
        # Dernier recours : toute colonne numérique hors date
        num_cols = [c for c in df.columns if c != date_col]
        # essaie de caster et garde la 1re qui marche
        flux_col = None
        for c in num_cols:
            try:
                pd.to_numeric(df[c])
                flux_col = c; break
            except Exception:
                pass
        if flux_col is None:
            raise ValueError("Aucune colonne de flux numérique détectée.")

    # Parse date/temps robuste
    s = df[date_col].str.strip()
    # Si chaîne de type YYYYMMDD -> parse explicite, sinon parse automatique
    if s.str.match(r"^\d{8}$").all():
        dt = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    else:
        dt = pd.to_datetime(s, infer_datetime_format=True, errors="coerce")
    if dt.isna().all():
        raise ValueError("Impossible de parser les dates du fichier (colonne: %s)." % date_col)

    df = pd.DataFrame({"date": dt, "flux": pd.to_numeric(df[flux_col], errors="coerce")}).dropna()
    df = df.sort_values("date").drop_duplicates("date")

    # Option : re-échantillonnage quotidien si nécessaire
    df = df.set_index("date").resample("1D").mean().interpolate(limit=7, limit_direction="both").reset_index()

    # Normalisation (centrage-réduction)
    y = df["flux"]
    Y = (y - y.mean()) / y.std(ddof=0)

    # Driver S(t) = dY/dt (différence 1 jour)
    S = Y.diff().fillna(0.0)

    # Axe temps en jours (float)
    t = (df["date"] - df["date"].iloc[0]).dt.total_seconds() / 86400.0

    out = pd.DataFrame({"time": t.values, "Y": Y.values, "S": S.values, "date": df["date"]})
    return out

# --------- Ajustement P–Q ---------
def fit_pq(df, f_guess=None):
    t = df["time"].values.astype(float)
    y = df["Y"].values.astype(float)

    # Estimation initiale de la fréquence gamma (période ~ 27 jours par exemple)
    if f_guess is None:
        # FFT grossière
        n = len(y)
        if n < 32:
            f_guess = 1/27.0
        else:
            y0 = y - y.mean()
            freqs = np.fft.rfftfreq(n, d=(t[1]-t[0]))
            psd = np.abs(np.fft.rfft(y0))**2
            # on évite f=0
            k = np.argmax(psd[1:]) + 1
            f_guess = max(freqs[k], 1/200.0)  # borne basse 200 j
    # Initialisation prudente
    theta0 = np.array([f_guess, 0.05, 0.0, 0.5, 0.5])  # [γ, λ, φ, a, b]
    bounds = ([1/4000, 0.0, -2*np.pi, -5, -5], [1.0, 1.0, 2*np.pi, 5, 5])

    res = least_squares(residuals, theta0, bounds=bounds, args=(t, y), verbose=0, max_nfev=2000)
    theta_hat = res.x
    yhat, P, Q = pq_model(t, theta_hat)

    rmse = np.sqrt(np.mean((y - yhat)**2))
    r2 = 1 - np.sum((y - yhat)**2) / np.sum((y - y.mean())**2)

    return theta_hat, yhat, P, Q, rmse, r2

# --------- Spectres & cohérence ---------
def spectra_and_coherence(df, yhat, out_prefix):
    t = df["time"].values
    y = df["Y"].values

    # Detrend doux pour spectres
    y_d = detrend(y, type="linear")
    yh_d = detrend(yhat, type="linear")

    dt = np.median(np.diff(t))
    fs = 1.0 / dt  # échantillonnage en 1/jour

    n = len(y)
    nperseg = int(min( max(64, n//8), 2048 ))
    noverlap = nperseg//2

    f_psd, Pyy = welch(y_d, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend=False)
    _,   Pyh = welch(yh_d, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend=False)
    f_coh, Pxy = csd(y_d, yh_d, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend=False)
    _,    Pxx  = welch(y_d,  fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend=False)
    _,    Pyyh = welch(yh_d, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend=False)

    # cohérence |Pxy|^2 / (Pxx*Pyyh)
    C = np.abs(Pxy)**2 / (Pxx * Pyyh + 1e-15)

    # --- Figures ---
    # 1) Observé vs fit
    fig1, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(t, y, label="Y (z-score)", lw=1.5)
    ax1.plot(t, yhat, label="PQ fit", lw=1.5, alpha=0.9)
    ax1.set_xlabel("time [days]"); ax1.set_ylabel("normalized amplitude")
    ax1.set_title("Observed vs PQ fit")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(f"{out_prefix}_fit.png", dpi=200)

    # 2) Spectres de puissance (log–log)
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.loglog(f_psd, Pyy, label="PSD(Y)")
    ax2.loglog(f_psd, Pyh, label="PSD(PQ)")
    ax2.set_xlabel("frequency [1/day]"); ax2.set_ylabel("power")
    ax2.set_title("Power spectra")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(f"{out_prefix}_psd.png", dpi=200)

    # 3) Cohérence
    fig3, ax3 = plt.subplots(figsize=(8,4))
    ax3.semilogx(f_coh, np.clip(C, 0, 1))
    ax3.set_ylim(0,1.02)
    ax3.set_xlabel("frequency [1/day]"); ax3.set_ylabel("coherence")
    ax3.set_title("Coherence: Y vs PQ")
    fig3.tight_layout()
    fig3.savefig(f"{out_prefix}_coh.png", dpi=200)

    plt.close(fig1); plt.close(fig2); plt.close(fig3)

    return {
        "fs": fs,
        "nperseg": nperseg,
        "coherence_peak": float(np.nanmax(np.clip(C,0,1))),
        "coherence_peak_freq": float(f_coh[np.argmax(C)] if len(f_coh)>0 else np.nan)
    }

# --------- Main ---------
def main():
    ap = argparse.ArgumentParser(description="PQ fit on solar radio flux CSV (comma-separated).")
    ap.add_argument("--csv", required=True, help="Chemin du CSV (ex: solar_radio_flux.csv)")
    ap.add_argument("--out", default="solar_pq", help="Préfixe de sortie (png/json)")
    ap.add_argument("--fguess", type=float, default=None, help="Fréquence initiale gamma [1/day] (optionnel)")
    args = ap.parse_args()

    df = load_and_prepare(args.csv)
    theta_hat, yhat, P, Q, rmse, r2 = fit_pq(df, f_guess=args.fguess)

    meta = spectra_and_coherence(df, yhat, args.out)

    results = {
        "theta_hat": {
            "gamma_[1/day]": float(theta_hat[0]),
            "lambda_[1/day]": float(theta_hat[1]),
            "phi_[rad]": float(theta_hat[2]),
            "a": float(theta_hat[3]),
            "b": float(theta_hat[4]),
        },
        "period_days": float(1.0/theta_hat[0]),
        "rmse": float(rmse),
        "r2": float(r2),
        **meta
    }
    with open(f"{args.out}_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== PQ fit summary ===")
    print(json.dumps(results, indent=2))
    print(f"\nFigures sauvegardées : {args.out}_fit.png, {args.out}_psd.png, {args.out}_coh.png")

if __name__ == "__main__":
    main()
