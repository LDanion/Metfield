#!/usr/bin/env python3

"""
Auteur: L. Danion — ORCID 0009-0008-8733-8261

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lombscargle

# Charger les données UTC-UTC(k)
df = pd.read_csv("runs/bipm/utc9098_parsed_long_fix.csv")
df = df.dropna(subset=["value_ns"])
t = df["mjd"].values
y = df["value_ns"].values - np.mean(df["value_ns"])

# Fréquences testées : 1e-6 à 1 / jour
f = np.logspace(-6, 0, 10000)
p = lombscargle(t, y, 2*np.pi*f, normalize=True)

# Conversion en période (jours)
period = 1 / f

plt.figure(figsize=(10, 6))
plt.loglog(period, p, lw=1.2)
plt.xlabel("Période (jours)")
plt.ylabel("Puissance normalisée")
plt.title("Spectre Lomb–Scargle — UTC-UTC(k) (BIPM 1990–1998)")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Sauvegarde
out = pd.DataFrame({"period_days": period, "power": p})
out.to_csv("runs/bipm/bipm9098_psd.csv", index=False)
print("✅ Spectre sauvegardé dans runs/bipm/bipm9098_psd.csv")
