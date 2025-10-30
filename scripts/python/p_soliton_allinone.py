#!/usr/bin/env python3
"""

Auteur: L. Danion — ORCID 0009-0008-8733-8261

p_soliton_allinone.py

python p_soliton_allinone.py profiles/profile_phi0_0.08000.npz \
  --outdir out_all --N 1200 --rmin 0.1 --rmax 400 --nev 20 \
  --do-evolve --tmax 200 --dt 0.04 --perturb-amp 1e-4 --seeded-mode 0 --save-plots

Single-file robust pipeline:
 - load profile (.npz)
 - regrid profile to requested (N, rmin, rmax)
 - build Sturm operator L = -d^2/dr^2 + V(r)
 - compute eigenpairs (nev)
 - normalize modes, project profile, diagnostics
 - (optional) run simple evolution experiments (seeded mode / random noise)
 - save eigens.npz, diagnostics.npz, evolution logs & basic plots


"""
import argparse
import json
import logging
import os
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# --- Utilities: second derivative & Sturm builder (robust for non-uniform grids) ---
def second_derivative_matrix(r):
    r = np.asarray(r, dtype=float)
    N = r.size
    if N < 3:
        raise ValueError("Grid too small for D2 (need N>=3)")

    h = np.diff(r)  # length N-1, positive
    # diagonals length N
    diag = np.zeros(N, dtype=float)
    off_m = np.zeros(N - 1, dtype=float)
    off_p = np.zeros(N - 1, dtype=float)

    for i in range(1, N - 1):
        h_im1 = h[i - 1]
        h_i = h[i]
        # centered nonuniform 2nd deriv coef
        a = 2.0 / (h_i * (h_i + h_im1))
        b = -2.0 / (h_i * h_im1)
        c = 2.0 / (h_im1 * (h_i + h_im1))
        off_m[i - 1] = a
        diag[i] = b
        off_p[i] = c

    # enforce Dirichlet BCs at boundaries (phi=0)
    diag[0] = 1.0
    diag[-1] = 1.0

    # stack into shape (3, N)
    data = np.vstack([
        np.concatenate(([0.0], off_m)),     # subdiagonal aligned with offset -1
        diag,
        np.concatenate((off_p, [0.0]))
    ])
    offsets = [-1, 0, 1]
    D2 = sparse.dia_matrix((data, offsets), shape=(N, N)).tocsr()
    return D2

def build_sturm_operator(r, V_diag):
    r = np.asarray(r, dtype=float)
    N = r.size
    V_diag = np.asarray(V_diag, dtype=float)
    if V_diag.size == 1:
        V_diag = np.full(N, float(V_diag))
    if V_diag.size != N:
        raise ValueError("V_diag length != grid length")
    D2 = second_derivative_matrix(r)
    L = -D2 + sparse.diags(V_diag, 0, format='csr')
    return L

# --- IO helpers ---
def load_profile(npzfile):
    d = np.load(npzfile, allow_pickle=True)
    # try common keys
    keys = list(d.keys())
    # find r and phi
    if 'r' in keys:
        r = np.array(d['r'])
    elif 'radius' in keys:
        r = np.array(d['radius'])
    else:
        raise KeyError("no 'r' key found in profile npz")

    phi = None
    for k in ('phi', 'psi', 'profile', 'phi_profile'):
        if k in keys:
            phi = np.array(d[k])
            break
    if phi is None:
        # sometimes stored as array-of-objects
        raise KeyError(f"No recognizable phi/psi/profile in {npzfile}; keys={keys}")
    meta = {}
    for mk in ('omega', 'meta', 'info'):
        if mk in keys:
            meta[mk] = d[mk].item() if hasattr(d[mk], 'item') else d[mk]
    return r.astype(float), phi.astype(float), meta

def regrid_profile(r_old, phi_old, r_new):
    f = interp1d(r_old, phi_old, kind='cubic', bounds_error=False, fill_value=0.0)
    return f(r_new)

# --- eigen solver wrapper robust ---
def compute_eigenpairs(L, nev=6, which='SA', sigma=0.0, dense_thresh=800):
    n = L.shape[0]
    if n <= dense_thresh:
        # use dense solver (np.linalg.eigh) on small systems
        M = L.toarray()
        vals_all, vecs_all = np.linalg.eigh(M)
        # return smallest nev (ascending)
        idx = np.argsort(vals_all)[:nev]
        vals = vals_all[idx]
        vecs = vecs_all[:, idx]
        return vals, vecs
    # else use sparse eigsh with shift-invert if sigma provided
    try:
        if sigma is None:
            vals, vecs = eigsh(L, k=nev, which='SA')
        else:
            vals, vecs = eigsh(L, k=nev, sigma=sigma, which='LM', tol=1e-8, maxiter=5000)
        # eigsh may not return sorted ascending
        order = np.argsort(vals)
        return vals[order], vecs[:, order]
    except Exception as e:
        logging.warning("eigsh failed: %s. Trying fallback with which='SM'..." % (e,))
        vals, vecs = eigsh(L, k=nev, which='SM')
        order = np.argsort(vals)
        return vals[order], vecs[:, order]

# --- normalize modes and project ---
def normalize_modes(vecs, r_modes, weight='r2'):
    # vecs shape: (N, nm)
    r = np.asarray(r_modes)
    if weight == 'r2':
        w = 4.0 * np.pi * (r ** 2)
    elif weight == 'r':
        w = r
    else:
        w = np.ones_like(r)
    # compute norms with trapezoid (safe)
    norms = np.sqrt(np.trapz((vecs ** 2) * w[:, None], r, axis=0))
    # avoid zero-norm
    norms[norms == 0.0] = 1.0
    vecs_norm = vecs / norms[None, :]
    return vecs_norm, norms, w

def project_profile_onto_modes(phi_on, vecs_norm, r_modes, w):
    # phi_on (N,), vecs_norm (N, nm), w (N,)
    coeffs = np.trapz(phi_on[:, None] * vecs_norm * w[:, None], r_modes, axis=0)
    return coeffs

# --- reconstruct and diagnostics ---
def reconstruct_from_modes(coeffs, vecs_norm):
    return (vecs_norm * coeffs[None, :]).sum(axis=1)

def compute_diagnostics(phi_on, phi_recon, r_modes, w):
    phi_norm_sq = np.trapz((phi_on ** 2) * w, r_modes)
    E_recon = np.trapz((phi_recon ** 2) * w, r_modes)
    resid = phi_on - phi_recon
    resid_norm_sq = np.trapz((resid ** 2) * w, r_modes)
    fraction = E_recon / (phi_norm_sq + 1e-300)
    return {
        'phi_norm_sq': float(phi_norm_sq),
        'E_recon': float(E_recon),
        'resid_norm_sq': float(resid_norm_sq),
        'fraction_explained': float(fraction),
    }

# --- simple evolution (toy) leapfrog for phi (wave-like toy) ---
def run_leapfrog_evolution(r, phi0, dt, nsteps, perturb=None, phi_clamp=1e6, save_every=10, outdir=None):
    """
    Simple leapfrog for scalar field: phi_tt = Laplacian_phi (radial 3D spherical laplacian approximated)
    This is a toy integrator used for diagnostics (not your production solver).
    """
    N = r.size
    phi_prev = phi0.copy()
    phi_curr = phi0.copy()
    # small perturb initial velocity if needed
    v0 = np.zeros_like(phi0)
    if perturb is not None:
        phi_curr += perturb

    logs = []
    # central finite difference radial laplacian approx using 2nd deriv matrix
    D2 = second_derivative_matrix(r)
    # To compute energy use w = 4*pi*r^2
    w = 4.0 * np.pi * (r ** 2)
    for step in range(nsteps):
        t = (step + 1) * dt
        # compute laplacian term
        lap = D2.dot(phi_curr)
        phi_next = 2.0 * phi_curr - phi_prev + (dt ** 2) * (-lap)  # toy sign
        # logging
        E = float(np.trapz(0.5 * (phi_next ** 2) * w, r))
        maxphi = float(np.max(np.abs(phi_next)))
        logs.append((t, maxphi, E))
        # clamp check
        if np.isnan(maxphi) or maxphi > phi_clamp:
            logging.warning(f"[warn] phi exceeded clamp at step {step}, stopping evolution")
            break
        phi_prev, phi_curr = phi_curr, phi_next
    # save logs as numpy array
    arr = np.array(logs, dtype=float)
    if outdir:
        np.savez_compressed(os.path.join(outdir, "evolution_log.npz"), log=arr)
    return arr

# --- plotting helpers ---
def plot_maxphi_vs_t(logarray, outpath):
    if logarray.size == 0:
        return
    t = logarray[:, 0]
    maxphi = logarray[:, 1]
    plt.figure()
    plt.plot(t, maxphi, '-o', ms=3)
    plt.xlabel('t'); plt.ylabel('max|phi|')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_profile(r, phi, outpath, title=None):
    plt.figure()
    plt.plot(r, phi, '-')
    if title: plt.title(title)
    plt.xlabel('r'); plt.ylabel('phi')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# --- main CLI pipeline ---
def main():
    p = argparse.ArgumentParser()
    p.add_argument('profile', help='profile .npz file (must contain r and phi)')
    p.add_argument('--outdir', default='out_all', help='output directory')
    p.add_argument('--N', type=int, default=1200)
    p.add_argument('--rmin', type=float, default=0.1)
    p.add_argument('--rmax', type=float, default=400.0)
    p.add_argument('--nev', type=int, default=12)
    p.add_argument('--do-evolve', action='store_true', help='run toy evolution after eigen analysis')
    p.add_argument('--tmax', type=float, default=200.0)
    p.add_argument('--dt', type=float, default=0.04)
    p.add_argument('--perturb-amp', type=float, default=1e-4)
    p.add_argument('--seeded-mode', type=int, default=0, help='mode index to seed (if -1 -> random noise)')
    p.add_argument('--weight', default='r2', choices=['r2', 'r', 'none'])
    p.add_argument('--dense-thresh', type=int, default=800)
    p.add_argument('--save-plots', action='store_true')
    p.add_argument('--phi-clamp', type=float, default=1e6)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    logging.info(f"Starting pipeline: profile={args.profile} outdir={outdir}")
    r_prof, phi_prof, meta = load_profile(args.profile)
    logging.info(f"Loaded profile: r range {r_prof.min():.3e} .. {r_prof.max():.3e} (len={r_prof.size})")
    # prepare target grid
    r_modes = np.linspace(args.rmin, args.rmax, args.N)
    phi_regr = regrid_profile(r_prof, phi_prof, r_modes)

    # build operator
    # Example potential V(r) from profile meta or zero if absent.
    V = np.zeros_like(r_modes)
    if 'meta' in meta and isinstance(meta['meta'], dict) and 'V' in meta['meta']:
        V = np.asarray(meta['meta']['V'], dtype=float)
        if V.size != r_modes.size:
            V = np.interp(r_modes, r_prof, V)
    logging.info("Sturm operator build (this may take a moment for large N)...")
    L = build_sturm_operator(r_modes, V)

    # compute eigenpairs
    logging.info(f"Solving eigenproblem (nev={args.nev})...")
    vals, vecs = compute_eigenpairs(L, nev=args.nev, sigma=0.0, dense_thresh=args.dense_thresh)
    logging.info("Eigenvalues (first %d): %s" % (min(len(vals), 50), np.array2string(vals[:min(len(vals), 50)], max_line_width=200)))
    # save eigen raw
    eigfile = outdir / "eigens.npz"
    np.savez_compressed(str(eigfile), r=r_modes, vals=vals, vecs=vecs, phi_profile=phi_regr, meta=meta)
    logging.info(f"Saved eigen data to {eigfile}")

    # normalize and project
    vecs_norm, norms, w = normalize_modes(vecs, r_modes, weight=args.weight)
    coeffs = project_profile_onto_modes(phi_regr, vecs_norm, r_modes, w)
    phi_recon = reconstruct_from_modes(coeffs, vecs_norm)
    diag = compute_diagnostics(phi_regr, phi_recon, r_modes, w)

    diagnostics_file = outdir / "diagnostics.npz"
    np.savez_compressed(str(diagnostics_file),
                        phi_norm_sq=diag['phi_norm_sq'],
                        E_recon=diag['E_recon'],
                        resid_norm_sq=diag['resid_norm_sq'],
                        fraction_explained=diag['fraction_explained'],
                        vals=vals, coeffs=coeffs, norms=norms, r=r_modes)
    logging.info(f"phi_norm={np.sqrt(diag['phi_norm_sq']):.6e} resid_norm={np.sqrt(diag['resid_norm_sq']):.6e} fraction_explained={diag['fraction_explained']:.6e}")
    logging.info("abs(coeffs) first 12: %s" % (np.array2string(np.abs(coeffs[:12]), precision=6, max_line_width=200)))

    # optional plotting
    if args.save_plots:
        plot_profile(r_modes, phi_regr, str(outdir / "profile_regridded.png"), title="profile_regridded")
        plot_profile(r_modes, phi_recon, str(outdir / "profile_recon.png"), title="profile_reconstructed")
        logging.info("Saved profile plots")

    # optional evolution: seeded and random_noise runs
    if args.do_evolve:
        nsteps = int(max(1, round(args.tmax / args.dt)))
        # seeded perturbation: mode 'seeded-mode'
        if args.seeded_mode >= 0 and args.seeded_mode < vecs_norm.shape[1]:
            mode_vec = vecs_norm[:, args.seeded_mode]
            perturb = args.perturb_amp * mode_vec
            run_out = outdir / f"seeded_mode{args.seeded_mode}"
            run_out.mkdir(exist_ok=True)
            log = run_leapfrog_evolution(r_modes, phi_regr, args.dt, nsteps, perturb=perturb, phi_clamp=args.phi_clamp, outdir=str(run_out))
            plot_maxphi_vs_t(log, str(run_out / "maxphi_vs_t.png"))
            logging.info(f"Seeded run saved in {run_out}")
        # random noise run
        perturb = args.perturb_amp * (np.random.randn(r_modes.size))
        run_out = outdir / "random_noise"
        run_out.mkdir(exist_ok=True)
        log = run_leapfrog_evolution(r_modes, phi_regr, args.dt, nsteps, perturb=perturb, phi_clamp=args.phi_clamp, outdir=str(run_out))
        plot_maxphi_vs_t(log, str(run_out / "maxphi_vs_t.png"))
        logging.info(f"Random-noise run saved in {run_out}")

    # produce a small JSON summary
    summary = {
        'eigenfile': str(eigfile),
        'diagnostics': str(diagnostics_file),
        'vals_first12': vals[:min(12, vals.size)].tolist(),
        'coeffs_first12': np.abs(coeffs[:min(12, coeffs.size)]).tolist(),
        'phi_norm': float(np.sqrt(diag['phi_norm_sq'])),
        'fraction_explained': float(diag['fraction_explained']),
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logging.info("Pipeline complete. Summary written to " + str(outdir / "summary.json"))

if __name__ == "__main__":
    main()
