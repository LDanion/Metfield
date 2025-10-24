#!/usr/bin/env bash
# ==========================================================
# run_japan_pipeline.sh
# Complete reproduction pipeline for P-field epicentre detection (Japan)
# Author: Laurent Danion (ORCID: 0000-0003-3852-6684)
# Project: P/Theory — Metronomic Field Analysis
# ==========================================================

set -e  # stop on first error
export OMP_NUM_THREADS=4
DATE_TAG=$(date +"%Y%m%d_%H%M%S")

BASE_DIR="results_eval/japan/${DATE_TAG}"
DATA_DIR="data/japan"

mkdir -p "${BASE_DIR}"

echo "=========================================================="
echo "[STEP 1] Fetching regional earthquake catalogue"
echo "=========================================================="

python fetch_catalog.py \
    --region japan \
    --start 2010-01-01 --end 2024-12-31 \
    --bbox 30.0 46.0 128.0 148.0 \
    --mag-min 2.5 \
    --providers usgs emsc resif \
    --outdir "${DATA_DIR}"

echo "=========================================================="
echo "[STEP 2] Generating alert series from P field"
echo "=========================================================="

python detectquakes.py \
    --region japan \
    --input "${DATA_DIR}/P_phys.csv" \
    --threshold 0.85 \
    --radius 45 \
    --outdir "${BASE_DIR}"

echo "=========================================================="
echo "[STEP 3] Matching alerts with USGS events"
echo "=========================================================="

python match_events.py \
    --alerts "${BASE_DIR}/alerts.csv" \
    --catalog "${DATA_DIR}/japan_2010_2024_M2.5.csv" \
    --max-days 30 \
    --out "${BASE_DIR}/matches.csv"

echo "=========================================================="
echo "[STEP 4] Generating spatial probability map"
echo "=========================================================="

python compute_pspatial.py \
    --alerts "${BASE_DIR}/alerts.csv" \
    --catalog "${DATA_DIR}/japan_2010_2024_M2.5.csv" \
    --outdir "${BASE_DIR}/p_spatial"

echo "=========================================================="
echo "[STEP 5] Evaluating spatial accuracy (pre-calibration)"
echo "=========================================================="

python eval_spatial_accuracy.py \
    --pred "${BASE_DIR}/p_spatial/epicenters_pred_geo.csv" \
    --truth "${DATA_DIR}/japan_2010_2024_M2.5.csv" \
    --outdir "${BASE_DIR}/spatial_accuracy_local"

echo "=========================================================="
echo "[STEP 6] Calibrating spatial bias"
echo "=========================================================="

python calibrate_spatial_bias.py \
    --matches "${BASE_DIR}/matches.csv" \
    --outdir "${BASE_DIR}/spatial_bias_calib"

# Apply calibration
python eval_spatial_accuracy.py \
    --pred "${BASE_DIR}/pspatial_cal/epicenters_pred_geo.csv" \
    --truth "${DATA_DIR}/japan_2010_2024_M2.5.csv" \
    --outdir "${BASE_DIR}/spatial_accuracy_calib"

echo "=========================================================="
echo "[STEP 7] Comparing pre- and post-calibration gains"
echo "=========================================================="

python compare_spatial_gains.py \
    --before "${BASE_DIR}/spatial_accuracy_local" \
    --after  "${BASE_DIR}/spatial_accuracy_calib" \
    --outdir "${BASE_DIR}/spatial_gain"

echo "=========================================================="
echo "[DONE] Complete pipeline executed successfully."
echo "Results available in: ${BASE_DIR}"
echo "Figures:"
echo " - spatial_gain/cdf_before_after.png"
echo " - pspatial_cal/pspatial_map.png"
echo " - spatial_accuracy_calib/distance_hist.png"
echo "=========================================================="
