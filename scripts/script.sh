#!/bin/bash
# script.sh
# Simple runner for the diagnosis pipeline on the test set

# Stop on error
set -e

# ---- CONFIG ----
PATIENTS="data/release_test_patients"        # path to your test CSV
CONDITIONS="data/release_conditions.json"        # path to your conditions file
EVIDENCES="data/release_evidences.json"          # path to your evidences file
OUTDIR="outputs"                           # directory for all generated files
LABEL_COL="PATHOLOGY"                      # ground-truth column name in CSV
LIMIT=100

# ---- RUN ----
echo "Running test_set_pipeline.py on the test set..."
python3 test_set_pipeline.py \
  --patients "$PATIENTS" \
  --evidences "$EVIDENCES" \
  --conditions "$CONDITIONS" \
  --label-col "$LABEL_COL" \
  --limit "$LIMIT" \
  --out-dir "$OUTDIR"

echo ""
echo "Done. Results saved to $OUT"
