#!/usr/bin/env bash
# Ejecuta las corridas del experimento TC5061_A03_AdultIncome_LogReg.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

$PY train.py --run-name 01_baseline_l2_C1
$PY train.py --run-name 02_l2_C0.001          --C 0.001
$PY train.py --run-name 03_l2_C100            --C 100
$PY train.py --run-name 04_l1_C0.01           --C 0.01 --l1-ratio 1
$PY train.py --run-name 05_l1_C0.001          --C 0.001 --l1-ratio 1
$PY train.py --run-name 06_elasticnet_C0.1    --C 0.1 --l1-ratio 0.5
$PY train.py --run-name 07_l2_C1_balanced     --class-weight balanced
$PY train.py --run-name 08_l2_C1_minfreq500   --min-frequency 500
$PY train.py --run-name 09_baseline_seed7     --seed 7
# Repetición de la corrida 01 para comprobar reproducibilidad
$PY train.py --run-name 10_baseline_repeticion
