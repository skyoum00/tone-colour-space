#!/usr/bin/env bash
# S1-S4 on the GPU KRR backend (falls back to CPU when CUDA is absent). Optional lab GPU lease: PKV_GPU_LEASE (see README).
cd "$(dirname "$0")"; L="${PKV_GPU_LEASE:-}"; PY="${PKV_PYTHON:-python}"; mkdir -p results
export KRR_BACKEND=gpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
[ -n "$L" ] && $L acquire p1-extra $$ ${PKV_GPU_LEASE_SUPERVISOR:-}
echo "$(date '+%F %T') start" >> results/p1_extra.log
nice -n 5 $PY p1_extra_analyses.py >> results/p1_extra.log 2>&1; rc=$?
echo "$(date '+%F %T') exit $rc" >> results/p1_extra.log
[ -n "$L" ] && $L release p1-extra $$
exit $rc
