#!/usr/bin/env bash
# P1 tone-colour GPU step (A2 duration subsets need CUDA for OpenVoice ref_enc). Optional lab GPU lease: PKV_GPU_LEASE (see README).
cd "$(dirname "$0")"; L="${PKV_GPU_LEASE:-}"; PY="${PKV_PYTHON:-python}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
[ -n "$L" ] && $L acquire p1-tc-gpu $$ ${PKV_GPU_LEASE_SUPERVISOR:-}
echo "$(date '+%F %T') start" >> results/tc_gpu.log
nice -n 5 $PY tc_gpu.py >> results/tc_gpu.log 2>&1; rc=$?
echo "$(date '+%F %T') exit $rc" >> results/tc_gpu.log
[ -n "$L" ] && $L release p1-tc-gpu $$
exit $rc
