#!/usr/bin/env bash
# P1: finish the TTS-source conversions on CUDA under the GPU lease (resumes the partial CPU run), release, then
# eval_runner --part gpu (takes its own lease if PKV_GPU_LEASE is set). PYTORCH_JIT=0: the scripted WaveNet gate needed nvrtc builtins missing on the study machine; plain eager ops are identical.
cd "$(dirname "$0")"; L="${PKV_GPU_LEASE:-}"; PY="${PKV_PYTHON:-python}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
[ -n "$L" ] && $L acquire p1-tc-convert $$ ${PKV_GPU_LEASE_SUPERVISOR:-}
echo "$(date '+%F %T') convert start" >> results/tc_convert.log
PYTORCH_JIT=0 PKV_CONV_DEVICE=cuda nice -n 5 $PY tc_convert.py >> results/tc_convert.log 2>&1; rc=$?
echo "$(date '+%F %T') convert exit $rc" >> results/tc_convert.log
[ -n "$L" ] && $L release p1-tc-convert $$
[ $rc -eq 0 ] || exit $rc
mkdir -p results/eval
echo "$(date '+%F %T') eval start" >> results/tc_eval_runner.log
PKV_EVAL_DEVICE=cuda nice -n 5 $PY ../../../simulation/eval_runner.py --manifest outputs/manifest.json --out results/eval/p1tc --part gpu >> results/tc_eval_runner.log 2>&1; rc=$?
echo "$(date '+%F %T') eval exit $rc" >> results/tc_eval_runner.log
exit $rc
