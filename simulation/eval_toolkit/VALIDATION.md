# eval_toolkit validation (run 2026-11-04, `validate.py` via `run_validate_gpu.sh`, 11,751 s wall clock)

Raw numbers: `validation.json` (every per-file value plus summaries). Figure: `eval_toolkit_validation.png`. Models and licences: `MODELS.md`.
The device was CUDA for WavLM, UTMOS, AASIST, Whisper and audeering, under GPU lease `eval-toolkit`, and CPU for ECAPA and the Praat/WORLD re-measurement (the study path).
This file reports measurements only. The notes say what each number does and does not establish.

## Audio sets
- **TTS**: `tts_sources/`, 15 files. These are all three ko-KR edge-tts voices (SunHi F, InJoon M, HyunsuMultilingual M) × 5 fixed sentences, at 16 kHz mono, 5.1–5.8 s long.
- **Non-study speakers**: AI Hub 71631 indoor speakers whose ID is not among the 2,525 eligible study speakers. Only `speaker_index.parquet` was read from `stage1/results`. Each speaker contributes own-channel speech (own label segments, concatenated) from the best-SNR conversation. These speakers are non-eligible because of low SNR (< 20 dB) or < 60 s of speech, so they are noisier than the study sample. Labelled SNR across the 40 real speakers: median 14 dB, IQR 9–20.5 dB, range 0–54 dB, with 10 of 40 at ≥ 20 dB.
- **Meter training** (`train_gender_meter.py`): 150 F + 150 M, 30 s each. **Held-out**: the next 50 F + 50 M in the same seed-0 random order, 30 s each, disjoint from training. **REAL40**: the first 20 F + 20 M held-out speakers. Clip A is 0–10 s and clip B is 20–30 s of the held-out 30-s clip. **PRAAT10**: the first 5 F + 5 M of REAL40, clip A.
- Clips are kept in `validation_clips/` on the NAS only (AI Hub terms; never distribute).

## 1. CER (Whisper large-v3, local) on TTS against the known text
| set | n | pooled CER | per-file mean | per-file max |
|---|---|---|---|---|
| all TTS | 15 | 0.000 | 0.000 | 0.000 |
| SunHi / InJoon / Hyunsu | 5 each | 0.000 / 0.000 / 0.000 | | |

Every hypothesis matched the reference after normalisation. The only raw differences were spacing (e.g. "걸어볼" vs "걸어 볼"), which `normalize_ko` removes. Note: on clean stock TTS the metric sits at its floor, so this run checks that the pipeline (forced Korean, normalisation, edit distance) works. It does not measure how sensitive the metric is to degraded audio.

## 2. UTMOS and AASIST: TTS (n = 15) vs real AI Hub clip A (n = 40)
| metric | TTS mean ± sd [min, max] | real mean ± sd [min, max] | AUC (real > TTS) |
|---|---|---|---|
| UTMOS (pred. MOS) | 3.757 ± 0.256 [3.33, 4.08] | 2.088 ± 0.386 [1.23, 2.82] | 0.000 |
| AASIST score (mean over 4.04-s windows; higher = bona fide) | 1.179 ± 1.091 [−0.96, 3.21] | −1.551 ± 2.294 [−6.66, 2.28] | 0.152 |
| AASIST official first-window score | 1.179 ± 1.091 | −1.402 ± 2.390 | 0.190 |
| AASIST P(bona fide) | 0.850 ± 0.273 | 0.374 ± 0.401 | 0.150 |

- UTMOS per voice: Hyunsu 3.96, SunHi 3.66, InJoon 3.64. Among the real clips, UTMOS did not track labelled SNR (Spearman ρ = 0.13, p = 0.42). The 10 clips with SNR ≥ 20 dB had a mean of 2.06.
- AASIST: 87% of TTS files scored > 0 (bona fide side), against 35% of real clips. The real clips with SNR ≥ 20 dB had a mean score of −2.52. **Measured result: in this setting the ASVspoof 2019 LA AASIST scores modern neural edge-tts speech as more bona fide than real AI Hub conversational recordings.** It therefore does not separate this TTS from real speech, and it reacts to the recording/channel condition of the real clips (phone/indoor conversational speech, concatenated segments). Any P2/P3 use should compare conditions against each other (same source, converted vs unconverted) rather than read the score as an absolute "synthetic" flag. The paper text must not present it as a detector of TTS/VC.

## 3. ASV: 40 non-study speakers, same speaker = (A_i, B_i), different speaker = (A_i, B_j), i ≠ j
| model | same-speaker cos mean ± sd [min] | diff-speaker cos mean ± sd [max] (n = 1560) | same-gender diff mean (n = 760) | EER, all impostors (thr) | EER, same-gender impostors (thr) | rank-1 identification |
|---|---|---|---|---|---|---|
| ECAPA (study) | 0.788 ± 0.068 [0.595] | 0.186 ± 0.118 [0.820] | 0.241 | 0.38 % (0.595) | 0.79 % (0.595) | 0.95 |
| WavLM-base-plus-SV | 0.960 ± 0.032 [0.793] | 0.713 ± 0.164 [0.976] | 0.841 | 5.19 % (0.926) | 7.96 % (0.932) | 0.85 |

Note: both clips of a trial come from the same conversation (same session and channel, 10 s apart), so these EERs are same-session figures and optimistic for cross-session use. With 40 target trials, one target trial is 2.5 % of the miss rate. WavLM-SV cosines are compressed near 1 (different-speaker mean 0.71), so its threshold is far from ECAPA's. Thresholds are not interchangeable between the two models.

## 4. Gender classifiers on 100 held-out non-study speakers (50 F, 50 M; 30-s clips; threshold 0.5)
| classifier | accuracy | female acc. | male acc. | AUC |
|---|---|---|---|---|
| ECAPA balanced meter (`gender_meter`) | 0.98 | 0.96 | 1.00 | 0.9996 |
| audeering, P(f)/(P(f)+P(m)) | 1.00 | 1.00 | 1.00 | 1.000 |
| audeering, 3-way argmax (child counted as error) | 0.99 | | | (1 speaker argmax = child; a female speaker in her 30s, ID withheld) |

- Meter vs audeering agreement: 0.98. Both meter errors were female speakers with meter P(f) of 0.45 and 0.48 (audeering 0.99 and 0.92).
- Meter training (`gender_meter_balanced.json`): 150 F + 150 M, 5-fold CV accuracy 0.980 ± 0.016 (folds 0.983, 0.950, 0.983, 1.000, 0.983), selected C = 0.36, training accuracy 0.99. Caveat: the age mix is unbalanced between genders (male training speakers are 62 % 20s; female ones are mostly 40s–50s), so the meter can partly use age cues.
- On the TTS sources (not held-out speakers), audeering gave SunHi P(f) 0.956–0.978 and the male voices ≤ 0.002. The ECAPA meter gave SunHi only 0.56–0.65 and the male voices 0.19–0.27, so it is much less confident on stock TTS than on real held-out speech (where the female median was about 0.8). This matters for P3: meter scores on TTS-derived audio sit in a compressed range.

## 5. remeasure() vs direct parselmouth calls (PRAAT10)
| quantity | study settings: mean / max abs diff | Praat defaults (to_pitch(), to_formant_burg()): mean abs diff (mean rel.) | max abs diff |
|---|---|---|---|
| F0 median (Hz) | 0 / 0 | 0.77 Hz (0.63 %) | 3.83 Hz |
| F0 range (st, p95/p5) | 0 / 0 | 1.77 st (7.3 %) | 15.28 st |
| F1 (Hz) | 0 / 0 | 12.2 Hz (2.4 %) | 61.0 Hz |
| F2 (Hz) | 0 / 0 | 42.7 Hz (2.9 %) | 174.6 Hz |
| F3 (Hz) | 0 / 0 | 72.6 Hz (2.7 %) | 276.4 Hz |
| F4 (Hz) | 0 / 0 | 101.6 Hz (2.9 %) | 299.6 Hz |

The study-settings comparison is an independent re-implementation with the same Praat parameters, and it matches exactly: the wrapper adds no resampling or segmentation artefact on 16-kHz input. The Praat-defaults column shows how much the gender-dependent floor/ceiling and formant ceiling matter. The 15-st F0-range outlier comes from the default 75–600 Hz range admitting octave errors. All 19 `env3oct_*` bands were returned for every clip. On TTS, `remeasure` gave F0 medians of 131–142 Hz (Hyunsu), 145–157 Hz (InJoon) and 219–227 Hz (SunHi).

## What did not work / deviations
- edge-tts offers only **3** Korean voices (1 F, 2 M). The P2/P3 plans assumed 4 (2 F + 2 M). The female side has a single voice.
- AASIST does not separate edge-tts from real AI Hub speech here (section 2).
- Whisper CER is at the floor on the clean TTS references (section 1).
- Runtime: under the concurrent stage-1 analysis load (load average 30–60 on 20 cores), CPU-side ECAPA took about 20 s per 10-s clip. Large P2/P3 batches should run when the CPU is free, or parallelise ECAPA in worker processes as `measure_voice` does.
