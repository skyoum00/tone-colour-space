> Release note: this is the original dated Stage 1-2 pre-registration (fixed 2026-11-02) with its dated amendments, copied unchanged.

# Stage 1–2 pre-registration (fixed 2026-11-02, before any audio was measured)

Gate 1 (H1) of the plan: interpretable acoustic variables explain ≥ 70% of speaker-embedding variance.
Everything below is fixed before measurement. Changes after measurement go to a dated "Deviations" section with the reason, and the original analysis is still reported.

## Data
- AI Hub 71631 (free conversation, adults), **indoor recordings only** (TS_01 실내_1–5, VS_01 실내). Outdoor recordings (SNR mostly 0–8 dB) are excluded from all H1 analyses; VS_02 실외 is used only to test code.
- Speaker key = label ID + gender + age band. Per speaker, the conversation with the highest labelled SNR is used.
- Inclusion: labelled SNR ≥ 20 dB, and ≥ 60 s of the speaker's own speech after the segment rules below. Expected about 3,400 eligible speakers before the duration rule.
- Segments: label utterances of that speaker, 1.0–15 s long, not overlapping any utterance of the other speaker (0.2 s guard). The channel carrying each speaker is identified from the audio (energy over that speaker's label segments); if the two channels are not separable, the mono mix is used and only non-overlapping segments are kept (this is recorded per speaker).
- Up to 120 s of speech per speaker, taking segments in time order. Audio resampled to 16 kHz mono.

## Variables (candidate set, 19 — the heading originally said 17, a miscount of the same list below; corrected 2026-11-02 before measurement)
Source: F0 mean (Hz; analysed as log), F0 range (st, 5th–95th percentile), jitter local (%), shimmer local (dB), HNR (dB), H1–H2 (dB), spectral slope 0–500 Hz and 500–1500 Hz (dB/Hz, eGeMAPS), alpha ratio (dB).
Vocal tract: F1–F4 means (Hz; Praat Burg, ceiling 5,000 Hz for male and 5,500 Hz for female labels), apparent vocal-tract length from formant spacing (cm).
Quality: low-band energy ratio 0–1 kHz (dB re total), speaker's-formant band energy 2.5–3.5 kHz (dB re total), spectral centroid (Hz).
Delivery: speech rate (Hangul syllables of the label transcript per second of labelled speech), utterance-final F0 slope (st/s, last 300 ms, mean over utterances).
Not measured in Stage 1: nasality, articulation clarity (need alignment; recorded as a limitation).
Tools: Praat via parselmouth 0.4.7, openSMILE 2.6.0 (eGeMAPSv02, functionals), pyworld 0.3.5 for cross-checking F0.

## Embedding
SpeechBrain ECAPA-TDNN `speechbrain/spkrec-ecapa-voxceleb` (192-d). Speaker embedding = mean of segment embeddings, L2-normalised.
Secondary: same analysis on WavLM-base-plus-sv x-vectors (512-d) to check that the result is not specific to one encoder.

## Primary H1 analysis
- Regression from standardised variables to the 192-d speaker embedding, 10-fold cross-validation split by speaker, fixed seed 0.
- R² = 1 − Σ(residual SS over all dimensions) / Σ(total SS over all dimensions) on held-out speakers (variance-weighted multi-output R²).
- Primary model: kernel ridge regression with an RBF kernel, hyperparameters by inner 5-fold CV. Linear ridge is reported alongside.
- **Gate 1 passes if the primary out-of-fold R² ≥ 0.70** (plan value; not changed after seeing results). 95% CI by speaker bootstrap (B = 2,000) of the out-of-fold predictions.

## Reported with H1 (not gates)
- Nuisance-only R² (recording device, labelled SNR) and variables + nuisance R², to show how much of the embedding is channel.
- Embedding reliability ceiling: split-half R² (embedding of the first half of a speaker's segments predicting the second half).
- R² within each gender label.
- Speaker verification EER from variables alone (cosine on standardised variables) vs from ECAPA, on held-out speakers.
- Per-variable contribution: drop-one ΔR², mutual information with the embedding's first 10 principal components, and the variable correlation matrix (to find entangled pairs such as F0 and vocal-tract length).

## Clarifications added before any indoor audio was read (2026-11-02, after a code test on one outdoor file)
- The wav files are 16 kHz stereo (the labels say 48 kHz); the two channels are nearly uncorrelated, one speaker per channel. Measurement uses the speaker's own channel at 16 kHz.
- Channel separation (energy of the chosen channel over the other, on the speaker's own segments) is recorded per speaker. The primary analysis uses all eligible speakers; a sensitivity analysis repeats H1 on speakers with separation ≥ 10 dB.
- Pitch range for Praat: 60–300 Hz for male labels, 100–500 Hz for female labels.
- Variables are also measured on the first and second half of each speaker's segments (same halves as the embedding), so that variable-only speaker verification (enrol on one half, test on the other) and split-half variable reliability can be computed.

## Amendment A1 (2026-11-02, before the sealed measurements were opened; ARIS Stage 1 complete)
Reasons: the user asked that every measurable voice variable, including frequency-band content, be considered; the literature survey found Huckvale (arXiv:2510.16489, 2025), who predicted speaker embeddings (incl. ECAPA) from 9 interpretable variables at roughly the level of 7 PCA components (~50% variance), and recommended reporting the backbone encoder too.
- **Full candidate pool** (measured for every speaker and both halves): the core 19 plus F0 median/p5/p95/SD(st)/mean slope, jitter RAP/PPQ5, shimmer local%/APQ3/APQ5/APQ11, CPPS, formant bandwidths B1–B4, formant dispersion, F1/F2 SD, intensity mean/SD, voiced fraction, pause fraction, 16 long-term-average-spectrum bands of 500 Hz from 0 to 8 kHz, spectral roll-off (85%), flatness and spread (interpretable pool), and all 88 eGeMAPSv02 functionals (standard pool).
- **Gate 1 stays as registered**: primary = the core 19 variables, RBF kernel ridge, out-of-fold R² ≥ 0.70. Not changed.
- **Reported alongside (not gates)**: (a) the full interpretable pool; (b) interpretable pool + all eGeMAPS; (c) the best k variables (k = 5, 10, 15, 20, 30) chosen from the full interpretable pool by forward selection run **inside each training fold** (nested), giving an R²-vs-k curve; (d) a Huckvale-comparable subset (F0 mean, F0 range, formant dispersion/VTL, HNR, CPPS, spectral slope, H1–H2, speech rate, intensity), same R² definition; (e) the same analyses with the backbone's own speaker encoder (CosyVoice 2 CAM++, 192-d) as the target, added when Stage 4 picks the backbone.
- The 224 speakers measured before this amendment are re-measured with the extended code (old file kept as superseded, not deleted).

## Amendment A2 (2026-11-02, before the sealed measurements were opened; from the Fable idea-stage review, idea-stage/IDEA_REVIEW_FABLE.md)
Gate 1 definition and the 0.70 threshold are unchanged. Added, all reported and none a gate:
- **Demographic baselines**: R² from gender + age-band labels alone; the between-gender share of total embedding SS; R² of the core-19 and pools after removing gender means from both X and Y (within-gender centring); R² of variables + gender/age over gender/age alone. Reason: 78% female sample, and the measurement uses gender-dependent pitch range and formant ceiling, so gender information enters the variables by construction.
- **Channel confound**: the interpretable pool is reported with and without the 16 LTAS bands, and with device/SNR nuisance added, because band energies also index the recording channel.
- **Reliability ceiling caveat**: split-half R² compares half-length means from the same session; it is reported as a same-session, half-length bound, not as a true ceiling.
- **Duplicate-speaker check**: number of speaker pairs with ECAPA cosine > 0.85 across different speaker keys, reported before the main analysis (speaker keys are label ID + gender + age band and could alias the same person).
- **Inner CV scorer**: hyperparameters are chosen with the same variance-weighted multi-output R² as the outer metric (the original code used sklearn's default uniform average; affects hyperparameter choice only).
- **Nested selection curve (A1-c) implementation**: forward selection with linear ridge inside each training fold (KRR selection is computationally prohibitive at this n); the selected set is then scored with RBF-KRR at the median core-model hyperparameters.
- CPPS uses Praat "Get CPPS" with the "Robust" trend fit ("Robust slow" took ~6 s per second of audio; code change only, recorded here).

## Amendment A3 (2026-11-02, before opening; remaining review notes)
- Bootstrap CIs resample out-of-fold predictions by speaker and do not include model-refit variance; they are labelled "prediction-conditional".
- The cross-half analysis (variables of half 1 → embedding of half 2) is reported next to the primary R² as the control for shared-audio inflation.
- The labelled SNR is a label, not a measurement; it is used only as a nuisance covariate and is described as such.
- Wording: results are described as "variance of the ECAPA speaker embedding explained", not as "differences people hear"; the perceptual claim waits for the listening test.

## Amendment A4 (2026-11-02, before opening; user request: spectral analysis matters)
- Added measurements: 1/3-octave LTAS (19 bands, centres 100 Hz–6.3 kHz, dB re total); WORLD spectral envelope (CheapTrick on DIO+StoneMask F0, voiced frames) in the same 19 bands (dB re envelope mean) — the envelope removes the harmonic fine structure, so it describes the vocal-tract/voice-source filter rather than F0; MFCC 1–20 mean and SD.
- The 1/3-octave LTAS and envelope bands join the interpretable pool (physical frequency bands, dB); MFCCs form a separate non-physical pool.
- Reported (non-gate): R² from the envelope bands alone, the 1/3-octave LTAS alone, MFCCs alone, core-19 + envelope, interpretable + MFCC; each also with device/SNR nuisance added (channel confound check).
- Gate 1 unchanged (core 19). The 26 speakers measured under A1 are re-measured (file kept unopened in results/superseded/).
- Generation (EXPERIMENT_PLAN E3/E4): a spectral-envelope matching post-filter toward the target envelope bands is added as a component with an ablation; envelope bands are re-measured on outputs.

## Amendment A5 (2026-11-02, before opening; from literature/SPECTRAL_FEATURES_REVIEW.md)
- Added measurements (code `harmonic_measures.py`, `measure_voice.py`; reference functions from opensauce-python commit e633608, Apache-2.0):
  C1 formant-corrected harmonic slopes H1*–H2*, H2*–H4*, H4*–H2k*, H2k*–H5k (Iseli–Alwan correction with Hawks–Miller formula bandwidths, VoiceSauce convention), and C9 H1*–A1*, H1*–A3*; per voiced 10-ms frame, speaker median.
  C2 50-ms moving coefficient of variation of F0, F1–F4, energy, SHR; moving SD of the four C1 slopes (dB differences).
  C3 band HNR 0–500/1500/2500/3500 Hz after de Krom 1993 (own implementation with inter-harmonic baseline correction; not VoiceSauce-identical; checked to decrease with added noise on a test tone).
  C4 SHR (Sun 2002; opensauce `shrp` defaults: 40 ms, 10 ms step, threshold 0.4, ceiling 1250 Hz): mean and fraction > 0.4.
  C5 WORLD envelope in 1-ERB bands, ERB-rate 20–32 (~1.7–7.7 kHz), and LFCC 1–20 mean/SD (40 linear filters).
- Not added: GNE (cost), LTFD extension and derived indices (redundant); recorded in the review.
- Pools: C1–C5 physical measures join the interpretable pool; LFCC joins MFCC in the non-physical pool.
- Reported (non-gate): voice-quality A5 alone, core 19 + A5, LFCC alone, high-band ERB envelope alone, interpretable + MFCC + LFCC, each with nuisance added; interpretable pool minus literature-flagged features (perturbation at 16 kHz, LPC bandwidths, near-Nyquist bands).
- Flags assigned from the literature before opening; flagged features stay in the main interpretable pool and the without-flagged pool is a sensitivity analysis.
- Gate 1 unchanged.
- Code fix (before opening): bootstrap CIs use the bias-corrected percentile interval (Efron 1987); on the random-array dry run the plain percentile interval excluded the point estimate because of the (n−1)/n variance shrinkage in resamples. Gate decision (point estimate) unaffected.

## Amendment A6 (2026-11-02, before opening; data-integrity check)
- Finding (from file sizes and label metadata only, no measured values opened): for 115 of 2,525 eligible speakers the wav is shorter than the label `FileLength` (wav/label < 0.99; 2 below 0.5), i.e. truncated audio; 2,410 match within 1%. Table: `results/audio_label_duration_check.csv`. 11 speakers had errored because label segments lay beyond the end of the audio.
- Primary sample for Gate 1 and all A1–A5 analyses: the length-matched speakers (wav/label in [0.99, 1.01]). Gate threshold and model unchanged.
- Sensitivity (`H1_SAMPLE=all`): all measured speakers, where truncated-audio speakers use only segments inside the audio and need ≥ 60 s of them.
- Code: `measure_voice.py` drops segments beyond the wav end and records the count; `--retry-errors` re-measures error lines into `measurements_retry.jsonl`, which the analysis merges. The change only affects speakers whose segments exceed the audio; others are measured identically.

## Amendment A7 (2026-11-02, before any tone-colour or measurement value is opened; user request: predict the conversion embedding from variables)
- New target, non-gate: OpenVoice v2 tone-colour embedding (ToneColorConverter `ref_enc`, 256-d, MIT, commit 74a1d14) of each study speaker, on the same segments and channel as the measurements, whole speech and first/second half (`extract_tone_colour.py` → `results/tone_colour.jsonl`). This is the vector the converter uses for the target voice (E-VC V3).
- Sample: A6 primary (length-matched), speakers with both a measurement and a tone-colour line; `H1_SAMPLE=all` as sensitivity.
- Inputs (pools as in A1–A5): core 19; interpretable pool; interpretable minus flagged; interpretable + MFCC + LFCC; demographic baseline (gender + age band one-hot); ECAPA embedding (non-variable reference, how much one speaker embedding predicts the other).
- Models and folds identical to Gate 1: RBF-KRR (same grid, inner 5-fold) and ridge; outer 10-fold by speaker, seed 0.
- Metrics: (1) variance-weighted OOF R² on the whole-speech vector (bias-corrected bootstrap CI); (2) conversion protocol — fit on training-fold speakers (whole-speech variables → whole-speech vector), predict a test speaker's vector from its half-1 variables, score against its half-2 vector: mean cosine, EER of same-speaker verification, identification rank of the true speaker among all sample speakers' half-2 vectors (top-1, top-10, median rank).
- References in the same protocol: upper — the speaker's own half-1 vector (what the converter gets from real audio); lower — the training-fold gender-mean vector (gender conversion) and the global mean.
- No threshold; reported as the measured fraction of the audio-based upper reference. Conversion audio for real AI Hub speakers is not generated until the AI Hub terms check is done.

## Code fixes before opening (2026-11-03; from error messages only, no measured values opened)
- 7 primary-sample speakers errored inside opensauce `shrp` (IndexError in `two_max`, ValueError on an array `peak_index`). `harmonic_measures.py` now catches a `shrp` failure per segment, treats that segment's SHR as missing, and records `n_shr_failed_segments`; all other measures of the segment are unchanged. Speakers measured without the failure are unaffected and not re-measured. The failed speakers are re-measured with `--retry-errors` (the analysis keeps the last line per speaker).
- `n_segments_dropped_truncation` (A6 metadata) and `n_shr_failed_segments` are added to the metadata exclusion list in `analyze_h1.py`; the first would otherwise have entered the interpretable pool as a variable.

## Computational backend (2026-11-04, before any gate result existed)
- The CPU run of the RBF-KRR grid search had not finished the A1 pools after 14 h (machine shared with other campaigns). `krr_gpu.py` implements the identical estimator on the GPU in float64: same grid, same inner KFold(5, seed 0), same variance-weighted R² scoring, same first-best tie rule, refit on the training fold. Equivalence on a test problem (`results/krr_gpu_equivalence.json`): identical selected parameters, inner-CV score difference 3e-15, prediction difference 1.5e-13. `analyze_h1.py` and `predict_tone_colour.py` use it with `KRR_BACKEND=gpu`; nothing else changes. The interrupted CPU runs produced no result files.
- (2026-11-04) Nested forward-selection curve: the CPU loop had covered k = 5 and 10 after ~6 h. `nested_gpu.py` computes the same selection (RidgeCV GCV with intercept, same alphas, same inner folds, same first-best rule) batched over candidates, and computes the greedy path once up to k = 30 per fold, whose prefixes are exactly the paths for smaller k because greedy selection is deterministic. Equivalence (`results/nested_gpu_equivalence.json`): identical selection paths on three test problems, score difference 6e-16. Used with `KRR_BACKEND=gpu`.
