# tone-colour-space

Code, dated analysis plans and aggregate results for the paper

> Sungkwan Youm. *Effective dimensionality and speaker identifiability of a voice-conversion tone-colour embedding.*
> Submitted to *Nature Communications* (under review), 2026.

Author: Sungkwan Youm, Department of Computer Engineering, Sunchon National University, Republic of Korea
(ORCID [0000-0002-9751-8732](https://orcid.org/0000-0002-9751-8732)).

The paper measures the 256-dimensional tone-colour space of the OpenVoice v2 voice converter on 2,408 Korean speaker keys:
how reliable a speaker's vector is, how much speech it needs, how many principal components identify a speaker, what
gender, age and device/SNR labels explain, and whether voices generated from the space without a person (cell means,
interpolations, Gaussian samples, principal-component moves) land near real speakers.

## What this repository contains

| Path | Contents |
|---|---|
| `papers/P1_VoiceVariableMeasurement/simulation/` | The paper's analysis, conversion, evaluation and plotting scripts (`tc_*.py`, `p1_extra_analyses.py`, `run_*.sh`). |
| `papers/P1_VoiceVariableMeasurement/simulation/results/` | Aggregate numerical results behind the figures and tables (JSON). |
| `papers/P1_VoiceVariableMeasurement/simulation/figures/` | The numbers plotted in every figure of the paper and supplement (JSON). The PNG images are produced by `tc_figures.py`. |
| `papers/P1_VoiceVariableMeasurement/PAPER_PLAN.md` | The dated P1 analysis plan (pre-registration, 2026-11-04 14:42, in Korean) with its dated execution log. |
| `simulation/stage1/` | Shared stage-1 pipeline: speaker index from the corpus labels, voice-variable measurement and ECAPA embeddings, tone-colour extraction, the pre-registered regression analyses and the GPU kernel-ridge back ends. |
| `simulation/stage1/PREREG.md` | The dated stage-1 pre-registration (fixed 2026-11-02) with its dated amendments (A1 to A7). |
| `simulation/stage1/results/` | Aggregate stage-1 results read by the figure script (`h1_results.json`, `tone_colour_prediction.json`) and the GPU/sklearn equivalence checks. |
| `simulation/eval_toolkit/` | Wrappers for the evaluation models (`metrics.py`), the stock-TTS source generator, the gender-meter trainer, `MODELS.md` (models, revisions, licences) and `VALIDATION.md` (validation of the toolkit). |
| `simulation/eval_runner.py` | Batch evaluation of converted audio (embeddings, classifiers, UTMOS, CER, re-measurement). |
| `simulation/common_data.py`, `simulation/config.py` | Shared data loader; all paths, read from environment variables with documented defaults. |
| `simulation/conversion_sources/tts_clips.json` | The three stock edge-tts voices and the five fixed Korean sentences used as conversion sources (file names only). |

The two Markdown plans are the original dated documents; only a one-line release note was added at the top of each.

The repository does **not** contain corpus audio, speaker IDs, file names or transcripts, per-speaker measurements,
per-speaker tone-colour vectors or speaker embeddings, the gender meter trained on corpus speakers, or any converted audio.

## Corpus

The speech data are the AI Hub corpus "Free conversation with emotion tags (adults)" (dataset 71631). Registered users can
request it from the National Information Society Agency of Korea (NIA) at https://www.aihub.or.kr. Its terms of use do not
permit redistributing the audio, so none is included. Download the indoor (`01.실내`) source and label archives with
the AI Hub download tool and point `PKV_DATA_ROOT` at the download root. The code reads the zip archives directly, in the
directory layout the download tool creates.

## Environment

Python 3.11.14. Key packages (pinned in `requirements.txt`): numpy 2.2.6, pandas 2.3.3, scipy 1.17.1, scikit-learn 1.6.1,
matplotlib 3.10.8, torch 2.11.0 (the study used the CUDA 13.0 build), torchaudio 2.11.0, librosa 0.11.0, soundfile 0.13.1,
praat-parselmouth 0.4.7, pyworld 0.3.5, opensmile 2.6.0, speechbrain 1.1.1, transformers 5.2.0, edge-tts 7.3.1.

```bash
pip install -r requirements.txt
git clone https://github.com/myshell-ai/OpenVoice ~/.local/share/OpenVoice              # study commit 74a1d14
git clone https://github.com/voicesauce/opensauce-python ~/.local/share/opensauce-python  # study commit e633608
```

All paths are set in `simulation/config.py`; override any of them with environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `PKV_DATA_ROOT` | AI Hub 71631 download root | `<repo>/data/raw/freetalk71631` |
| `PKV_STAGE1_RESULTS` | stage-1 outputs (per-speaker data stay here, git-ignored) | `<repo>/simulation/stage1/results` |
| `PKV_STAGE1_FIGURES` | figures written by the stage-1 scripts | `<repo>/paper_figures` |
| `PKV_MODEL_CACHE` | model cache root | `~/.cache/pkv_models` |
| `PKV_OPENVOICE_DIR` | OpenVoice source checkout | `~/.local/share/OpenVoice` |
| `PKV_OPENVOICE_CKPT` | OpenVoice v2 converter checkpoint dir (`config.json`, `checkpoint.pth`) | `$PKV_MODEL_CACHE/openvoice_v2/converter` |
| `PKV_ECAPA_DIR` | SpeechBrain ECAPA savedir | `$PKV_MODEL_CACHE/ecapa` |
| `PKV_OPENSAUCE_DIR` | opensauce-python source checkout | `~/.local/share/opensauce-python` |
| `PKV_WHISPER_CACHE` | Hugging Face hub cache for Whisper large-v3 | `~/.cache/huggingface/hub` |
| `PKV_TTS_SOURCES` | stock-TTS wavs | `<repo>/simulation/eval_toolkit/tts_sources` |
| `PKV_PYTHON` | interpreter used by the `run_*.sh` scripts | `python` |
| `PKV_GPU_LEASE` | optional GPU-lease helper used on the study's shared machine; leave unset | unset |

Hardware: the measurement and analysis steps run on CPU. `tc_gpu.py` (duration subsets) and `eval_runner.py --part gpu`
need a CUDA GPU. With `KRR_BACKEND=gpu` the kernel-ridge grid search runs in float64 through `krr_gpu.py`, on the GPU if
one is present and otherwise on the CPU. It gives the same selected hyperparameters as the scikit-learn estimator and
predictions within about 1e-8 (`simulation/stage1/results/krr_gpu_equivalence.json`). The study used this back end for
the stage-1 primary analyses and for P1.

## External models

| Use | Model | Source | Licence |
|---|---|---|---|
| Tone-colour vectors and conversion | OpenVoice v2 ToneColorConverter (`ref_enc`, tau 0.3) | github.com/myshell-ai/OpenVoice, V2 checkpoints `checkpoints_v2/converter` | MIT |
| Speaker embeddings, novelty test | ECAPA-TDNN, `speechbrain/spkrec-ecapa-voxceleb` (HF commit 0f99f2d) | huggingface.co/speechbrain | Apache-2.0 |
| Second speaker-verification model | WavLM-base-plus-SV, `microsoft/wavlm-base-plus-sv` (HF commit feb593a) | huggingface.co/microsoft | CC BY-SA 3.0 (UniSpeech licence named on the model card) |
| Naturalness (relative only) | UTMOS22 strong learner via SpeechMOS v1.2.0 (`tarepan/SpeechMOS`, `utmos22_strong`) | torch.hub / GitHub | MIT |
| Character error rate | Whisper large-v3, `openai/whisper-large-v3` (HF commit 06f233f) | huggingface.co/openai | Apache-2.0 |
| Anti-spoofing score (computed by `eval_runner`, not interpreted in the paper) | AASIST, ASVspoof 2019 LA weights (commit a04c986); place the clone at `$PKV_MODEL_CACHE/aasist_src/aasist` | github.com/clovaai/aasist | MIT |
| Gender classifier | `audeering/wav2vec2-large-robust-24-ft-age-gender` (HF commit bad2737) | huggingface.co/audeering | CC BY-NC-SA 4.0 (non-commercial) |
| Gender classifier | Balanced logistic regression on ECAPA embeddings of 300 non-study corpus speakers; retrain with `eval_toolkit/train_gender_meter.py` (the trained model is not distributed because it is derived from corpus speakers) | this repository | MIT (code) |
| Voice measurement | Praat via praat-parselmouth; WORLD via pyworld; opensauce-python; openSMILE eGeMAPSv02 via opensmile | PyPI / GitHub | GPL-3.0; MIT; Apache-2.0; openSMILE research licence (non-commercial) |
| Stock TTS sources | Microsoft Edge read-aloud voices ko-KR SunHi, InJoon, HyunsuMultilingual via edge-tts 7.3.1 (only the sentence text is sent) | PyPI `edge-tts` | edge-tts LGPL-3.0; voice output under Microsoft's service terms |

Full revisions and notes: `simulation/eval_toolkit/MODELS.md`. Toolkit validation: `simulation/eval_toolkit/VALIDATION.md`.

## Running the analyses

Steps 1 to 8 build the per-speaker data from the corpus. They are shared with the companion analyses and follow
`simulation/stage1/PREREG.md`. Steps 9 to 16 are the paper (`PAPER_PLAN.md`). All random choices use seed 0.

```bash
export PKV_DATA_ROOT=/path/to/aihub71631        # plus any other PKV_* overrides

# ---- stage 1 (cwd: simulation/stage1) ----
cd simulation/stage1
python build_speaker_index.py                    # 1 speaker keys, best-SNR indoor conversation, segments -> speaker_index.parquet
python measure_voice.py --workers 4              # 2 voice variables + ECAPA (whole, half 1, half 2) -> measurements.jsonl (CPU, slow; resumable)
python measure_voice.py --workers 4 --retry-errors   #   retry pass -> measurements_retry.jsonl
python check_audio_label_duration.py             # 3 A6 table: wav vs label length -> audio_label_duration_check.csv (primary sample = 2,410 keys)
python extract_tone_colour.py --workers 2        # 4 OpenVoice tone-colour vectors (whole, half 1, half 2) -> tone_colour.jsonl (CPU)
KRR_BACKEND=gpu H1_SAMPLE=primary python analyze_h1.py           # 5 -> h1_results.json, h1_oof_predictions.npz
KRR_BACKEND=gpu H1_SAMPLE=primary python predict_tone_colour.py  # 6 -> tone_colour_prediction.json

# ---- evaluation toolkit (cwd: simulation/eval_toolkit) ----
cd ../eval_toolkit
python make_tts_sources.py                       # 7 three ko-KR edge-tts voices x 5 sentences -> tts_sources/
python train_gender_meter.py                     # 8 ECAPA gender meter on non-study speakers -> gender_meter_balanced.joblib

# ---- the paper (cwd: papers/P1_VoiceVariableMeasurement/simulation) ----
cd ../../papers/P1_VoiceVariableMeasurement/simulation
python tc_analysis.py                            # 9  A1, B1-B4, C1, D0, E(a-c), conversion targets -> results/tc_analysis.json (CPU)
bash run_tc_gpu.sh                               # 10 A2 duration subsets, E(d,e) kernel ridge -> results/tc_gpu.json (CUDA)
bash run_tc_convert_eval.sh                      # 11 1,251 conversions of the 3 TTS sources -> outputs/, then eval_runner --part gpu (CUDA)
python -c "import json; r = json.load(open('outputs/manifest.json')); json.dump([x for x in r if x['cond'] in ('E_pc', 'N0', 'SRC')], open('outputs/manifest_remeasure_E.json', 'w'), ensure_ascii=False, indent=1)"
python ../../../simulation/eval_runner.py --manifest outputs/manifest_remeasure_E.json --out results/eval/p1tc --part remeasure   # 12 re-measure the 54 PC-move demo rows (CPU)
python tc_x1_short_real.py                       # 13 exploratory X1: 5-s real clips -> results/tc_x1_short_real.json (CPU)
python tc_eval.py                                # 14 novelty, real-target conversions, classifiers -> results/tc_eval.json + figures
python tc_figures.py                             # 15 remaining figures -> figures/*.png + .json (+ main-text .pdf)
bash run_p1_extra.sh                             # 16 pre-declared non-gate diagnostics S1-S4 -> results/p1_extra_analyses.json
```

Notes:

- Step 12's sub-manifest rule (PC-move demos `E_pc`, self-conversions `N0` and the unconverted sources `SRC`, 54 rows) is the
  rule recorded in the execution log of `PAPER_PLAN.md`. The one-liner reproduces the study's sub-manifest exactly.
- In the study the first 319 conversions of step 11 ran on CPU and the rest on CUDA (recorded per row in the manifest as
  `conv_device`, see the execution log in `PAPER_PLAN.md`). A fresh run on one device gives equivalent but not
  bit-identical audio.
- The edge-tts voices are a remote service. The voices on offer, or their output, may change over time, which would change
  the conversion results slightly. The vector-space analyses (steps 9, 10, 16) do not use them.
- `results/tc_x1_short_real_vs_whole_v1.json` came from the first version of the exploratory X1 analysis, which compared
  the 5-s clips with whole-speech embeddings. The current `tc_x1_short_real.py` recomputes the same quantities. They are the
  `own_whole`, `max_other_whole` and `max_any_whole_including_own` entries of `tc_x1_short_real.json`, and they are
  identical to the v1 file.
- `simulation/stage1/check_audio_label_duration.py` was written for this release. In the study the A6 table was made by an
  unsaved command. On the full corpus this script reproduces the study table exactly, with the same 2,410 primary keys.
- Steps 1 to 8 and 11 to 13 need the corpus. Steps 14 to 16 need the per-speaker outputs of the earlier steps. `tc_figures.py`
  needs only the aggregate JSON files, so it can be re-run on the files in this repository.

## Acknowledgement

This research used datasets from The Open AI Dataset Project (AI-Hub, S. Korea). All data information can be accessed
through AI-Hub (www.aihub.or.kr). This paper is a result of using the AI Hub data "Free conversation with emotion tags
(adults)" of the National Information Society Agency (NIA) of Korea.

## Ethics

Voice conversion can be misused to imitate real people. The study converted only the three stock text-to-speech voices.
Conversions toward real corpus speakers (the real-target conditions `C2_k*` and the interpolations `D2`) were made solely
to measure how close such conversions come to their targets. They are for evaluation only and must stay on the
corpus holder's storage. No converted audio of any kind, and no corpus audio, is distributed with this repository.
If you re-run the pipeline, keep the outputs under the AI Hub terms of use and do not publish conversions toward corpus
speakers.

## Licence

Code: MIT (see `LICENSE`). The corpus and the external models are under their own terms (see above).
