"""Stage-1 voice-variable measurement and speaker embeddings (PREREG.md, fixed 2026-11-02).

Per eligible speaker: read the conversation wav straight from the AI Hub zip (16 kHz stereo; one speaker per channel),
pick the speaker's channel by energy over the speaker's own label segments, cut the segments from speaker_index.parquet,
and measure the 17 candidate variables plus ECAPA-TDNN embeddings (mean, and first/second half for the reliability ceiling).
Resumable: one JSON line per speaker in results/measurements.jsonl.
Usage: python measure_voice.py [--workers 4] [--limit N]
"""
import argparse, glob, io, json, os, zipfile
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from harmonic_measures import voice_quality_measures
import pandas as pd

import sys as _sys
_SIMDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # simulation/ (config.py)
if _SIMDIR not in _sys.path:
    _sys.path.append(_SIMDIR)
import config  # noqa: E402
RAW = config.DATA_ROOT
OUT = config.STAGE1_RESULTS
SR = 16000
EGEMAPS = {"h1_h2_db": "logRelF0-H1-H2_sma3nz_amean", "slope_0_500": "slopeV0-500_sma3nz_amean",
           "slope_500_1500": "slopeV500-1500_sma3nz_amean", "alpha_ratio_db": "alphaRatioV_sma3nz_amean"}


def zip_map():
    m = {}
    for zp in glob.glob(os.path.join(RAW, "*/*/*/01.원천데이터/*.zip")):
        if glob.glob(zp + ".part*"):  # still being merged by aihubshell
            continue
        for n in zipfile.ZipFile(zp).namelist():
            if n.endswith(".wav"):
                m[os.path.basename(n)[:-4]] = (zp, n)
    return m


def pick_channel(x, segs):
    idx = np.concatenate([np.arange(int(s["start"] * SR), min(int(s["end"] * SR), len(x))) for s in segs])
    e = (x[idx] ** 2).mean(0)
    ch = int(np.argmax(e))
    return ch, float(10 * np.log10(e[ch] / max(e[1 - ch], 1e-12)))


def praat_measures(seg_audio, gender):
    """Core 19-set keys are unchanged; extended keys added 2026-11-02 (user request: consider every variable)."""
    import parselmouth
    from parselmouth.praat import call
    lo, hi = (60, 300) if gender == "남성" else (100, 500)
    ceiling = 5000 if gender == "남성" else 5500
    f0_all, finals, slopes, fm, bw = [], [], [], [[] for _ in range(4)], [[] for _ in range(4)]
    J = {k: [] for k in ("local", "rap", "ppq5")}
    S = {k: [] for k in ("local_dB", "local", "apq3", "apq5", "apq11")}
    hnr, cpps, inten, voiced_frac, pause_frac = [], [], [], [], []
    for y in seg_audio:
        snd = parselmouth.Sound(y, SR)
        pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=lo, pitch_ceiling=hi)
        f0 = pitch.selected_array["frequency"]; t = pitch.xs()
        v = f0 > 0
        it = snd.to_intensity(minimum_pitch=lo, time_step=0.01)
        iv = it.values[0]
        if iv.size:
            inten.append(iv); pause_frac.append(float((iv < iv.max() - 25).mean()))
        voiced_frac.append(float(v.mean()) if v.size else 0.0)
        if v.sum() < 10:
            continue
        f0_all.append(f0[v])
        stv = 12 * np.log2(f0[v] / 100.0)
        slopes.append(np.polyfit(t[v], stv, 1)[0])
        tail = v & (t >= t[-1] - 0.3)
        if tail.sum() >= 5:
            finals.append(np.polyfit(t[tail], 12 * np.log2(f0[tail] / 100.0), 1)[0])
        pp = call(snd, "To PointProcess (periodic, cc)", lo, hi)
        for k in J:
            J[k].append(call(pp, f"Get jitter ({k})", 0, 0, 0.0001, 0.02, 1.3) * 100)
        for k in S:
            val = call([snd, pp], f"Get shimmer ({k})", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            S[k].append(val if k == "local_dB" else val * 100)
        h = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=lo)
        hv = h.values[h.values > -200]
        if hv.size:
            hnr.append(float(hv.mean()))
        pc = call(snd, "To PowerCepstrogram", 60, 0.002, 5000, 50)
        cpps.append(call(pc, "Get CPPS", "yes", 0.02, 0.0005, 60, 330, 0.05, "parabolic", 0.001, 0.05, "Exponential decay", "Robust"))
        fo = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=ceiling)
        for tt in t[v]:
            for i in range(4):
                val = fo.get_value_at_time(i + 1, tt)
                if val == val:
                    fm[i].append(val)
                b = fo.get_bandwidth_at_time(i + 1, tt)
                if b == b:
                    bw[i].append(b)
    f0c = np.concatenate(f0_all)
    stc = 12 * np.log2(f0c / 100.0)
    F = [float(np.median(f)) for f in fm]
    k = np.array([(2 * i + 1) / 2 for i in range(4)])
    dF = float((np.array(F) * k).sum() / (k ** 2).sum())
    ic = np.concatenate(inten) if inten else np.array([np.nan])
    med = lambda a: float(np.nanmedian(a)) if len(a) else float("nan")
    out = {"f0_mean_hz": float(f0c.mean()), "f0_range_st": float(12 * np.log2(np.percentile(f0c, 95) / np.percentile(f0c, 5))),
           "jitter_pct": med(J["local"]), "shimmer_db": med(S["local_dB"]), "hnr_db": med(hnr),
           "F1_hz": F[0], "F2_hz": F[1], "F3_hz": F[2], "F4_hz": F[3], "vtl_cm": 35000.0 / (2 * dF),
           "final_f0_slope_st_s": float(np.mean(finals)) if finals else float("nan"),
           "n_voiced_frames": int(f0c.size)}
    # extended set
    out.update({"f0_median_hz": float(np.median(f0c)), "f0_p5_hz": float(np.percentile(f0c, 5)), "f0_p95_hz": float(np.percentile(f0c, 95)),
                "f0_sd_st": float(stc.std()), "f0_mean_slope_st_s": float(np.mean(slopes)),
                "jitter_rap_pct": med(J["rap"]), "jitter_ppq5_pct": med(J["ppq5"]),
                "shimmer_local_pct": med(S["local"]), "shimmer_apq3_pct": med(S["apq3"]), "shimmer_apq5_pct": med(S["apq5"]),
                "shimmer_apq11_pct": med(S["apq11"]), "cpps_db": med(cpps),
                "B1_hz": med(bw[0]), "B2_hz": med(bw[1]), "B3_hz": med(bw[2]), "B4_hz": med(bw[3]),
                "formant_dispersion_hz": float((F[3] - F[0]) / 3), "F1_sd_hz": float(np.std(fm[0])), "F2_sd_hz": float(np.std(fm[1])),
                "intensity_mean_db": float(np.nanmean(ic)), "intensity_sd_db": float(np.nanstd(ic)),
                "voiced_fraction": float(np.mean(voiced_frac)), "pause_fraction": float(np.mean(pause_frac))})
    return out


def spectral_measures(y):
    from scipy.signal import welch
    f, p = welch(y, SR, nperseg=1024)
    tot = p.sum()
    out = {"low_band_0_1k_db": float(10 * np.log10(p[f <= 1000].sum() / tot)),
           "spk_formant_2_5_3_5k_db": float(10 * np.log10(p[(f >= 2500) & (f <= 3500)].sum() / tot)),
           "spectral_centroid_hz": float((f * p).sum() / tot)}
    # extended: long-term average spectrum in 16 bands of 500 Hz (dB re total), rolloff, flatness, spread
    for b in range(16):
        m = (f >= b * 500) & (f < (b + 1) * 500 if b < 15 else f <= 8000)
        out[f"ltas_{b * 500}_{(b + 1) * 500}hz_db"] = float(10 * np.log10(p[m].sum() / tot + 1e-12))
    c = np.cumsum(p) / tot
    out["spectral_rolloff85_hz"] = float(f[np.searchsorted(c, 0.85)])
    out["spectral_flatness"] = float(np.exp(np.mean(np.log(p + 1e-20))) / np.mean(p))
    out["spectral_spread_hz"] = float(np.sqrt(((f - out["spectral_centroid_hz"]) ** 2 * p).sum() / tot))
    # A4: 1/3-octave LTAS (centres 100 Hz - 6.3 kHz), dB re total
    for fc in THIRD_OCT:
        m = (f >= fc / 2 ** (1 / 6)) & (f < fc * 2 ** (1 / 6))
        out[f"ltas3oct_{int(fc)}hz_db"] = float(10 * np.log10(p[m].sum() / tot + 1e-12))
    return out


THIRD_OCT = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300]


def envelope_measures(seg_audio, gender):
    """A4: WORLD spectral envelope (CheapTrick on DIO+StoneMask F0) averaged over voiced frames, in 1/3-octave bands
    (dB re the envelope's mean level), plus MFCC 1-20 mean and SD over all speech frames."""
    import pyworld, librosa
    lo, hi = (60, 300) if gender == "남성" else (100, 500)
    envs, mf = [], []
    for y in seg_audio:
        x = y.astype(np.float64)
        f0, t = pyworld.dio(x, SR, f0_floor=lo, f0_ceil=hi, frame_period=10)
        f0 = pyworld.stonemask(x, f0, t, SR)
        sp = pyworld.cheaptrick(x, f0, t, SR, f0_floor=lo)
        if (f0 > 0).sum() >= 5:
            envs.append(sp[f0 > 0])
        mf.append(librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20, n_fft=512, hop_length=160))
    out = {}
    if envs:
        E = np.vstack(envs).mean(0)
        fr = np.linspace(0, SR / 2, E.size)
        ref = E[(fr >= 80) & (fr <= 7000)].mean()
        for fc in THIRD_OCT:
            m = (fr >= fc / 2 ** (1 / 6)) & (fr < fc * 2 ** (1 / 6))
            out[f"env3oct_{int(fc)}hz_db"] = float(10 * np.log10(E[m].mean() / ref)) if m.any() else float("nan")
        # A5 C5: ERB-rate 1-ERB bands 20-33 (~1.7-7.7 kHz), dB re envelope mean
        erb = 21.4 * np.log10(1 + 0.00437 * fr)
        for e in range(20, 33):
            m = (erb >= e) & (erb < e + 1)
            out[f"envERB{e}_db"] = float(10 * np.log10(E[m].mean() / ref)) if m.any() else float("nan")
    M = np.hstack(mf)
    # A5 C5: LFCC 1-20 (40 linear triangular filters 0-8 kHz on the 512-pt STFT power, log, DCT-II)
    from scipy.fft import dct
    edges = np.linspace(0, SR / 2, 42); fb = np.zeros((40, 257)); fq = np.linspace(0, SR / 2, 257)
    for j in range(40):
        l, c, r = edges[j], edges[j + 1], edges[j + 2]
        fb[j] = np.clip(np.minimum((fq - l) / (c - l), (r - fq) / (r - c)), 0, None)
    L = []
    for y in seg_audio:
        S = np.abs(librosa.stft(y, n_fft=512, hop_length=160)) ** 2
        L.append(dct(np.log(fb @ S + 1e-10), type=2, axis=0, norm="ortho")[1:21])
    L = np.hstack(L)
    for i in range(20):
        out[f"lfcc{i + 1}_mean"] = float(L[i].mean()); out[f"lfcc{i + 1}_sd"] = float(L[i].std())
    for i in range(20):
        out[f"mfcc{i + 1}_mean"] = float(M[i].mean()); out[f"mfcc{i + 1}_sd"] = float(M[i].std())
    return out


_SMILE = _ECAPA = None


def init_worker():
    global _SMILE, _ECAPA
    import torch, opensmile
    from speechbrain.inference.speaker import EncoderClassifier
    torch.set_num_threads(1)
    _SMILE = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
    _ECAPA = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                            savedir=config.ECAPA_DIR, run_opts={"device": "cpu"})


def embed(segs):
    import torch
    out = []
    for y in segs:
        with torch.no_grad():
            out.append(_ECAPA.encode_batch(torch.tensor(y, dtype=torch.float32)[None]).squeeze().numpy())
    return np.array(out)


def l2mean(e):
    m = e.mean(0)
    return (m / np.linalg.norm(m)).round(6).tolist()


def measure(task):
    row, (zp, member) = task
    import soundfile as sf
    try:
        x, sr = sf.read(io.BytesIO(zipfile.ZipFile(zp).read(member)), dtype="float32")
        assert sr == SR and x.ndim == 2, (sr, x.shape)
        segs = json.loads(row["segments"])
        # A6: some wavs are shorter than the label FileLength (truncated); keep only segments inside the audio
        n_all = len(segs); dur = len(x) / SR
        segs = [s for s in segs if s["end"] <= dur]
        if sum(s["end"] - s["start"] for s in segs) < 60:
            return {"speaker": row["speaker"], "error": f"insufficient in-bounds speech after truncation ({len(segs)}/{n_all} segments, wav {dur:.1f} s)"}
        ch, sep_db = pick_channel(x, segs)
        audio = [x[int(s["start"] * SR):int(s["end"] * SR), ch] for s in segs]
        cat = np.concatenate(audio)
        res = {"speaker": row["speaker"], "channel": ch, "channel_sep_db": round(sep_db, 2), "speech_s": len(cat) / SR,
               "n_segments_dropped_truncation": n_all - len(segs)}
        h = len(audio) // 2

        def variables(aud, sg):
            c = np.concatenate(aud)
            v = {"speech_rate_syl_s": sum(s["syllables"] for s in sg) / (len(c) / SR)}
            v.update(praat_measures(aud, row["gender"]))
            sm = _SMILE.process_signal(c, SR)
            v.update({k: float(sm[col].iloc[0]) for k, col in EGEMAPS.items()})
            v.update({"egemaps__" + col: float(sm[col].iloc[0]) for col in sm.columns})  # all 88 eGeMAPSv02 functionals
            v.update(spectral_measures(c))
            v.update(envelope_measures(aud, row["gender"]))
            v.update(voice_quality_measures(aud, row["gender"]))
            return v
        res.update(variables(audio, segs))
        # halves (first/second half of the segments) for split-half reliability and variable-only verification
        res["half1"] = variables(audio[:h], segs[:h])
        res["half2"] = variables(audio[h:], segs[h:])
        e = embed(audio)
        res.update({"ecapa": l2mean(e), "ecapa_half1": l2mean(e[:h]), "ecapa_half2": l2mean(e[h:]), "n_segments": len(e)})
        return res
    except Exception as ex:  # recorded, not hidden
        return {"speaker": row["speaker"], "error": f"{type(ex).__name__}: {ex}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retry-errors", action="store_true", help="re-measure speakers whose line in measurements.jsonl is an error; write to measurements_retry.jsonl")
    a = ap.parse_args()
    idx = pd.read_parquet(os.path.join(OUT, "speaker_index.parquet"))
    idx = idx[idx.eligible]
    zm = zip_map()
    path = os.path.join(OUT, "measurements.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["speaker"] for l in open(path)}
    if a.retry_errors:
        errs = {d["speaker"] for d in map(json.loads, open(path)) if "error" in d}
        path = os.path.join(OUT, "measurements_retry.jsonl")
        redone = {d["speaker"] for d in map(json.loads, open(path)) if "error" not in d} if os.path.exists(path) else set()  # error lines are retried again; the analysis keeps the last line per speaker
        done = (done - errs) | redone
    tasks = [(r, zm[r["file"]]) for r in idx.to_dict("records") if r["file"] in zm and r["speaker"] not in done]
    if a.limit:
        tasks = tasks[:a.limit]
    print(f"eligible {len(idx)}, audio available {sum(f in zm for f in idx.file)}, done {len(done)}, to run {len(tasks)}", flush=True)
    with ProcessPoolExecutor(a.workers, initializer=init_worker) as ex, open(path, "a") as f:
        for i, res in enumerate(ex.map(measure, tasks, chunksize=1)):
            f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
            if i % 50 == 0:
                print(i, res.get("speaker"), res.get("error", "ok"), flush=True)


if __name__ == "__main__":
    main()
