"""A5 voice-quality measures from the spectral-feature review (literature/SPECTRAL_FEATURES_REVIEW.md, C1-C5).

Reference code: opensauce-python (VoiceSauce port, Apache-2.0, commit e633608, ~/.local/share/opensauce-python):
  correction_iseli_i (Iseli & Alwan formant correction), bandwidth_hawks_miller (formula bandwidths, not LPC), shrp (Sun 2002 SHR).
C1  H1*-H2*, H2*-H4*, H4*-H2k*, H2k*-H5k  (VoiceSauce convention: H1,H2,H4 corrected for F1,F2; H2k for F1-F3; H5k uncorrected),
    plus H1*-A1*, H1*-A3* (C9). Harmonic amplitude = max of a 3-period Hamming-windowed FFT within +-10% of k*F0;
    H2k / H5k = harmonic nearest 2 kHz / 5 kHz. Per voiced 10-ms frame (Praat F0, Burg formants); speaker value = median.
C2  CoV: 50-ms moving SD / moving mean (5 frames), averaged; for F0, F1-F4, the four C1 slopes (SD only: they are dB differences),
    energy and SHR (Lee & Kreiman 2022).
C3  Band HNR after de Krom (1993): 5-period Hann frame, real cepstrum, rahmonic peaks liftered out -> noise log spectrum, baseline-corrected to the inter-harmonic troughs;
    HNR_b = 10 log10( sum P / sum P_noise ) over 0-500/1500/2500/3500 Hz. Own implementation (not VoiceSauce-identical).
C4  SHR (Sun 2002, opensauce shrp defaults: 40 ms frames, 10 ms step, threshold 0.4, ceiling 1250 Hz): mean SHR over frames
    with F0>0 and fraction of frames with SHR > 0.4.
"""
import os, sys
import numpy as np

import sys as _sys
_SIMDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # simulation/ (config.py)
if _SIMDIR not in _sys.path:
    _sys.path.append(_SIMDIR)
import config  # noqa: E402
sys.path.insert(0, config.OPENSAUCE_DIR)
from opensauce.harmonics import correction_iseli_i, bandwidth_hawks_miller  # noqa: E402
from opensauce.shrp import shrp  # noqa: E402

SR = 16000
HNR_BANDS = (500, 1500, 2500, 3500)


def _peak(spec_db, freqs, f, tol):
    m = (freqs >= f - tol) & (freqs <= f + tol)
    if not m.any():
        return np.nan, np.nan
    i = np.argmax(spec_db[m])
    return float(spec_db[m][i]), float(freqs[m][i])


def _corr(f, F, B, n):
    return sum(correction_iseli_i(np.array([f]), np.array([F[i]]), np.array([B[i]]), SR)[0] for i in range(n))


def _band_hnr(seg, f0, nfft=2048):
    x = seg * np.hanning(len(seg))
    P = np.abs(np.fft.rfft(x, nfft)) ** 2 + 1e-20
    L = np.log(P)
    full = np.concatenate([L, L[-2:0:-1]])
    C = np.fft.ifft(full).real
    q0 = SR / f0                                # quefrency of the first rahmonic, samples
    Cn = C.copy()
    k = 1
    while k * q0 < nfft / 2:
        c = int(round(k * q0)); w = max(2, int(0.1 * q0))
        Cn[max(1, c - w):c + w + 1] = 0; Cn[nfft - c - w:nfft - max(1, c - w) + 1] = 0
        k += 1
    Ln = np.fft.fft(Cn).real[:len(L)]
    fr = np.fft.rfftfreq(nfft, 1 / SR)
    # de Krom baseline correction: shift the liftered (noise) log spectrum so that it meets the original
    # spectrum at the troughs between adjacent harmonics (mean offset at the inter-harmonic minima, 0-3.5 kHz)
    d = []
    k = 1
    while (k + 1) * f0 <= HNR_BANDS[-1]:
        m = (fr > k * f0) & (fr < (k + 1) * f0)
        if m.any():
            i = np.argmin(L[m]); d.append(L[m][i] - Ln[m][i])
        k += 1
    if d:
        Ln = Ln + float(np.mean(d))
    Pn = np.exp(Ln)
    return {b: 10 * np.log10(P[fr <= b].sum() / Pn[fr <= b].sum()) for b in HNR_BANDS}


def _cov(v, win=5):
    v = np.asarray(v, float)
    if np.isfinite(v).sum() < win * 2:
        return np.nan
    out = []
    for i in range(len(v) - win + 1):
        w = v[i:i + win]
        if np.isfinite(w).all() and abs(w.mean()) > 0:
            out.append(w.std() / abs(w.mean()))
    return float(np.mean(out)) if out else np.nan


def _msd(v, win=5):
    v = np.asarray(v, float); out = [v[i:i + win].std() for i in range(len(v) - win + 1) if np.isfinite(v[i:i + win]).all()]
    return float(np.mean(out)) if out else np.nan


def voice_quality_measures(seg_audio, gender, nfft=4096):
    import parselmouth
    lo, hi = (60, 300) if gender == "남성" else (100, 500)
    ceiling = 5000 if gender == "남성" else 5500
    freqs = np.fft.rfftfreq(nfft, 1 / SR)
    n_shr_fail = [0]
    rows, seq, shr_all = [], {k: [] for k in ("f0", "F1", "F2", "F3", "F4", "e", "s1", "s2", "s3", "s4", "shr")}, []
    for y in seg_audio:
        y = np.asarray(y, np.float64)
        snd = parselmouth.Sound(y, SR)
        pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=lo, pitch_ceiling=hi)
        fo = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=ceiling)
        f0s = pitch.selected_array["frequency"]; ts = pitch.xs()
        try:
            st, sv, shr, _ = shrp(y, SR, F0MinMax=[lo, hi])
        except (IndexError, ValueError):   # opensauce shrp crashes on some frames (two_max empty index / array peak_index)
            st, sv, shr = [], [], []; n_shr_fail[0] += 1
        shr = np.asarray(shr, float); sv = np.asarray(sv, float); st = np.asarray(st, float) / 1000.0
        shr_all.append(shr[(sv > 0) & np.isfinite(shr)])
        for f0, t in zip(f0s, ts):
            nanrow = f0 <= 0
            n = int(3 * SR / f0) if not nanrow else 0
            c = int(t * SR); a, b = c - n // 2, c + n // 2
            if nanrow or a < 0 or b > len(y):
                for key in seq: seq[key].append(np.nan)
                continue
            F = [fo.get_value_at_time(i, t) for i in (1, 2, 3, 4)]
            if not all(v == v for v in F[:3]):
                for key in seq: seq[key].append(np.nan)
                continue
            B = [float(bandwidth_hawks_miller(np.array([F[i]]), np.array([f0]))[0]) for i in range(3)]
            seg = y[a:b]
            spec = 20 * np.log10(np.abs(np.fft.rfft(seg * np.hamming(len(seg)), nfft)) + 1e-12)
            tol = 0.1 * f0
            H = {k: _peak(spec, freqs, k * f0, tol) for k in (1, 2, 4)}
            h2k = _peak(spec, freqs, round(2000 / f0) * f0, tol); h5k = _peak(spec, freqs, round(5000 / f0) * f0, tol)
            A1 = _peak(spec, freqs, F[0], f0 / 2); A3 = _peak(spec, freqs, F[2], f0 / 2)
            hc = {k: H[k][0] - _corr(H[k][1], F, B, 2) for k in H}
            h2kc = h2k[0] - _corr(h2k[1], F, B, 3)
            r = {"H1c_H2c": hc[1] - hc[2], "H2c_H4c": hc[2] - hc[4], "H4c_H2kc": hc[4] - h2kc, "H2kc_H5k": h2kc - h5k[0],
                 "H1c_A1c": hc[1] - (A1[0] - _corr(A1[1], F, B, 2)), "H1c_A3c": hc[1] - (A3[0] - _corr(A3[1], F, B, 3))}
            n5 = int(5 * SR / f0); a5, b5 = c - n5 // 2, c + n5 // 2
            if a5 >= 0 and b5 <= len(y):
                r.update({f"HNR{bb // 100:02d}": v for bb, v in _band_hnr(y[a5:b5], f0).items()})
            rows.append(r)
            j = np.argmin(np.abs(st - t)) if st.size else None
            for key, v in zip(("f0", "F1", "F2", "F3", "F4", "e", "s1", "s2", "s3", "s4"),
                              (f0, *F, float(np.mean(seg ** 2)), r["H1c_H2c"], r["H2c_H4c"], r["H4c_H2kc"], r["H2kc_H5k"])):
                seq[key].append(v)
            seq["shr"].append(float(shr[j]) if j is not None and abs(st[j] - t) < 0.006 else np.nan)
        for key in seq: seq[key].append(np.nan)      # break windows across segments
    out = {}
    if rows:
        for k in rows[0]:
            vals = [r.get(k, np.nan) for r in rows]
            out[f"vq_{k}_db"] = float(np.nanmedian(vals))
        for k in HNR_BANDS:
            key = f"HNR{k // 100:02d}"
            out[f"vq_{key}_db"] = float(np.nanmedian([r.get(key, np.nan) for r in rows]))
        out["vq_n_frames"] = len(rows)
    for k in ("f0", "F1", "F2", "F3", "F4", "e", "shr"):
        out[f"cov_{k}"] = _cov(seq[k])
    for k, name in zip(("s1", "s2", "s3", "s4"), ("H1c_H2c", "H2c_H4c", "H4c_H2kc", "H2kc_H5k")):
        out[f"msd_{name}_db"] = _msd(seq[k])
    s = np.concatenate(shr_all) if shr_all else np.array([])
    out["shr_mean"] = float(s.mean()) if s.size else np.nan
    out["shr_frac_gt04"] = float((s > 0.4).mean()) if s.size else np.nan
    out["n_shr_failed_segments"] = n_shr_fail[0]   # metadata, excluded from pools (analyze_h1.META)
    return out
