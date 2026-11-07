"""Gate 1 / H1 analysis exactly as fixed in PREREG.md.

Out-of-fold (10-fold, split by speaker, seed 0) variance-weighted multi-output R² of the 192-d ECAPA speaker
embedding from the core 19 standardised voice variables (PREREG.md; amendments A1, A2 add non-gate analyses). Primary: RBF kernel ridge with inner 5-fold CV; linear ridge alongside.
Gate 1 passes if the primary R² >= 0.70. Outputs results/h1_*.json and paper_figures/*.png (+ .json pairs).
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.metrics import make_scorer, r2_score, roc_curve
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys as _sys
_SIMDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # simulation/ (config.py)
if _SIMDIR not in _sys.path:
    _sys.path.append(_SIMDIR)
import config  # noqa: E402
RES = config.STAGE1_RESULTS
FIG = config.STAGE1_FIGURES
VARS = ["f0_mean_hz", "f0_range_st", "jitter_pct", "shimmer_db", "hnr_db", "h1_h2_db", "slope_0_500", "slope_500_1500",
        "alpha_ratio_db", "F1_hz", "F2_hz", "F3_hz", "F4_hz", "vtl_cm", "low_band_0_1k_db", "spk_formant_2_5_3_5k_db",
        "spectral_centroid_hz", "speech_rate_syl_s", "final_f0_slope_st_s"]
GATE = 0.70
SAMPLE = os.environ.get("H1_SAMPLE", "primary")   # A6: "primary" (length-matched audio) or "all" (sensitivity)
SEED = 0
META = {"channel", "channel_sep_db", "speech_s", "n_segments", "n_voiced_frames", "n_segments_dropped_truncation", "n_shr_failed_segments"}
HUCKVALE_LIKE = ["f0_mean_hz", "f0_range_st", "vtl_cm", "formant_dispersion_hz", "hnr_db", "cpps_db", "slope_0_500", "h1_h2_db",
                 "speech_rate_syl_s", "intensity_mean_db"]


def design(df, cols=VARS):
    X = df[cols].astype(float).copy()
    for c in ("f0_mean_hz", "f0_median_hz", "f0_p5_hz", "f0_p95_hz"):
        if c in X:
            X[c] = np.log(X[c])
    return X.values


def pools(df):
    """A1: interpretable pool = every scalar measured variable that is not metadata or an eGeMAPS functional."""
    interp = [c for c in df.columns if c not in META and not c.startswith("egemaps__") and not c.startswith(("mfcc", "lfcc", "vq_n_frames"))
              and c not in ("ecapa", "ecapa_half1", "ecapa_half2", "half1", "half2", "gender", "age", "device", "snr_db", "split", "error")
              and pd.api.types.is_numeric_dtype(df[c])]
    eg = [c for c in df.columns if c.startswith("egemaps__")]
    return interp, eg


def spectral_pools(df):
    """A4: spectral-detail pools."""
    return {"env3oct": [c for c in df.columns if c.startswith("env3oct_")],
            "ltas3oct": [c for c in df.columns if c.startswith("ltas3oct_")],
            "mfcc": [c for c in df.columns if c.startswith("mfcc")],
            "lfcc": [c for c in df.columns if c.startswith("lfcc")],
            "envERB_hi": [c for c in df.columns if c.startswith("envERB")],
            "voice_quality_A5": [c for c in df.columns if c.startswith(("vq_", "cov_", "msd_", "shr_")) and c != "vq_n_frames"]}


# A5 reliability flags from literature/SPECTRAL_FEATURES_REVIEW.md section 5 (assigned before opening)
FLAG_PATTERNS = {"perturbation@16kHz": ("jitter", "shimmer", "egemaps__jitter", "egemaps__shimmer"),
                 "LPC-bandwidth": ("B1_", "B2_", "B3_", "B4_", "egemaps__F1bandwidth", "egemaps__F2bandwidth", "egemaps__F3bandwidth"),
                 "near-Nyquist": ("ltas_7000_7500", "ltas_7500_8000")}


def flagged(cols):
    return [c for c in cols if any(c.startswith(p) or p in c for ps in FLAG_PATTERNS.values() for p in ps)]


def r2(Y, P):
    return float(r2_score(Y, P, multioutput="variance_weighted"))


def krr():
    if os.environ.get("KRR_BACKEND") == "gpu":   # identical estimator on GPU (krr_gpu.py; equivalence: results/krr_gpu_equivalence.json)
        from krr_gpu import GPUKRRGrid
        return GPUKRRGrid([1e-3, 1e-2, 1e-1, 1], [0.005, 0.01, 0.02, 0.05, 0.1], SEED)
    g = GridSearchCV(KernelRidge(kernel="rbf"), {"alpha": [1e-3, 1e-2, 1e-1, 1], "gamma": [0.005, 0.01, 0.02, 0.05, 0.1]},
                     cv=KFold(5, shuffle=True, random_state=SEED),
                     scoring=make_scorer(r2_score, multioutput="variance_weighted"))  # A2: same metric as the outer R²
    return g


def oof(X, Y, model_fn, impute=True):
    P = np.zeros_like(Y); params = []
    for tr, te in KFold(10, shuffle=True, random_state=SEED).split(X):
        Xtr, Xte = X[tr].copy(), X[te].copy()
        if impute:
            med = np.nanmedian(Xtr, 0)
            Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
        sc = StandardScaler().fit(Xtr)
        m = model_fn().fit(sc.transform(Xtr), Y[tr])
        P[te] = m.predict(sc.transform(Xte))
        params.append(getattr(m, "best_params_", None))
    return P, params


def boot_ci(Y, P, B=2000):
    rng = np.random.default_rng(SEED); n = len(Y)
    v = np.array([r2(Y[i], P[i]) for i in (rng.integers(0, n, n) for _ in range(B))])
    # bias-corrected percentile interval (Efron 1987): resampling with replacement shrinks the total variance by (n-1)/n,
    # which shifts every resampled R2 down by about (1-R2)/n; the plain percentile interval can then exclude the estimate.
    from scipy.stats import norm
    z0 = norm.ppf(np.clip((v < r2(Y, P)).mean(), 1 / B, 1 - 1 / B))
    lo, hi = norm.cdf(2 * z0 + norm.ppf(0.025)), norm.cdf(2 * z0 + norm.ppf(0.975))
    return [float(np.percentile(v, 100 * lo)), float(np.percentile(v, 100 * hi))]


def eer(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    return float((fpr[i] + 1 - tpr[i]) / 2)


def verification(A, B):
    """enrol on A[i], test on B[j]; target = same speaker."""
    A = A / np.linalg.norm(A, axis=1, keepdims=True); B = B / np.linalg.norm(B, axis=1, keepdims=True)
    S = A @ B.T
    lab = np.eye(len(A), dtype=bool)
    rng = np.random.default_rng(SEED)
    non = S[~lab]; non = rng.choice(non, min(len(non), 200000), replace=False)
    sc = np.concatenate([np.diag(S), non]); lb = np.concatenate([np.ones(len(A)), np.zeros(len(non))])
    return eer(sc, lb)


def main():
    idx = pd.read_parquet(os.path.join(RES, "speaker_index.parquet")).set_index("speaker")
    rows = [json.loads(l) for l in open(os.path.join(RES, "measurements.jsonl"))]
    rp = os.path.join(RES, "measurements_retry.jsonl")
    if os.path.exists(rp):                      # A6: retried speakers replace their error line
        retry = {r["speaker"]: r for r in map(json.loads, open(rp))}
        rows = [retry.get(r["speaker"], r) if "error" in r else r for r in rows]
    err = [r for r in rows if "error" in r]
    ok = [r for r in rows if "error" not in r]
    df = pd.DataFrame(ok).set_index("speaker").join(idx[["gender", "age", "device", "snr_db", "split"]])
    # A6 primary sample: wav length within 1% of label FileLength (results/audio_label_duration_check.csv)
    chk = pd.read_csv(os.path.join(RES, "audio_label_duration_check.csv")).set_index("speaker")
    matched = chk.index[(chk.wav_over_label >= 0.99) & (chk.wav_over_label <= 1.01)]
    n_trunc_ok = int((~df.index.isin(matched)).sum())
    if SAMPLE == "primary":
        df = df[df.index.isin(matched)]
    Y = np.vstack(df.ecapa.values)
    X = design(df)
    out = {"sample": SAMPLE, "n_measured": len(rows), "n_errors": len(err), "n_used": len(df), "n_truncated_audio_measured": n_trunc_ok,
           "errors_sample": [e["error"] for e in err[:5]], "gate_threshold": GATE}

    P_k, params = oof(X, Y, krr)
    P_l, _ = oof(X, Y, lambda: RidgeCV(alphas=np.logspace(-3, 3, 13)))
    out["primary_krr_r2"] = r2(Y, P_k); out["primary_krr_r2_ci95"] = boot_ci(Y, P_k)
    out["krr_inner_params"] = params
    out["linear_ridge_r2"] = r2(Y, P_l); out["linear_ridge_r2_ci95"] = boot_ci(Y, P_l)
    out["gate1_pass"] = bool(out["primary_krr_r2"] >= GATE)

    # nuisance (channel) and variables + nuisance
    oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit_transform(df[["device"]])
    N = np.column_stack([oh, df.snr_db.values])
    P_n, _ = oof(N, Y, lambda: RidgeCV(alphas=np.logspace(-3, 3, 13)))
    P_vn, _ = oof(np.column_stack([X, N]), Y, krr)
    out["nuisance_only_r2"] = r2(Y, P_n); out["variables_plus_nuisance_krr_r2"] = r2(Y, P_vn)

    # reliability ceiling: half-1 embedding as the prediction of half-2 (and the mean)
    H1, H2 = np.vstack(df.ecapa_half1.values), np.vstack(df.ecapa_half2.values)
    out["split_half_embedding_r2"] = r2(H2, H1)
    # variables of half 1 predicting the half-2 embedding with the primary model (same folds)
    Xh1 = design(pd.DataFrame(list(df.half1.values)))
    Xh2 = design(pd.DataFrame(list(df.half2.values)))
    P_h, _ = oof(Xh1, H2, krr)
    out["variables_half1_to_embedding_half2_krr_r2"] = r2(H2, P_h)

    # within gender
    out["within_gender"] = {}
    for g, sub in df.groupby("gender"):
        if len(sub) < 100:
            continue
        m = df.index.isin(sub.index)
        Pg, _ = oof(X[m], Y[m], krr)
        out["within_gender"][g] = {"n": int(m.sum()), "krr_r2": r2(Y[m], Pg), "ci95": boot_ci(Y[m], Pg)}

    # sensitivity: channel separation >= 10 dB
    m = (df.channel_sep_db >= 10).values
    Ps, _ = oof(X[m], Y[m], krr)
    out["sensitivity_sep_ge_10db"] = {"n": int(m.sum()), "krr_r2": r2(Y[m], Ps)}

    # verification EER: variables (standardised on all, NaN -> median) vs ECAPA, enrol half 1 / test half 2
    allh = np.vstack([Xh1, Xh2]); med = np.nanmedian(allh, 0); sc = StandardScaler().fit(np.where(np.isnan(allh), med, allh))
    f = lambda Z: sc.transform(np.where(np.isnan(Z), med, Z))
    out["eer_variables"] = verification(f(Xh1), f(Xh2))
    out["eer_ecapa"] = verification(H1, H2)
    out["variable_split_half_r"] = {v: float(pd.Series(Xh1[:, i]).corr(pd.Series(Xh2[:, i]))) for i, v in enumerate(VARS)}

    # contributions: drop-one with the primary model at the median selected hyperparameters
    a = float(np.median([p["alpha"] for p in params])); gm = float(np.median([p["gamma"] for p in params]))
    fixed = lambda: KernelRidge(kernel="rbf", alpha=a, gamma=gm)
    base = r2(Y, oof(X, Y, fixed)[0])
    out["drop_one_delta_r2"] = {v: base - r2(Y, oof(np.delete(X, i, 1), Y, fixed)[0]) for i, v in enumerate(VARS)}
    pcs = np.linalg.svd(Y - Y.mean(0), full_matrices=False)
    Z = (Y - Y.mean(0)) @ pcs[2][:10].T
    Xi = np.where(np.isnan(X), np.nanmedian(X, 0), X)
    out["mi_with_top10_pcs"] = {v: float(sum(mutual_info_regression(Xi[:, [i]], Z[:, k], random_state=SEED)[0] for k in range(10)))
                                for i, v in enumerate(VARS)}
    out["pc_variance_share_top10"] = (pcs[1][:10] ** 2 / (pcs[1] ** 2).sum()).round(4).tolist()
    out["variable_correlation"] = pd.DataFrame(Xi, columns=VARS).corr().round(3).to_dict()
    out["sample"] = {"gender": df.gender.value_counts().to_dict(), "age": df.age.value_counts().to_dict(),
                     "device": df.device.value_counts().to_dict(), "channel_sep_db_median": float(df.channel_sep_db.median())}
    # ---------------- Amendment A1/A2 (non-gate) ----------------
    interp, eg = pools(df)
    fixed_rbf = lambda: KernelRidge(kernel="rbf", alpha=a, gamma=gm)
    A = {"interpretable_pool_size": len(interp), "egemaps_pool_size": len(eg)}
    lt = [c for c in interp if c.startswith("ltas_")]
    for name, cols in {"interpretable_pool": interp, "interpretable_pool_without_ltas": [c for c in interp if c not in lt],
                       "interpretable_plus_egemaps": interp + eg, "huckvale_like_subset": [c for c in HUCKVALE_LIKE if c in df]}.items():
        Pp, pp = oof(design(df, cols), Y, krr)
        A[name] = {"n_vars": len(cols), "krr_r2": r2(Y, Pp), "ci95": boot_ci(Y, Pp)}
    sp = spectral_pools(df)
    A["flagged_in_interpretable"] = flagged(interp)
    for name, cols in {"spectral_envelope_3oct_only": sp["env3oct"], "ltas_3oct_only": sp["ltas3oct"], "mfcc_only": sp["mfcc"],
                       "core19_plus_spectral_envelope": VARS + sp["env3oct"], "interpretable_plus_mfcc": interp + sp["mfcc"],
                       "lfcc_only": sp["lfcc"], "high_band_erb_envelope_only": sp["envERB_hi"],
                       "voice_quality_A5_only": sp["voice_quality_A5"], "core19_plus_voice_quality_A5": VARS + sp["voice_quality_A5"],
                       "interpretable_minus_flagged": [c for c in interp if c not in flagged(interp)],
                       "interpretable_plus_mfcc_lfcc": interp + sp["mfcc"] + sp["lfcc"]}.items():
        if cols:
            Pp, _ = oof(design(df, cols), Y, krr)
            A[name] = {"n_vars": len(cols), "krr_r2": r2(Y, Pp), "ci95": boot_ci(Y, Pp)}
            Ppn, _ = oof(np.column_stack([design(df, cols), N]), Y, krr)
            A[name]["plus_nuisance_krr_r2"] = r2(Y, Ppn)
    Pn2, _ = oof(np.column_stack([design(df, interp), N]), Y, krr)
    A["interpretable_pool_plus_nuisance_krr_r2"] = r2(Y, Pn2)
    # demographic baselines
    D = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit_transform(df[["gender", "age"]])
    Pd, _ = oof(D, Y, lambda: RidgeCV(alphas=np.logspace(-3, 3, 13)))
    A["gender_age_only_r2"] = r2(Y, Pd)
    Pvd, _ = oof(np.column_stack([X, D]), Y, krr)
    A["core19_plus_gender_age_krr_r2"] = r2(Y, Pvd)
    g = df.gender.values
    Yc = Y.copy(); Xc = np.where(np.isnan(X), np.nanmedian(X, 0), X).copy()
    gm_share = 0.0
    for lev in np.unique(g):
        m = g == lev
        gm_share += m.sum() * ((Y[m].mean(0) - Y.mean(0)) ** 2).sum()
        Yc[m] -= Y[m].mean(0); Xc[m] -= Xc[m].mean(0)
    A["between_gender_share_of_total_ss"] = float(gm_share / ((Y - Y.mean(0)) ** 2).sum())
    Pc, _ = oof(Xc, Yc, krr)
    A["core19_within_gender_centred_krr_r2"] = r2(Yc, Pc)
    Xic = design(df, interp); Xic = np.where(np.isnan(Xic), np.nanmedian(Xic, 0), Xic)
    for lev in np.unique(g):
        Xic[g == lev] -= Xic[g == lev].mean(0)
    Pic, _ = oof(Xic, Yc, krr)
    A["interpretable_pool_within_gender_centred_krr_r2"] = r2(Yc, Pic)
    # duplicate-speaker check
    Sm = Y @ Y.T; iu = np.triu_indices(len(Y), 1)
    A["speaker_pairs_cosine_gt_0_85"] = int((Sm[iu] > 0.85).sum())
    A["split_half_note"] = "same-session, half-length bound, not a true ceiling (A2)"
    # nested forward selection curve (ridge selection inside each training fold, RBF-KRR scoring)
    XI = design(df, interp)
    curve = {}
    if os.environ.get("KRR_BACKEND") == "gpu":   # same selection (nested_gpu.py; equivalence: results/nested_gpu_equivalence.json)
        from nested_gpu import greedy_path
        ks = [k for k in (5, 10, 15, 20, 30) if k <= XI.shape[1]]; paths, folds = [], list(KFold(10, shuffle=True, random_state=SEED).split(XI))
        for tr, te in folds:
            med = np.nanmedian(XI[tr], 0)
            Xtr = np.where(np.isnan(XI[tr]), med, XI[tr]); Xte = np.where(np.isnan(XI[te]), med, XI[te])
            sc = StandardScaler().fit(Xtr); paths.append((sc.transform(Xtr), sc.transform(Xte), greedy_path(sc.transform(Xtr), Y[tr], max(ks), SEED)))
        for k in ks:
            P = np.zeros_like(Y); chosen_all = []
            for (tr, te), (Xtr, Xte, path) in zip(folds, paths):
                chosen = path[:k]; P[te] = fixed_rbf().fit(Xtr[:, chosen], Y[tr]).predict(Xte[:, chosen]); chosen_all.append([interp[c] for c in chosen])
            sel = pd.Series([v for c in chosen_all for v in c]).value_counts()
            curve[k] = {"krr_r2": r2(Y, P), "variables_selected_in_ge_8_of_10_folds": sel[sel >= 8].index.tolist()}
    for k in ([] if curve else [k for k in (5, 10, 15, 20, 30) if k <= XI.shape[1]]):
        P = np.zeros_like(Y); chosen_all = []
        for tr, te in KFold(10, shuffle=True, random_state=SEED).split(XI):
            med = np.nanmedian(XI[tr], 0)
            Xtr = np.where(np.isnan(XI[tr]), med, XI[tr]); Xte = np.where(np.isnan(XI[te]), med, XI[te])
            sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            chosen = []
            inner = list(KFold(3, shuffle=True, random_state=SEED).split(Xtr))
            while len(chosen) < k:
                best, bs = None, -np.inf
                for j in range(Xtr.shape[1]):
                    if j in chosen:
                        continue
                    cols = chosen + [j]; sc_j = 0.0
                    for itr, ite in inner:
                        mdl = RidgeCV(alphas=np.logspace(-3, 3, 7)).fit(Xtr[itr][:, cols], Y[tr][itr])
                        sc_j += r2(Y[tr][ite], mdl.predict(Xtr[ite][:, cols]))
                    if sc_j > bs:
                        best, bs = j, sc_j
                chosen.append(best)
            P[te] = fixed_rbf().fit(Xtr[:, chosen], Y[tr]).predict(Xte[:, chosen])
            chosen_all.append([interp[c] for c in chosen])
        sel = pd.Series([v for c in chosen_all for v in c]).value_counts()
        curve[k] = {"krr_r2": r2(Y, P), "variables_selected_in_ge_8_of_10_folds": sel[sel >= 8].index.tolist()}
    A["nested_selection_curve"] = curve
    out["amendments_A1_A2"] = A
    json.dump(out, open(os.path.join(RES, "h1_results.json" if SAMPLE == "primary" else f"h1_results_{SAMPLE}.json"), "w"), ensure_ascii=False, indent=1)
    np.savez_compressed(os.path.join(RES, "h1_oof_predictions.npz" if SAMPLE == "primary" else f"h1_oof_predictions_{SAMPLE}.npz"), Y=Y, P_krr=P_k, P_lin=P_l, speakers=df.index.values)
    print(json.dumps({k: v for k, v in out.items() if k not in ("variable_correlation", "krr_inner_params")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
