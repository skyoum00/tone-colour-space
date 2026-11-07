"""P1 pre-declared non-gate diagnostics S1-S4 (PAPER_PLAN.md, declared 2026-11-04 13:03 just before running).
Primary sample (A6), same variable definitions and helpers as the pre-registered analysis (analyze_h1), KRR_BACKEND=gpu.
Output: results/p1_extra_analyses.json (this paper's simulation directory)."""
import json, os, sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
HERE = os.path.dirname(os.path.abspath(__file__)); SIM = os.path.join(HERE, "../../../simulation")
sys.path.insert(0, SIM); sys.path.insert(0, os.path.join(SIM, "stage1"))
os.environ.setdefault("KRR_BACKEND", "gpu")
import common_data as C, analyze_h1 as A
from krr_gpu import GPUKRRGrid
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
NPZ = np.load(os.path.join(C.RES, "h1_oof_predictions.npz"), allow_pickle=True)
df = C.load("primary"); df = df.loc[NPZ["speakers"]]
Y = np.vstack(df.ecapa.values); assert np.allclose(Y, NPZ["Y"]), "speaker order mismatch"
X = A.design(df); H1E, H2E = np.vstack(df.ecapa_half1.values), np.vstack(df.ecapa_half2.values)
Xh1 = A.design(pd.DataFrame(list(df.half1.values)))
res = {"n": len(df), "declared": "PAPER_PLAN.md 2026-11-04 13:03", "started": time.strftime("%F %T")}


def oof_split(X, Y, splits, model_fn):
    P = np.zeros_like(Y); params = []
    for tr, te in splits:
        med = np.nanmedian(X[tr], 0); Xtr = np.where(np.isnan(X[tr]), med, X[tr]); Xte = np.where(np.isnan(X[te]), med, X[te])
        sc = StandardScaler().fit(Xtr); m = model_fn().fit(sc.transform(Xtr), Y[tr]); P[te] = m.predict(sc.transform(Xte))
        params.append(getattr(m, "best_params_", None))
    return P, params


outer = list(KFold(10, shuffle=True, random_state=A.SEED).split(X))
# ---- S1 extended grid --------------------------------------------------------------------------------
AL, GA = [1e-3, 1e-2, 1e-1, 1, 3, 10, 30], [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
P1, pr = oof_split(X, Y, outer, lambda: GPUKRRGrid(AL, GA, A.SEED))
res["S1_extended_grid"] = {"alphas": AL, "gammas": GA, "core19_krr_r2": A.r2(Y, P1), "ci95": A.boot_ci(Y, P1), "params": pr,
                           "preregistered_grid_r2": A.r2(Y, NPZ["P_krr"])}
print("S1", res["S1_extended_grid"]["core19_krr_r2"], flush=True)

# ---- S2 high-cosine pairs ------------------------------------------------------------------------------
S = Y @ Y.T; iu = np.triu_indices(len(Y), 1); s = S[iu]
Sx = (H1E / np.linalg.norm(H1E, axis=1, keepdims=True)) @ (H2E / np.linalg.norm(H2E, axis=1, keepdims=True)).T
same = np.diag(Sx); off = Sx[~np.eye(len(Y), dtype=bool)]
hi = s > 0.85; I, J = iu[0][hi], iu[1][hi]
f, dv, g = df.file.values, df.device.values, df.gender.values
q = lambda v: {p: float(np.percentile(v, p)) for p in (1, 5, 25, 50, 75, 95, 99)}
deg = np.bincount(np.r_[I, J], minlength=len(Y))
res["S2_high_cosine_pairs"] = {
    "n_pairs_total": int(len(s)), "n_pairs_cos_gt_0_85": int(hi.sum()), "fraction": float(hi.mean()),
    "between_speaker_cos_whole_quantiles": q(s), "between_speaker_cos_mean": float(s.mean()),
    "same_speaker_half1_half2_cos_quantiles": q(same), "different_speaker_half1_half2_cos_quantiles": q(off),
    "n_cross_half_different_speaker_cos_gt_0_85": int((off > 0.85).sum()),
    "n_same_speaker_half_cos_below_0_85": int((same < 0.85).sum()),
    "high_pairs_same_conversation_file": int((f[I] == f[J]).sum()), "high_pairs_same_device": int((dv[I] == dv[J]).sum()),
    "high_pairs_same_gender": int((g[I] == g[J]).sum()),
    "base_rate_same_device": float(np.mean([(dv == d).mean() ** 2 for d in np.unique(dv)]) * len(np.unique(dv))),
    "speakers_in_any_high_pair": int((deg > 0).sum()), "max_high_pairs_per_speaker": int(deg.max()),
    "median_high_pairs_per_involved_speaker": float(np.median(deg[deg > 0])) if (deg > 0).any() else 0.0,
    "high_pairs_by_gender_pair": pd.Series([" / ".join(sorted((g[i], g[j]))) for i, j in zip(I, J)]).value_counts().to_dict()}
print("S2", {k: v for k, v in res["S2_high_cosine_pairs"].items() if "quantiles" not in k}, flush=True)

# ---- S3 grouped CV by conversation file ------------------------------------------------------------------
groups = pd.factorize(df.file.values)[0]
gsplit = list(GroupKFold(10).split(X, groups=groups))
krr = lambda: GPUKRRGrid([1e-3, 1e-2, 1e-1, 1], [0.005, 0.01, 0.02, 0.05, 0.1], A.SEED)
P3, _ = oof_split(X, Y, gsplit, krr); P3h, _ = oof_split(Xh1, H2E, gsplit, krr)
res["S3_grouped_by_conversation"] = {"n_files": int(groups.max() + 1), "speakers_sharing_a_file": int(pd.Series(groups).duplicated(keep=False).sum()),
                                      "core19_krr_r2": A.r2(Y, P3), "ci95": A.boot_ci(Y, P3), "cross_half_krr_r2": A.r2(H2E, P3h)}
print("S3", res["S3_grouped_by_conversation"], flush=True)

# ---- S4 identification from predicted embeddings (cross-half) -------------------------------------------
Ph, _ = oof_split(Xh1, H2E, outer, krr)
gm = np.zeros_like(H2E)
for tr, te in outer:
    for lev in np.unique(g):
        m_tr = tr[g[tr] == lev]; m_te = te[g[te] == lev]; gm[m_te] = H2E[m_tr].mean(0)


def ident(Pred, Tgt):
    Pn = Pred / np.linalg.norm(Pred, axis=1, keepdims=True); Tn = Tgt / np.linalg.norm(Tgt, axis=1, keepdims=True)
    M = Pn @ Tn.T; d = np.diag(M); rank = (M > d[:, None]).sum(1) + 1
    lab = np.eye(len(M), dtype=bool); rng = np.random.default_rng(A.SEED); non = rng.choice(M[~lab], 200000, replace=False)
    return {"mean_cos_to_own": float(d.mean()), "eer": A.eer(np.r_[d, non], np.r_[np.ones(len(d)), np.zeros(len(non))]),
            "top1": float((rank == 1).mean()), "top10": float((rank <= 10).mean()), "median_rank": float(np.median(rank)), "n_candidates": int(len(M))}


res["S4_identification"] = {"cross_half_r2": A.r2(H2E, Ph), "predicted_from_half1_variables": ident(Ph, H2E),
                            "lower_training_fold_gender_mean": ident(gm, H2E), "upper_own_half1_embedding": ident(H1E, H2E)}
print("S4", res["S4_identification"], flush=True)
res["finished"] = time.strftime("%F %T")
json.dump(res, open(os.path.join(OUT, "p1_extra_analyses.json"), "w"), indent=1, ensure_ascii=False, default=float)
np.savez_compressed(os.path.join(OUT, "p1_extra_predictions.npz"), P_extended=P1, P_grouped=P3, P_crosshalf=Ph, speakers=df.index.values)
