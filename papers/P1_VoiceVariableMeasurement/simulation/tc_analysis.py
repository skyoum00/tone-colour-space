"""P1 (tone-colour space) CPU analyses, pre-registered in PAPER_PLAN.md (2026-11-04 14:42): A1, B1-B4, C1, D0, E(a-c),
and the target vectors for C2 / D1-D3 / E demos. Output: results/tc_analysis.json, results/tc_targets.npz.
All random choices seed 0. GroupKFold(10) over near-duplicate components (ECAPA cosine > 0.85 graph, fallback rule)."""
import json, os, sys, time
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
HERE = os.path.dirname(os.path.abspath(__file__)); SIM = os.path.join(HERE, "../../../simulation")
sys.path.insert(0, SIM); sys.path.insert(0, os.path.join(SIM, "stage1"))
import common_data as C, analyze_h1 as A
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
SEED = 0; KS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
R = {"plan": "PAPER_PLAN.md 2026-11-04 14:42", "started": time.strftime("%F %T")}

df = C.load("primary", with_tone_colour=True)
S, S1, S2 = (np.vstack(df[k].values) for k in ("se", "se_half1", "se_half2"))
E, E1, E2 = (np.vstack(df[k].values) for k in ("ecapa", "ecapa_half1", "ecapa_half2"))
g, age, dev, snr, fil = (df[k].values for k in ("gender", "age", "device", "snr_db", "file"))
n = len(df); R["n"] = n; R["gender_counts"] = pd.Series(g).value_counts().to_dict()
AGE_ORD = {"20대": 0, "30대": 1, "40대": 2, "50대": 3, "60대 이상": 4}
nrm = lambda M: M / np.linalg.norm(M, axis=1, keepdims=True)


def r2(Y, P):
    return A.r2(Y, P)


def eer_thr(scores, labels):
    fpr, tpr, thr = roc_curve(labels, scores); i = np.argmin(np.abs(fpr - (1 - tpr)))
    return float((fpr[i] + 1 - tpr[i]) / 2), float(thr[i])


def ident(Q, T, centre=None):
    if centre is not None:
        Q, T = Q - centre, T - centre
    M = nrm(Q) @ nrm(T).T; d = np.diag(M); rank = (M > d[:, None]).sum(1) + 1
    rng = np.random.default_rng(SEED); non = rng.choice(M[~np.eye(len(M), dtype=bool)], 200000, replace=False)
    e, _ = eer_thr(np.r_[d, non], np.r_[np.ones(len(d)), np.zeros(len(non))])
    return {"mean_cos_to_own": float(d.mean()), "eer": e, "top1": float((rank == 1).mean()), "top10": float((rank <= 10).mean()),
            "median_rank": float(np.median(rank)), "n_candidates": int(len(M))}


q = lambda v: {p: float(np.percentile(v, p)) for p in (1, 5, 25, 50, 75, 95, 99)}

# ---------------- near-duplicate components and groups ------------------------------------------------------------------
iu = np.triu_indices(n, 1); Ec = (E @ E.T)[iu]; Tc = (nrm(S) @ nrm(S).T)[iu]


def comps(mask):
    I, J = iu[0][mask], iu[1][mask]
    G = coo_matrix((np.ones(len(I)), (I, J)), shape=(n, n))
    k, lab = connected_components(G, directed=False); sz = np.bincount(lab)
    return lab, {"n_edges": int(mask.sum()), "n_components": int(k), "n_singletons": int((sz == 1).sum()),
                 "largest_component": int(sz.max()), "n_components_size_ge_2": int((sz >= 2).sum()),
                 "size_quantiles_nonsingleton": q(sz[sz >= 2]) if (sz >= 2).any() else {}}


struct = {}; groups = None; used_thr = None
for t in (0.85, 0.90, 0.95):
    lab, info = comps(Ec > t); struct[str(t)] = info
    if groups is None and info["largest_component"] <= 0.2 * n:
        groups, used_thr = lab, t
if groups is None:
    groups, used_thr = np.arange(n), "none (no threshold met the 20% rule; speaker keys as groups)"
R["near_duplicate_components_ecapa"] = struct; R["grouping_threshold_used"] = used_thr
FOLDS = list(GroupKFold(10).split(S, groups=groups))
FILE_FOLDS = list(GroupKFold(10).split(S, groups=pd.factorize(fil)[0]))
print("groups", used_thr, struct, flush=True)

# B4 metadata agreement for edge pairs
frac = float((Ec > 0.85).mean()); t_tc = float(np.quantile(Tc, 1 - frac))


def meta(mask):
    I, J = iu[0][mask], iu[1][mask]
    f = lambda a: float((a[I] == a[J]).mean()) if len(I) else float("nan")
    return {"n_pairs": int(len(I)), "same_gender": f(g), "same_age_band": f(age), "same_device": f(dev), "same_conversation_file": f(fil),
            "median_abs_snr_label_diff": float(np.median(np.abs(snr[I] - snr[J]))) if len(I) else float("nan")}


allp = np.ones(len(iu[0]), bool)
mE, mT = Ec > 0.85, Tc > t_tc
R["B4_near_duplicates"] = {"ecapa_gt_0_85": meta(mE), "tone_colour_quantile_matched": {"threshold": t_tc, **meta(mT)},
                           "all_pairs_base_rate": meta(allp),
                           "edge_jaccard_ecapa_vs_tone_colour": float((mE & mT).sum() / max(1, (mE | mT).sum())),
                           "tone_colour_components": comps(mT)[1]}
print("B4", R["B4_near_duplicates"], flush=True)

# ---------------- A1 split-half reliability -----------------------------------------------------------------------------
cen = np.vstack([S1, S2]).mean(0)
X12 = nrm(S1) @ nrm(S2).T
R["A1_split_half"] = {"raw": ident(S1, S2), "centred": ident(S1, S2, cen),
                      "same_speaker_cos_quantiles": q(np.diag(X12)), "different_speaker_cos_quantiles": q(X12[~np.eye(n, dtype=bool)]),
                      "same_speaker_centred_cos_quantiles": q(np.diag(nrm(S1 - cen) @ nrm(S2 - cen).T))}
print("A1", R["A1_split_half"]["raw"], R["A1_split_half"]["centred"], flush=True)

# ---------------- B1 PCA spectrum ---------------------------------------------------------------------------------------
mu = S.mean(0); U, sv, Vt = np.linalg.svd(S - mu, full_matrices=False); lam = sv ** 2 / (n - 1); cum = np.cumsum(lam) / lam.sum()
Z1, Z2 = (S1 - mu) @ Vt.T, (S2 - mu) @ Vt.T
R["B1_pca"] = {"eigenvalues": lam.tolist(), "cumulative": cum.tolist(), "participation_ratio": float(lam.sum() ** 2 / (lam ** 2).sum()),
               "n_for": {str(p): int(np.searchsorted(cum, p) + 1) for p in (0.5, 0.8, 0.9, 0.95, 0.99)},
               "pc_split_half_r_first32": [float(np.corrcoef(Z1[:, k], Z2[:, k])[0, 1]) for k in range(32)],
               "total_variance": float(lam.sum())}
print("B1 PR", R["B1_pca"]["participation_ratio"], R["B1_pca"]["n_for"], flush=True)


def oof_ridge(X, Y, folds=FOLDS):
    P = np.zeros_like(Y, dtype=float)
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr]); m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(sc.transform(X[tr]), Y[tr]); P[te] = m.predict(sc.transform(X[te]))
    return P


oh = lambda *cols: OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit_transform(np.column_stack(cols))
Dg, Da, Dga, Dcell = oh(g), oh(age), oh(g, age), oh(np.char.add(g.astype(str), age.astype(str)))
Ddev = np.column_stack([oh(dev), snr])


def ss_share(lab):
    return float(sum((lab == l).sum() * ((S[lab == l].mean(0) - mu) ** 2).sum() for l in np.unique(lab)) / ((S - mu) ** 2).sum())


# ---------------- B2 / B3 -----------------------------------------------------------------------------------------------
B2 = {"between_gender_ss_share": ss_share(g), "between_age_ss_share": ss_share(age), "between_gender_x_age_cell_ss_share": ss_share(np.char.add(g.astype(str), age.astype(str)))}
for name, X in (("gender", Dg), ("age_band", Da), ("gender_plus_age", Dga), ("gender_x_age_cells", Dcell)):
    B2[name + "_ridge_r2"] = r2(S, oof_ridge(X, S))
B2["gender_plus_age_ridge_r2_file_grouped"] = r2(S, oof_ridge(Dga, S, FILE_FOLDS))
# gender direction AUC (OOF)
score = np.zeros(n)
for tr, te in FOLDS:
    d = S[tr][g[tr] == "여성"].mean(0) - S[tr][g[tr] == "남성"].mean(0); score[te] = S[te] @ d / np.linalg.norm(d)
B2["gender_direction_auc_female"] = float(roc_auc_score(g == "여성", score))
# age direction within gender (OOF ridge on ordinal age)
ag = np.array([AGE_ORD[a] for a in age]); B2["age_direction_spearman_within_gender"] = {}
for lev in ("여성", "남성"):
    m = np.where(g == lev)[0]; pred = np.zeros(len(m))
    for tr, te in GroupKFold(10).split(m, groups=groups[m]):
        sc = StandardScaler().fit(S[m[tr]]); mdl = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(sc.transform(S[m[tr]]), ag[m[tr]]); pred[te] = mdl.predict(sc.transform(S[m[te]]))
    rho = spearmanr(pred, ag[m]); B2["age_direction_spearman_within_gender"][{"여성": "female", "남성": "male"}[lev]] = {"rho": float(rho.correlation), "p": float(rho.pvalue), "n": int(len(m))}
R["B2_gender_age"] = B2
B3 = {"device_snr_labels_ridge_r2": r2(S, oof_ridge(Ddev, S)), "gender_age_device_snr_ridge_r2": r2(S, oof_ridge(np.column_stack([Dga, Ddev]), S))}
B3["increment_over_gender_age"] = B3["gender_age_device_snr_ridge_r2"] - B2["gender_plus_age_ridge_r2"]
B3["device_counts"] = pd.Series(dev).value_counts().to_dict()
R["B3_device_snr_label_effects"] = B3
print("B2", B2, "B3", B3, flush=True)

# ---------------- C1 SVD knobs ------------------------------------------------------------------------------------------
C1 = {}
for k in KS:
    Q = np.zeros_like(S1); Qc = np.zeros_like(S1); num = den = 0.0
    for tr, te in FOLDS:
        m_ = S[tr].mean(0); _, _, V = np.linalg.svd(S[tr] - m_, full_matrices=False); Vk = V[:k]
        Q[te] = m_ + (S1[te] - m_) @ Vk.T @ Vk
        Xt = S[te] - S[te].mean(0); num += ((Xt @ Vk.T) ** 2).sum(); den += (Xt ** 2).sum()
    C1[str(k)] = {"raw": ident(Q, S2), "centred": ident(Q, S2, cen), "between_speaker_variance_retained": float(num / den)}
    print("C1", k, C1[str(k)]["raw"], C1[str(k)]["between_speaker_variance_retained"], flush=True)
ok = [k for k in KS if C1[str(k)]["raw"]["eer"] <= 0.02 and C1[str(k)]["raw"]["top10"] >= 0.90]
R["C1_svd_knobs"] = {"by_k": C1, "knob_count_rule": "smallest k with EER <= 0.02 and top-10 >= 0.90 (raw cosine)",
                     "knob_count": ok[0] if ok else None, "full_vector_reference": {"eer": 0.0136, "top1": 0.715}}

# ---------------- D0 ECAPA threshold (primary 2,410, half1 vs half2, same trials as the 19-variable analysis) --------------
dp = C.load("primary"); P1e, P2e = nrm(np.vstack(dp.ecapa_half1.values)), nrm(np.vstack(dp.ecapa_half2.values))
M = P1e @ P2e.T; lab = np.eye(len(M), dtype=bool); rng = np.random.default_rng(SEED); non = rng.choice(M[~lab], 200000, replace=False)
e_, th = eer_thr(np.r_[np.diag(M), non], np.r_[np.ones(len(M)), np.zeros(len(non))])
R["D0_threshold"] = {"n_real": int(len(M)), "eer": e_, "theta_eer": th, "theta1_same_speaker_p1": float(np.percentile(np.diag(M), 1)),
                     "real_same_speaker_cos_quantiles": q(np.diag(M)), "real_max_cos_to_other_speakers_quantiles": q(np.where(lab, -1, M).max(1))}
print("D0", R["D0_threshold"]["eer"], th, flush=True)

# ---------------- E (a-c) top PCs vs labels; Spearman with core-19 ------------------------------------------------------
Zs = (S - mu) @ Vt[:8].T
E_ = {"pc_sd": Zs.std(0).tolist(), "variance_share": (lam[:8] / lam.sum()).tolist()}
for name, X in (("gender", Dg), ("age_band", Da), ("gender_plus_age", Dga), ("device_snr_labels", Ddev)):
    P = oof_ridge(X, Zs); E_[name + "_r2_per_pc"] = [float(1 - ((Zs[:, j] - P[:, j]) ** 2).sum() / ((Zs[:, j] - Zs[:, j].mean()) ** 2).sum()) for j in range(8)]
Xv = df[A.VARS].astype(float); Xv["f0_mean_hz"] = np.log(Xv["f0_mean_hz"])
E_["spearman_core19"] = {v: [float(spearmanr(Xv[v], Zs[:, j], nan_policy="omit").correlation) for j in range(8)] for v in A.VARS}
E_["spearman_core19_within_female"] = {v: [float(spearmanr(Xv[v][g == "여성"], Zs[g == "여성", j], nan_policy="omit").correlation) for j in range(8)] for v in A.VARS}
R["E_pc_interpretation_labels"] = E_
print("E", {k: v for k, v in E_.items() if "r2" in k}, flush=True)

# ---------------- target vectors for conversions (C2, D1-D3, E demos) -------------------------------------------------------
rng = np.random.default_rng(SEED)
tg_f = rng.choice(np.where(g == "여성")[0], 30, replace=False); tg_m = rng.choice(np.where(g == "남성")[0], 30, replace=False)
tgt = np.r_[tg_f, tg_m]; rest = np.setdiff1d(np.arange(n), tgt)
m_ = S[rest].mean(0); _, _, V = np.linalg.svd(S[rest] - m_, full_matrices=False)
C2 = {str(k): m_ + (S1[tgt] - m_) @ V[:k].T @ V[:k] for k in (4, 16, 64, 256)}
cells = [(gg, aa) for gg in ("여성", "남성") for aa in AGE_ORD]
D1 = np.vstack([S[(g == gg) & (age == aa)].mean(0) for gg, aa in cells]); D1n = [int(((g == gg) & (age == aa)).sum()) for gg, aa in cells]
fe, ma = np.where(g == "여성")[0], np.where(g == "남성")[0]
pairs = [tuple(rng.choice(fe, 2, replace=False)) for _ in range(8)] + [tuple(rng.choice(ma, 2, replace=False)) for _ in range(7)] + \
        [(rng.choice(fe), rng.choice(ma)) for _ in range(15)]
D2 = np.vstack([(1 - a) * S[i] + a * S[j] for i, j in pairs for a in (0.25, 0.5, 0.75)])
D3, D3g = [], []
for gg in ("여성", "남성"):
    X = S[g == gg]; D3.append(rng.multivariate_normal(X.mean(0), np.cov(X.T), 30)); D3g += [gg] * 30
D3 = np.vstack(D3)
np.savez_compressed(os.path.join(OUT, "tc_targets.npz"), C2_targets=tgt, C2_speakers=df.index.values[tgt], C2_gender=g[tgt],
                    **{f"C2_k{k}": v for k, v in C2.items()}, D1=D1, D1_cells=np.array([f"{a}|{b}" for a, b in cells]), D1_n=np.array(D1n),
                    D2=D2, D2_pairs=np.array(pairs), D2_pair_speakers=df.index.values[np.array(pairs)], D3=D3, D3_gender=np.array(D3g),
                    pca_mean=mu, pca_components=Vt[:8], pca_sd=Zs.std(0), speakers=df.index.values)
# vector-space novelty of generated vectors: max raw cosine to the 2,408 real whole vectors (D)
Sn = nrm(S)
vn = lambda Gv: (nrm(Gv) @ Sn.T).max(1)
R["D_vector_space_novelty"] = {"D1_max_cos": q(vn(D1)), "D2_max_cos": q(vn(D2)), "D3_max_cos": q(vn(D3)),
                               "real_same_speaker_half_cos": q(np.diag(X12)),
                               "real_max_cos_to_other_speaker_whole": q(np.where(np.eye(n, dtype=bool), -1, Sn @ Sn.T).max(1))}
R["targets"] = {"C2_n_targets": int(len(tgt)), "D1_cells": [f"{a}|{b}" for a, b in cells], "D1_cell_n": D1n, "D2_n_pairs": len(pairs), "D3_n": int(len(D3))}
R["finished"] = time.strftime("%F %T")
json.dump(R, open(os.path.join(OUT, "tc_analysis.json"), "w"), indent=1, ensure_ascii=False, default=float)
print("done", flush=True)
