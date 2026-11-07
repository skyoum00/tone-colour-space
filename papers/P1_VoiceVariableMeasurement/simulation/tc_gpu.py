"""P1 GPU part (PAPER_PLAN.md 2026-11-04 14:42): A2 duration subsets (3/5/10/20/30/60 s of half 1 -> OpenVoice ref_enc on CUDA)
and E(d,e) RBF-KRR (pre-registered grid, GPU backend) from core-19 / interpretable pool to PC1-PC8 scores, GroupKFold(10) over
near-duplicate components (ECAPA cosine > 0.85, same rule and code as tc_analysis.py). Run under the GPU lease (run_tc_gpu.sh).
Audio is read from the lab NAS only; only vectors and aggregate metrics are written."""
import io, json, os, sys, time, types, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, torch, librosa
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
HERE = os.path.dirname(os.path.abspath(__file__)); SIM = os.path.join(HERE, "../../../simulation")
sys.path.insert(0, SIM); sys.path.insert(0, os.path.join(SIM, "stage1"))
os.environ["KRR_BACKEND"] = "gpu"
import common_data as C, analyze_h1 as A, measure_voice as mv
import config
from krr_gpu import GPUKRRGrid
OUT = os.path.join(HERE, "results"); DUR = [3, 5, 10, 20, 30, 60]; SEED = 0
CK = config.OPENVOICE_CKPT
df = C.load("primary", with_tone_colour=True); n = len(df)
E = np.vstack(df.ecapa.values); iu = np.triu_indices(n, 1); m = (E @ E.T)[iu] > 0.85
k_, groups = connected_components(coo_matrix((np.ones(m.sum()), (iu[0][m], iu[1][m])), shape=(n, n)), directed=False)
FOLDS = list(GroupKFold(10).split(np.zeros(n), groups=groups))
R = {"started": time.strftime("%F %T"), "n": n, "n_groups": int(k_)}

# ---------------- E(d,e) KRR to PC scores ------------------------------------------------------------------------------------
S = np.vstack(df.se.values); mu = S.mean(0); _, _, Vt = np.linalg.svd(S - mu, full_matrices=False); Z = (S - mu) @ Vt[:8].T
interp, _ = A.pools(df)
Ek = {}
for name, cols in (("core19", A.VARS), ("interpretable_pool", interp)):
    X = A.design(df, cols); r2s = []
    for j in range(8):
        P = np.zeros(n)
        for tr, te in FOLDS:
            med = np.nanmedian(X[tr], 0); Xtr = np.where(np.isnan(X[tr]), med, X[tr]); Xte = np.where(np.isnan(X[te]), med, X[te])
            sc = StandardScaler().fit(Xtr)
            mdl = GPUKRRGrid([1e-3, 1e-2, 1e-1, 1], [0.005, 0.01, 0.02, 0.05, 0.1], SEED).fit(sc.transform(Xtr), Z[tr, j:j + 1])
            P[te] = mdl.predict(sc.transform(Xte))[:, 0]
        r2s.append(float(1 - ((Z[:, j] - P) ** 2).sum() / ((Z[:, j] - Z[:, j].mean()) ** 2).sum()))
    Ek[name] = {"n_inputs": len(cols), "krr_r2_per_pc": r2s}
    print("E", name, r2s, flush=True)
R["E_krr_variables_to_pcs"] = Ek

# ---------------- A2 duration subsets -------------------------------------------------------------------------------------
sys.modules["openvoice.text"] = types.SimpleNamespace(text_to_sequence=None); sys.path.insert(0, config.OPENVOICE_DIR)
from openvoice.api import ToneColorConverter, OpenVoiceBaseClass
from openvoice.mel_processing import spectrogram_torch
cv = ToneColorConverter.__new__(ToneColorConverter); OpenVoiceBaseClass.__init__(cv, CK + "config.json", device="cuda")
cv.watermark_model = None; cv.version = getattr(cv.hps, "_version_", "v1"); cv.load_ckpt(CK + "checkpoint.pth"); h = cv.hps.data
zm = mv.zip_map()


def se(y16):
    y = torch.FloatTensor(librosa.resample(y16, orig_sr=mv.SR, target_sr=h.sampling_rate))[None].cuda()
    s = spectrogram_torch(y, h.filter_length, h.sampling_rate, h.hop_length, h.win_length, center=False)
    with torch.no_grad():
        return cv.model.ref_enc(s.transpose(1, 2)).squeeze().cpu().numpy()


def half1(row):
    import soundfile as sf
    zp, member = zm[row["file"]]
    x, _ = sf.read(io.BytesIO(zipfile.ZipFile(zp).read(member)), dtype="float32")
    segs = [s for s in json.loads(row["segments"]) if s["end"] <= len(x) / mv.SR]
    ch, _ = mv.pick_channel(x, segs)
    audio = [x[int(s["start"] * mv.SR):int(s["end"] * mv.SR), ch] for s in segs]
    return np.concatenate(audio[:len(audio) // 2])


rows = df.reset_index()[["speaker", "file", "segments"]].to_dict("records")
V = {d: np.full((n, 256), np.nan) for d in DUR + ["full_check"]}; L = np.zeros(n)
with ThreadPoolExecutor(4) as ex:
    for i, y in enumerate(ex.map(half1, rows)):
        L[i] = len(y) / mv.SR
        for d in DUR:
            if L[i] >= d:
                V[d][i] = se(y[:int(d * mv.SR)])
        if i < 50:
            V["full_check"][i] = se(y)          # reproducibility check against stored se_half1
        if i % 200 == 0:
            print("A2", i, n, time.strftime("%T"), flush=True)
S1, S2 = np.vstack(df.se_half1.values), np.vstack(df.se_half2.values)
chk = np.abs(V["full_check"][:50] - S1[:50]).max()
nrm = lambda M: M / np.linalg.norm(M, axis=1, keepdims=True)


def ident(Q, T):
    M_ = nrm(Q) @ nrm(T).T; d = np.diag(M_); rank = (M_ > d[:, None]).sum(1) + 1
    rng = np.random.default_rng(SEED); non = rng.choice(M_[~np.eye(len(M_), dtype=bool)], 200000, replace=False)
    return {"mean_cos_to_own": float(d.mean()), "eer": A.eer(np.r_[d, non], np.r_[np.ones(len(d)), np.zeros(len(non))]),
            "top1": float((rank == 1).mean()), "top10": float((rank <= 10).mean()), "median_rank": float(np.median(rank)), "n_candidates": int(len(M_))}


A2 = {"half1_seconds_quantiles": {p: float(np.percentile(L, p)) for p in (0, 1, 5, 50, 95, 100)},
      "full_half1_reextraction_max_abs_diff_first50": float(chk), "by_duration": {}}
for d in DUR:
    ok = ~np.isnan(V[d][:, 0]); M_ = nrm(V[d][ok]) @ nrm(S2).T          # candidates: all 2,408 half-2 vectors
    d_ = M_[np.arange(ok.sum()), np.where(ok)[0]]; rank = (M_ > d_[:, None]).sum(1) + 1
    lab = np.zeros_like(M_, bool); lab[np.arange(ok.sum()), np.where(ok)[0]] = True
    rng = np.random.default_rng(SEED); non = rng.choice(M_[~lab], 200000, replace=False)
    A2["by_duration"][str(d)] = {"n_speakers": int(ok.sum()), "mean_cos_to_own": float(d_.mean()),
                                 "eer": A.eer(np.r_[d_, non], np.r_[np.ones(len(d_)), np.zeros(len(non))]),
                                 "top1": float((rank == 1).mean()), "top10": float((rank <= 10).mean()), "median_rank": float(np.median(rank)), "n_candidates": n}
    print("A2", d, A2["by_duration"][str(d)], flush=True)
A2["full_half1_reference"] = ident(S1, S2)
st = [d for d in DUR if A2["by_duration"][str(d)]["eer"] <= 0.02 and A2["by_duration"][str(d)]["top10"] >= 0.90]
A2["stable_duration_rule"] = "shortest duration with EER <= 0.02 and top-10 >= 0.90"; A2["stable_duration_s"] = st[0] if st else None
R["A2_duration"] = A2; R["finished"] = time.strftime("%F %T")
json.dump(R, open(os.path.join(OUT, "tc_gpu.json"), "w"), indent=1, ensure_ascii=False, default=float)
np.savez_compressed(os.path.join(OUT, "tc_duration_vectors.npz"), speakers=df.index.values, half1_seconds=L, **{f"d{d}": V[d] for d in DUR})
print("done", flush=True)
