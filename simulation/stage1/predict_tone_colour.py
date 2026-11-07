"""E-VC V3 step 1 (PREREG A7): predict the OpenVoice tone-colour vector (256-d) from measured voice variables.

Same folds, models and pools as Gate 1 (analyze_h1.py). Two protocols:
  whole  - variables of the whole speech -> vector of the whole speech (OOF variance-weighted R2)
  convert - fit on training-fold speakers (whole -> whole); predict a test speaker's vector from its HALF-1 variables and
            score against its HALF-2 vector (cosine, verification EER, identification rank among all half-2 vectors).
References in the convert protocol: own half-1 vector (audio-based upper), training-fold gender mean and global mean (lower).
Outputs: results/tone_colour_prediction[_all].json, paper_figures/tone_colour_prediction_from_variables[_all].png/.json.
DRY=1 runs the whole pipeline on random arrays (code check only)."""
import json, os
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import analyze_h1 as A

DRY = os.environ.get("DRY") == "1"
SUF = "" if A.SAMPLE == "primary" else "_" + A.SAMPLE
if DRY:
    SUF += "_DRYRUN"


def load():
    idx = pd.read_parquet(os.path.join(A.RES, "speaker_index.parquet")).set_index("speaker")
    rows = [json.loads(l) for l in open(os.path.join(A.RES, "measurements.jsonl"))]
    rp = os.path.join(A.RES, "measurements_retry.jsonl")
    if os.path.exists(rp):
        retry = {r["speaker"]: r for r in map(json.loads, open(rp))}
        rows = [retry.get(r["speaker"], r) if "error" in r else r for r in rows]
    ok = [r for r in rows if "error" not in r]
    tc = {r["speaker"]: r for r in map(json.loads, open(os.path.join(A.RES, "tone_colour.jsonl"))) if "error" not in r}
    ok = [r for r in ok if r["speaker"] in tc]
    df = pd.DataFrame(ok).set_index("speaker").join(idx[["gender", "age", "device", "snr_db", "split"]])
    if A.SAMPLE == "primary":
        chk = pd.read_csv(os.path.join(A.RES, "audio_label_duration_check.csv")).set_index("speaker")
        df = df[df.index.isin(chk.index[(chk.wav_over_label >= 0.99) & (chk.wav_over_label <= 1.01)])]
    h1 = pd.DataFrame(list(df.half1.values), index=df.index)
    Y = np.array([tc[s]["se"] for s in df.index]); Y1 = np.array([tc[s]["se_half1"] for s in df.index])
    Y2 = np.array([tc[s]["se_half2"] for s in df.index])
    return df, h1, Y, Y1, Y2


def fake(n=300):
    rng = np.random.default_rng(1); cols = A.VARS + [f"env3oct_{i}" for i in range(5)] + [f"mfcc{i}_mean" for i in range(4)]
    base = rng.normal(size=(n, len(cols))); df = pd.DataFrame(np.abs(base) + 1, columns=cols, index=[f"s{i}" for i in range(n)])
    h1 = df + rng.normal(scale=0.3, size=df.shape); W = rng.normal(size=(len(cols), 256))
    df["gender"] = rng.choice(["여성", "남성"], n); df["age"] = rng.choice(["20대", "40대"], n)
    df["ecapa"] = list(rng.normal(size=(n, 192))); df["ecapa_half1"] = list(rng.normal(size=(n, 192)))
    Y = base @ W * 0.1 + rng.normal(size=(n, 256)); return df, h1, Y, Y + rng.normal(scale=0.5, size=Y.shape), Y + rng.normal(scale=0.5, size=Y.shape)


def unit(M):
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def convert_scores(P, Y2):
    S = unit(P) @ unit(Y2).T
    rank = (S > np.diag(S)[:, None]).sum(1) + 1
    return {"mean_cos_to_own_half2": float(np.diag(S).mean()), "eer": A.verification(P, Y2),
            "top1": float((rank == 1).mean()), "top10": float((rank <= 10).mean()), "median_rank": float(np.median(rank)),
            "n_candidates": int(len(Y2))}


def fit_predict(X, Xh, Y, model_fn):
    """OOF predictions for the whole protocol (X[te]) and the convert protocol (Xh[te]) from one fit per fold."""
    Pw, Ph = np.zeros_like(Y), np.zeros_like(Y); params = []
    for tr, te in KFold(10, shuffle=True, random_state=A.SEED).split(X):
        med = np.nanmedian(X[tr], 0); fill = lambda M: np.where(np.isnan(M), med, M)
        sc = StandardScaler().fit(fill(X[tr])); m = model_fn().fit(sc.transform(fill(X[tr])), Y[tr])
        Pw[te] = m.predict(sc.transform(fill(X[te]))); Ph[te] = m.predict(sc.transform(fill(Xh[te])))
        params.append(getattr(m, "best_params_", None))
    return Pw, Ph, params


def main():
    df, h1, Y, Y1, Y2 = fake() if DRY else load()
    common = [c for c in df.columns if c in h1.columns]
    interp = [c for c in A.pools(df[common + ["gender"]])[0] if c in common]
    sp = A.spectral_pools(df[common])
    pools = {"core19": A.VARS, "interpretable": interp, "interpretable_minus_flagged": [c for c in interp if c not in A.flagged(interp)],
             "interpretable+mfcc+lfcc": interp + sp["mfcc"] + sp["lfcc"]}
    out = {"sample": A.SAMPLE, "dry_run": DRY, "n": len(df), "target": "OpenVoice v2 ref_enc tone-colour, 256-d", "pools": {}}
    ridge = lambda: RidgeCV(alphas=np.logspace(-3, 3, 13))
    for name, cols in pools.items():
        X, Xh = A.design(df, cols), A.design(h1, cols)
        r = {"n_inputs": len(cols)}
        for mname, mf in (("rbf_krr", A.krr), ("ridge", ridge)):
            Pw, Ph, params = fit_predict(X, Xh, Y, mf)
            r[mname] = {"r2_whole": A.r2(Y, Pw), "r2_whole_ci95": A.boot_ci(Y, Pw), "convert": convert_scores(Ph, Y2)}
            if params[0]: r[mname]["params"] = params
        out["pools"][name] = r; print(name, json.dumps({k: v for k, v in r["rbf_krr"].items() if k != "params"}), flush=True)
    D = pd.get_dummies(df[["gender", "age"]].astype(str)).values.astype(float)
    Pw, Ph, _ = fit_predict(D, D, Y, ridge)
    out["pools"]["demographic_gender_age"] = {"ridge": {"r2_whole": A.r2(Y, Pw), "convert": convert_scores(Ph, Y2)}}
    E, Eh = np.vstack(df.ecapa.values), np.vstack(df.ecapa_half1.values)
    Pw, Ph, _ = fit_predict(E, Eh, Y, A.krr)
    out["pools"]["ecapa_reference"] = {"rbf_krr": {"r2_whole": A.r2(Y, Pw), "convert": convert_scores(Ph, Y2)}}
    # references
    out["upper_own_half1_vector"] = convert_scores(Y1, Y2)
    G = np.zeros_like(Y); M0 = np.zeros_like(Y); g = df.gender.values
    for tr, te in KFold(10, shuffle=True, random_state=A.SEED).split(Y):
        for gg in np.unique(g):
            G[te[g[te] == gg]] = Y[tr][g[tr] == gg].mean(0)
        M0[te] = Y[tr].mean(0)
    out["lower_gender_mean_vector"] = convert_scores(G, Y2)
    out["lower_global_mean_vector"] = {"mean_cos_to_own_half2": float((unit(M0) * unit(Y2)).sum(1).mean())}
    json.dump(out, open(os.path.join(A.RES, f"tone_colour_prediction{SUF}.json"), "w"), indent=1, ensure_ascii=False)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    names = list(pools) + ["demographic_gender_age", "ecapa_reference"]
    rows = [(n, out["pools"][n].get("rbf_krr", out["pools"][n].get("ridge"))["convert"]) for n in names]
    rows += [("upper: own half-1 vector", out["upper_own_half1_vector"]), ("lower: gender mean", out["lower_gender_mean_vector"])]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
    for a, k, t in zip(ax, ["mean_cos_to_own_half2", "top10", "eer"], ["cosine to own half-2 vector", f"top-10 identification (of {len(Y2)})", "verification EER"]):
        a.barh([r[0] for r in rows], [r[1].get(k, np.nan) for r in rows], color=["#4a6fa5"] * len(names) + ["#5a9e6f", "#b0603a"]); a.set_title(t); a.invert_yaxis()
    for a in ax[1:]: a.set_yticklabels([])
    fig.suptitle(f"Tone-colour vector predicted from half-1 variables, scored on half-2 (n={len(Y)}, sample={A.SAMPLE}{', DRY RUN' if DRY else ''})")
    fig.tight_layout(); os.makedirs(A.FIG, exist_ok=True)
    fp = os.path.join(A.FIG, f"tone_colour_prediction_from_variables{SUF}")
    fig.savefig(fp + ".png", dpi=200); json.dump({"bars": {r[0]: r[1] for r in rows}}, open(fp + ".json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
