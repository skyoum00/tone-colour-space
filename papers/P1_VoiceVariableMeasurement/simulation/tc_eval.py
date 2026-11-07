"""P1 evaluation of the conversions (PAPER_PLAN.md 2026-11-04 14:42, C2, D, E demos) from eval_runner outputs.
Novelty = max ECAPA cosine of each output to the whole-speech embeddings of all 2,410 real speakers (primary sample);
thresholds theta (EER threshold of real half-1/half-2 trials) and theta1 (1st percentile of real same-speaker cosines) from
results/tc_analysis.json (D0). Output: results/tc_eval.json + figures (png/json)."""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); SIM = os.path.join(HERE, "../../../simulation"); sys.path.insert(0, SIM)
import common_data as C
RES = os.path.join(HERE, "results"); FIG = os.path.join(HERE, "figures"); EV = os.path.join(RES, "eval", "p1tc")
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
T = json.load(open(os.path.join(RES, "tc_analysis.json"))); th, th1 = T["D0_threshold"]["theta_eer"], T["D0_threshold"]["theta1_same_speaker_p1"]
rows = json.load(open(os.path.join(HERE, "outputs/manifest.json")))
ev = {d["file"]: d for d in map(json.loads, open(EV + "_gpu.jsonl")) if "error" not in d}
rm = {d["file"]: d for d in map(json.loads, open(EV + "_remeasure.jsonl")) if "error" not in d} if os.path.exists(EV + "_remeasure.jsonl") else {}
df = C.load("primary"); spk = df.index.values; nrm = lambda M: M / np.linalg.norm(M, axis=1, keepdims=True)
Ew, Eh2 = nrm(np.vstack(df.ecapa.values)), nrm(np.vstack(df.ecapa_half2.values)); pos = {s: i for i, s in enumerate(spk)}
recs = []
for r in rows:
    e = ev.get(r["file"])
    if e is None:
        continue
    x = np.load(os.path.join(EV + "_emb", e["emb_key"] + "_ecapa.npy")); x = x / np.linalg.norm(x)
    cw = Ew @ x; d = {**r, **{k: e[k] for k in ("p_female_meter", "p_female_audeering", "p_male_audeering", "utmos") if k in e}, "cer": e.get("cer"),
                    "max_cos_real": float(cw.max()), "nearest_real": str(spk[cw.argmax()]), "n_real_ge_theta": int((cw >= th).sum())}
    if r.get("target_speaker"):
        ch = Eh2 @ x; i = pos[r["target_speaker"]]; d["cos_target_half2"] = float(ch[i]); d["rank_target_half2"] = int((ch > ch[i]).sum() + 1)
        d["nearest_is_target"] = d["nearest_real"] == r["target_speaker"]
    if r.get("endpoints"):
        d["cos_endpoints"] = [float(cw[pos[s]]) for s in r["endpoints"]]
    if r["file"] in rm:
        d["rm"] = {k: rm[r["file"]][k] for k in ("f0_median_hz", "f0_mean_hz", "F1_hz", "F2_hz", "F3_hz", "F4_hz", "hnr_db", "h1_h2_db", "vtl_cm", "spectral_centroid_hz") if k in rm[r["file"]]}
    recs.append(d)
D = pd.DataFrame(recs); print(len(D), "evaluated of", len(rows))
# EXPLORATORY (added 2026-11-04 after viewing partial results; not pre-registered): cosine of each output to its own
# unconverted source clip (how much of the source voice remains), and real speakers' nearest-other-speaker cosine per key.
src_e = {r.src_id: np.load(os.path.join(EV + "_emb", ev[r.file]["emb_key"] + "_ecapa.npy")) for r in D[D.cond == "SRC"].itertuples()}
src_e = {k: v / np.linalg.norm(v) for k, v in src_e.items()}
D["cos_to_source"] = [float(np.load(os.path.join(EV + "_emb", ev[f]["emb_key"] + "_ecapa.npy")) @ src_e[s] / np.linalg.norm(np.load(os.path.join(EV + "_emb", ev[f]["emb_key"] + "_ecapa.npy"))))
                      if s in src_e else np.nan for f, s in zip(D.file, D.src_id)]
G_ = Ew @ Ew.T; np.fill_diagonal(G_, -1); real_nn = G_.max(1)
# p_female_audeering is already P(female | adult) = p_female / (p_female + p_male) from the toolkit
q = lambda v: {p: float(np.nanpercentile(v, p)) for p in (5, 25, 50, 75, 95)}


def summ(sub):
    o = {"n": int(len(sub)), "max_cos_real": q(sub.max_cos_real), "frac_ge_theta": float((sub.max_cos_real >= th).mean()),
         "frac_ge_theta1": float((sub.max_cos_real >= th1).mean()), "utmos_mean": float(sub.utmos.mean()), "cer_mean": float(sub.cer.mean()),
         "p_female_meter_mean": float(sub.p_female_meter.mean()), "p_female_audeering_mean": float(sub.p_female_audeering.mean())}
    o["exploratory_cos_to_source_mean"] = float(np.nanmean(sub.cos_to_source)); o["exploratory_cos_to_source"] = q(sub.cos_to_source.dropna()) if sub.cos_to_source.notna().any() else {}
    tg = sub.target_gender_label.dropna()
    if len(tg):
        s2 = sub.loc[tg.index]; want_f = (tg == "female").values
        o["gender_agree_meter"] = float(((s2.p_female_meter > 0.5).values == want_f).mean())
        o["gender_agree_audeering"] = float(((s2.p_female_audeering > 0.5).values == want_f).mean())
    return o


R = {"theta": th, "theta1": th1, "n_rows": len(rows), "n_evaluated": int(len(D)), "by_condition": {}, "by_source": {},
     "exploratory_real_nearest_other": {"quantiles": q(real_nn), "frac_ge_theta": float((real_nn >= th).mean()), "frac_ge_theta1": float((real_nn >= th1).mean())}}
for c, sub in D.groupby("cond"):
    R["by_condition"][c] = summ(sub)
    if c.startswith("C2"):
        R["by_condition"][c].update({"cos_target_half2": q(sub.cos_target_half2), "cos_target_mean": float(sub.cos_target_half2.mean()),
                                     "target_top1": float((sub.rank_target_half2 == 1).mean()), "target_top10": float((sub.rank_target_half2 <= 10).mean()),
                                     "target_median_rank": float(sub.rank_target_half2.median()), "frac_cos_target_ge_theta": float((sub.cos_target_half2 >= th).mean()),
                                     "nearest_real_is_target": float(sub.nearest_is_target.mean())})
for (c, s), sub in D.groupby(["cond", "src_id"]):
    R["by_source"][f"{c}|{s}"] = summ(sub)
d2 = D[D.cond == "D2"]
if len(d2):
    R["D2_by_alpha"] = {str(a): {"max_cos_real": q(s.max_cos_real), "frac_ge_theta": float((s.max_cos_real >= th).mean()),
                                  "cos_endpoint_a_mean": float(np.mean([c[0] for c in s.cos_endpoints])), "cos_endpoint_b_mean": float(np.mean([c[1] for c in s.cos_endpoints])),
                                  "nearest_is_endpoint": float(np.mean([n_ in e_ for n_, e_ in zip(s.nearest_real, s.endpoints)]))} for a, s in d2.groupby("alpha")}
# E demos: effect of +-2 SD per PC relative to N0 of the same source
ed = D[D.cond.isin(["E_pc", "N0"])]; eff = {}
if len(ed) and "rm" in ed and "pc" in ed and (ed.cond == "E_pc").any():
    for pc in range(1, 9):
        o = {}
        for s in (-2, 2):
            sub = ed[(ed.cond == "E_pc") & (ed.pc == pc) & (ed.sd_step == s)]
            dd = []
            for _, r in sub.iterrows():
                b = ed[(ed.cond == "N0") & (ed.src_id == r.src_id)].iloc[0]
                if isinstance(r.rm, dict) and isinstance(b.rm, dict):
                    dd.append({k: r.rm[k] - b.rm[k] for k in r.rm if k in b.rm} | {"p_female_meter": r.p_female_meter - b.p_female_meter, "utmos": r.utmos - b.utmos,
                                                                                   "f0_ratio": r.rm.get("f0_median_hz", np.nan) / b.rm.get("f0_median_hz", np.nan)})
            o[str(s)] = pd.DataFrame(dd).mean().to_dict() if dd else {}
        eff[f"PC{pc}"] = o
R["E_demo_effects_vs_N0"] = eff
json.dump(R, open(os.path.join(RES, "tc_eval.json"), "w"), indent=1, ensure_ascii=False, default=float)
D.drop(columns=["rm"], errors="ignore").to_csv(os.path.join(RES, "tc_eval_rows.csv"), index=False)


def save(fig, name, data, pdf=False):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    if pdf:
        fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig); json.dump(data, open(os.path.join(FIG, name + ".json"), "w"), indent=1, ensure_ascii=False, default=float)


BC = R["by_condition"]; order = [c for c in ["SRC", "N0", "D1", "D2", "D3", "C2_k4", "C2_k16", "C2_k64", "C2_k256"] if c in BC]
LAB = {"SRC": "stock TTS (unconverted)", "N0": "TTS -> own vector", "D1": "gender x age cell means", "D2": "interpolations", "D3": "Gaussian samples",
       "C2_k4": "real target, k = 4", "C2_k16": "real target, k = 16", "C2_k64": "real target, k = 64", "C2_k256": "real target, k = 256"}
# main: novelty
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.4, 1]})
for i, c in enumerate(order):
    v = D[D.cond == c].max_cos_real.values; ax[0].scatter(v, np.full(len(v), i) + np.random.default_rng(0).uniform(-0.25, 0.25, len(v)), s=3, alpha=0.5,
                                                          color="#c0703f" if c.startswith("C2") else "#2f5d8a")
    ax[0].plot(np.median(v), i, "k|", ms=10)
X1 = np.load(os.path.join(RES, "tc_x1_short_real_maxcos.npy"))   # exploratory X1: 5-s real clip vs own / other keys' whole embeddings
for yy, v in ((-3, real_nn), (-2, X1[:, 3]), (-1, X1[:, 2])):   # v2 columns: 5-s first-half clip vs second-half embeddings (disjoint)
    ax[0].scatter(v, np.full(len(v), yy) + np.random.default_rng(1).uniform(-0.25, 0.25, len(v)), s=1, alpha=0.12, color="#8c8c8c"); ax[0].plot(np.median(v), yy, "k|", ms=10)
ax[0].axvline(th, color="#b03030", ls="--", lw=0.8); ax[0].text(th, len(order) - 0.4, " θ", color="#b03030")
ax[0].set_yticks(range(-3, len(order)), ["real, whole: nearest other key", "real 5-s clip: nearest other key's half 2", "real 5-s clip: own half 2"] + [LAB[c] for c in order]); ax[0].set_xlabel("max ECAPA cosine to any of 2,410 real speakers"); ax[0].set_title("a  nearest real speaker", loc="left")
cs = [c for c in order if c.startswith("C2")]
if cs:
    kk = [int(c.split("k")[1]) for c in cs]
    ax[1].plot(kk, [BC[c]["cos_target_mean"] for c in cs], "o-", color="#c0703f", label="mean cosine to target")
    ax[1].plot(kk, [BC[c]["target_top10"] for c in cs], "s-", color="#2f5d8a", label="target in top 10 of 2,410")
    ax[1].plot(kk, [BC[c]["nearest_real_is_target"] for c in cs], "^-", color="#5a8f62", label="nearest real = target")
    ax[1].set_xscale("log", base=2); ax[1].set_xticks(kk, [str(k) for k in kk]); ax[1].set_ylim(0, 1); ax[1].set_xlabel("components kept, k"); ax[1].legend(fontsize=6.5)
    ax[1].set_title("b  real-target conversions", loc="left")
save(fig, "generated_voices_novelty_and_real_target_conversion", {c: BC[c] for c in order} | {"theta": th, "theta1": th1}, pdf=True)
# supplementary: gender agreement, UTMOS, CER by condition
for key, name, ylab in (("utmos_mean", "utmos_by_condition", "mean UTMOS (relative only)"), ("cer_mean", "cer_by_condition", "mean CER")):
    fig, ax = plt.subplots(figsize=(5, 2.6)); ax.bar(range(len(order)), [BC[c][key] for c in order], color="#2f5d8a"); ax.set_xticks(range(len(order)), [LAB[c] for c in order], rotation=40, ha="right", fontsize=6.5)
    ax.set_ylabel(ylab); save(fig, name, {c: BC[c][key] for c in order})
gc = [c for c in order if "gender_agree_meter" in BC[c]]
fig, ax = plt.subplots(figsize=(5, 2.6)); x = np.arange(len(gc))
ax.bar(x - 0.2, [BC[c]["gender_agree_meter"] for c in gc], 0.4, label="ECAPA gender meter", color="#2f5d8a"); ax.bar(x + 0.2, [BC[c]["gender_agree_audeering"] for c in gc], 0.4, label="audeering", color="#c0703f")
ax.set_xticks(x, [LAB[c] for c in gc], rotation=30, ha="right", fontsize=6.5); ax.set_ylim(0, 1.25); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_ylabel("agreement with intended gender"); ax.legend(fontsize=7, ncol=2, loc="upper center")
save(fig, "gender_classifier_agreement_by_condition", {c: {k: BC[c][k] for k in ("gender_agree_meter", "gender_agree_audeering")} for c in gc})
if "D2_by_alpha" in R:
    a2 = R["D2_by_alpha"]; al = sorted(a2, key=float)
    fig, ax = plt.subplots(figsize=(4, 2.6)); ax.plot([float(a) for a in al], [a2[a]["cos_endpoint_a_mean"] for a in al], "o-", label="endpoint a")
    ax.plot([float(a) for a in al], [a2[a]["cos_endpoint_b_mean"] for a in al], "o-", label="endpoint b"); ax.plot([float(a) for a in al], [a2[a]["frac_ge_theta"] for a in al], "s--", label="fraction max cos >= θ")
    ax.set_xlabel("interpolation weight α (toward b)"); ax.legend(fontsize=7); save(fig, "interpolation_endpoint_similarity", a2)
if eff:
    keys = ["f0_ratio", "F1_hz", "F2_hz", "F3_hz", "p_female_meter", "utmos"]
    M = np.array([[eff[f"PC{p}"].get(str(s), {}).get(k, np.nan) for k in keys] for p in range(1, 9) for s in (-2, 2)])
    fig, ax = plt.subplots(figsize=(5.5, 4)); Mz = (M - np.nanmean(M, 0)) / np.nanstd(M, 0)
    ax.imshow(Mz, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto"); ax.set_xticks(range(len(keys)), ["F0 ratio", "ΔF1", "ΔF2", "ΔF3", "ΔP(female)", "ΔUTMOS"], fontsize=7)
    ax.set_yticks(range(16), [f"PC{p} {'-' if s < 0 else '+'}2 SD" for p in range(1, 9) for s in (-2, 2)], fontsize=6.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): ax.text(j, i, f"{M[i, j]:.2f}" if abs(M[i, j]) < 10 else f"{M[i, j]:.0f}", ha="center", va="center", fontsize=5.5, color="w" if abs(Mz[i, j]) > 1.4 else "k")
    save(fig, "pc_moves_measured_effects", eff)
print(json.dumps({c: {k: v for k, v in BC[c].items() if not isinstance(v, dict)} for c in order}, indent=0, ensure_ascii=False)[:3000])
