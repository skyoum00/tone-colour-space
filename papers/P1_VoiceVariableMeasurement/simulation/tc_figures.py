"""P1 tone-colour figures (png + json pairs, names describe content) from results/tc_analysis.json, results/tc_gpu.json,
stage1 tone_colour_prediction.json (A7) and h1_results.json (ECAPA Gate 1). Main-text figures are also saved as pdf."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); SIM = os.path.join(HERE, "../../../simulation")
sys.path.insert(0, SIM); sys.path.insert(0, os.path.join(SIM, "stage1"))
FIG = os.path.join(HERE, "figures"); RES = os.path.join(HERE, "results"); os.makedirs(FIG, exist_ok=True)
T = json.load(open(os.path.join(RES, "tc_analysis.json"))); G = json.load(open(os.path.join(RES, "tc_gpu.json")))
import config  # noqa: E402  (stage-1 results directory, env PKV_STAGE1_RESULTS)
A7 = json.load(open(os.path.join(config.STAGE1_RESULTS, "tone_colour_prediction.json"))); H = json.load(open(os.path.join(config.STAGE1_RESULTS, "h1_results.json")))
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
C = {"a": "#2f5d8a", "b": "#c0703f", "c": "#5a8f62", "g": "#8c8c8c", "r": "#b03030"}
VN = {"f0_mean_hz": "log mean F0", "f0_range_st": "F0 range", "jitter_pct": "jitter", "shimmer_db": "shimmer", "hnr_db": "HNR", "h1_h2_db": "H1-H2",
      "slope_0_500": "slope 0-0.5 kHz", "slope_500_1500": "slope 0.5-1.5 kHz", "alpha_ratio_db": "alpha ratio", "F1_hz": "F1", "F2_hz": "F2", "F3_hz": "F3",
      "F4_hz": "F4", "vtl_cm": "vocal-tract length", "low_band_0_1k_db": "energy <1 kHz", "spk_formant_2_5_3_5k_db": "energy 2.5-3.5 kHz",
      "spectral_centroid_hz": "spectral centroid", "speech_rate_syl_s": "speech rate", "final_f0_slope_st_s": "final F0 slope"}


def save(fig, name, data, pdf=False):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    if pdf:
        fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig); json.dump(data, open(os.path.join(FIG, name + ".json"), "w"), indent=1, ensure_ascii=False, default=float)


def thr_lines(ax, kind):
    if kind == "eer":
        ax.axhline(0.02, color=C["r"], ls="--", lw=0.8)
    if kind == "top10":
        ax.axhline(0.90, color=C["r"], ls="--", lw=0.8)


# ---- main: reliability and duration (A1, A2) ----------------------------------------------------------------------------------
A1 = T["A1_split_half"]; D = G["A2_duration"]["by_duration"]; ds = [int(k) for k in D]; ref = G["A2_duration"]["full_half1_reference"]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
ax[0].plot(ds, [D[str(d)]["eer"] for d in ds], "o-", color=C["a"]); ax[0].axhline(ref["eer"], color=C["g"], ls=":"); thr_lines(ax[0], "eer")
ax[0].set_xscale("log"); ax[0].set_xticks(ds, [str(d) for d in ds]); ax[0].set_xlabel("first-half speech used (s)"); ax[0].set_ylabel("EER"); ax[0].set_title("a  verification", loc="left")
for key, col, lab in (("top1", C["b"], "top-1"), ("top10", C["a"], "top-10")):
    ax[1].plot(ds, [D[str(d)][key] for d in ds], "o-", color=col, label=lab); ax[1].axhline(ref[key], color=col, ls=":", lw=0.8)
thr_lines(ax[1], "top10"); ax[1].set_xscale("log"); ax[1].set_xticks(ds, [str(d) for d in ds]); ax[1].set_ylim(0, 1); ax[1].set_xlabel("first-half speech used (s)")
ax[1].set_ylabel("identification rate (2,408)"); ax[1].legend(fontsize=7); ax[1].set_title("b  identification", loc="left")
qs = A1["same_speaker_cos_quantiles"], A1["different_speaker_cos_quantiles"]
for i, (qq, col, lab) in enumerate(zip(qs, (C["a"], C["g"]), ("same key, half 1 vs half 2", "different keys"))):
    p = [1, 5, 25, 50, 75, 95, 99]; v = [qq[str(x)] if str(x) in qq else qq[x] for x in p]
    ax[2].plot([v[0], v[-1]], [i, i], color=col, lw=1); ax[2].plot([v[1], v[-2]], [i, i], color=col, lw=3); ax[2].plot([v[2], v[4]], [i, i], color=col, lw=7); ax[2].plot(v[3], i, "w|", ms=8)
ax[2].set_yticks([0, 1], ["same key", "different keys"]); ax[2].set_ylim(-0.6, 1.6); ax[2].set_xlabel("cosine of tone-colour vectors"); ax[2].set_title("c  split-half cosine", loc="left")
save(fig, "tone_colour_reliability_and_duration", {"duration": D, "full_half_reference": ref, "split_half": A1, "thresholds": {"eer": 0.02, "top10": 0.90},
     "stable_duration_s": G["A2_duration"]["stable_duration_s"], "half1_seconds_quantiles": G["A2_duration"]["half1_seconds_quantiles"]}, pdf=True)

# ---- main: PCA spectrum and knob count (B1, C1) ----------------------------------------------------------------------------------
B1 = T["B1_pca"]; K = T["C1_svd_knobs"]["by_k"]; ks = [int(k) for k in K]
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6))
lam = np.array(B1["eigenvalues"]); ax[0].semilogy(np.arange(1, 129), lam[:128] / lam.sum(), color=C["a"]); ax0 = ax[0].twinx()
ax0.plot(np.arange(1, 257), B1["cumulative"], color=C["b"]); ax0.set_ylim(0, 1.02); ax0.spines["top"].set_visible(False)
ax[0].set_xlabel("component"); ax[0].set_ylabel("variance share (log)"); ax0.set_ylabel("cumulative", color=C["b"]); ax[0].set_title(f"a  spectrum, PR = {B1['participation_ratio']:.1f}", loc="left")
ax[1].plot(ks, [K[str(k)]["raw"]["eer"] for k in ks], "o-", color=C["a"]); thr_lines(ax[1], "eer"); ax[1].set_xscale("log", base=2); ax[1].set_xticks(ks, [str(k) for k in ks], fontsize=6)
ax[1].set_yscale("log"); ax[1].set_xlabel("components kept, k"); ax[1].set_ylabel("EER"); ax[1].set_title("b  verification", loc="left")
for key, col, lab in (("top1", C["b"], "top-1"), ("top10", C["a"], "top-10")):
    ax[2].plot(ks, [K[str(k)]["raw"][key] for k in ks], "o-", color=col, label=lab)
ax[2].plot(ks, [K[str(k)]["between_speaker_variance_retained"] for k in ks], "s--", color=C["c"], ms=3, label="variance kept")
thr_lines(ax[2], "top10"); ax[2].set_xscale("log", base=2); ax[2].set_xticks(ks, [str(k) for k in ks], fontsize=6); ax[2].set_ylim(0, 1.02); ax[2].set_xlabel("components kept, k")
ax[2].legend(fontsize=6.5); ax[2].set_title(f"c  identification, k* = {T['C1_svd_knobs']['knob_count']}", loc="left")
save(fig, "tone_colour_pca_spectrum_and_knob_count", {"pca": {k: v for k, v in B1.items() if k != "eigenvalues"}, "eigen_share": (lam / lam.sum()).tolist(),
     "knobs": T["C1_svd_knobs"]}, pdf=True)

# ---- main: labels and structure (B2, B3, E) ------------------------------------------------------------------------------------------
B2, B3, E = T["B2_gender_age"], T["B3_device_snr_label_effects"], T["E_pc_interpretation_labels"]; Ek = G["E_krr_variables_to_pcs"]
rows = [("gender (linear)", E["gender_r2_per_pc"]), ("age band (linear)", E["age_band_r2_per_pc"]), ("device/SNR labels (linear)", E["device_snr_labels_r2_per_pc"]),
        ("19 core variables (kernel)", Ek["core19"]["krr_r2_per_pc"]), ("135 interpretable variables (kernel)", Ek["interpretable_pool"]["krr_r2_per_pc"])]
M = np.clip(np.array([r[1] for r in rows]), 0, 1)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1, 1.5]})
bars = [("gender", B2["gender_ridge_r2"]), ("age band", B2["age_band_ridge_r2"]), ("gender + age", B2["gender_plus_age_ridge_r2"]),
        ("device/SNR labels", B3["device_snr_labels_ridge_r2"]), ("+ device/SNR over\ngender + age", B3["increment_over_gender_age"])]
ax[0].barh([b[0] for b in bars][::-1], [b[1] for b in bars][::-1], color=[C["a"], C["a"], C["a"], C["b"], C["b"]][::-1])
for i, b in enumerate(bars[::-1]): ax[0].text(b[1] + 0.003, i, f"{b[1]:.3f}", va="center", fontsize=7)
ax[0].set_xlabel("out-of-fold $R^2$ of the whole vector"); ax[0].set_title(f"a  label effects (gender AUC {B2['gender_direction_auc_female']:.3f})", loc="left")
im = ax[1].imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto"); ax[1].set_yticks(range(len(rows)), [r[0] for r in rows]); ax[1].set_xticks(range(8), [f"PC{i + 1}" for i in range(8)])
for i in range(M.shape[0]):
    for j in range(8): ax[1].text(j, i, f"{rows[i][1][j]:.2f}", ha="center", va="center", fontsize=6, color="w" if M[i, j] > 0.55 else "k")
ax[1].set_title("b  out-of-fold $R^2$ of PC scores", loc="left")
save(fig, "tone_colour_labels_and_component_meaning", {"labels": dict(bars), "gender_direction_auc": B2["gender_direction_auc_female"],
     "age_direction": B2["age_direction_spearman_within_gender"], "pc_r2": {r[0]: r[1] for r in rows}, "pc_variance_share": E["variance_share"]}, pdf=True)

# ---- supplementary ---------------------------------------------------------------------------------------------------------------
# split-half centred vs raw
fig, ax = plt.subplots(figsize=(4, 2.6)); keys = ["eer", "top1", "top10"]; x = np.arange(3)
ax.bar(x - 0.2, [A1["raw"][k] for k in keys], 0.4, label="raw cosine", color=C["a"]); ax.bar(x + 0.2, [A1["centred"][k] for k in keys], 0.4, label="centred cosine", color=C["b"])
ax.set_xticks(x, ["EER", "top-1", "top-10"]); ax.legend(); save(fig, "split_half_raw_vs_centred_cosine", {"raw": A1["raw"], "centred": A1["centred"]})
fig, ax = plt.subplots(figsize=(4, 2.6)); ax.plot(ds, [D[str(d)]["mean_cos_to_own"] for d in ds], "o-", color=C["a"]); ax.set_xscale("log"); ax.set_xticks(ds, [str(d) for d in ds])
ax2 = ax.twinx(); ax2.bar(range(0), []); ax.set_xlabel("first-half speech used (s)"); ax.set_ylabel("mean cosine to own half-2 vector")
for d in ds: ax.annotate(f"n={D[str(d)]['n_speakers']}", (d, D[str(d)]["mean_cos_to_own"]), xytext=(0, 6), textcoords="offset points", fontsize=6, ha="center")
ax2.set_yticks([]); save(fig, "duration_mean_cosine_with_n", D)
fig, ax = plt.subplots(figsize=(4.5, 2.6)); r_ = B1["pc_split_half_r_first32"]; ax.bar(range(1, 33), r_, color=C["a"]); ax.set_xlabel("component"); ax.set_ylabel("split-half r of PC score")
save(fig, "pc_split_half_reliability", {"r": r_})
fig, ax = plt.subplots(figsize=(4, 2.6)); ax.plot(ks, [K[str(k)]["raw"]["mean_cos_to_own"] for k in ks], "o-", color=C["a"], label="raw")
ax.set_xscale("log", base=2); ax.set_xticks(ks, [str(k) for k in ks]); ax.set_xlabel("k"); ax.set_ylabel("mean cosine to own half 2"); save(fig, "knob_reconstruction_mean_cosine", {k: K[k]["raw"]["mean_cos_to_own"] for k in K})
fig, ax = plt.subplots(figsize=(4, 2.6)); ax.plot(ks, [K[str(k)]["centred"]["eer"] for k in ks], "o-", color=C["b"], label="centred"); ax.plot(ks, [K[str(k)]["raw"]["eer"] for k in ks], "o-", color=C["a"], label="raw")
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(ks, [str(k) for k in ks]); ax.legend(); ax.set_xlabel("k"); ax.set_ylabel("EER"); save(fig, "knob_eer_raw_vs_centred", {k: {"raw": K[k]["raw"]["eer"], "centred": K[k]["centred"]["eer"]} for k in K})
ag = B2["age_direction_spearman_within_gender"]
fig, ax = plt.subplots(figsize=(3.2, 2.4)); ax.bar(list(ag), [ag[k]["rho"] for k in ag], color=C["a"]); ax.set_ylabel("Spearman $\\rho$, age score vs age band"); save(fig, "age_direction_within_gender", ag)
nd = T["near_duplicate_components_ecapa"]
fig, ax = plt.subplots(figsize=(4.5, 2.6)); th = list(nd); x = np.arange(len(th))
ax.bar(x - 0.2, [nd[t]["n_components_size_ge_2"] for t in th], 0.4, label="components with >= 2 keys", color=C["a"]); ax.bar(x + 0.2, [nd[t]["largest_component"] for t in th], 0.4, label="largest component", color=C["b"])
ax.set_xticks(x, [f"ECAPA cos > {t}" for t in th]); ax.legend(fontsize=7); save(fig, "near_duplicate_component_structure", nd)
B4 = T["B4_near_duplicates"]; cols = ["same_gender", "same_age_band", "same_device", "same_conversation_file"]
fig, ax = plt.subplots(figsize=(5, 2.6)); x = np.arange(len(cols))
for i, (k, lab, col) in enumerate((("all_pairs_base_rate", "all pairs", C["g"]), ("ecapa_gt_0_85", "ECAPA cos > 0.85", C["a"]), ("tone_colour_quantile_matched", "tone colour, matched", C["b"]))):
    ax.bar(x + (i - 1) * 0.27, [B4[k][c] for c in cols], 0.27, label=lab, color=col)
ax.set_xticks(x, [c.replace("_", " ") for c in cols], fontsize=7); ax.set_ylabel("fraction of pairs"); ax.set_ylim(0, 1.35); ax.legend(fontsize=7, ncol=3, loc="upper center"); save(fig, "near_duplicate_metadata_agreement", B4)
sp = E["spearman_core19"]; Ms = np.array([sp[v] for v in VN])
fig, ax = plt.subplots(figsize=(5, 5)); im = ax.imshow(Ms, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto"); ax.set_yticks(range(len(VN)), list(VN.values()), fontsize=7)
ax.set_xticks(range(8), [f"PC{i + 1}" for i in range(8)]); fig.colorbar(im, shrink=0.6); save(fig, "pc_core_variable_spearman", sp)
sp2 = E["spearman_core19_within_female"]; Ms2 = np.array([sp2[v] for v in VN])
fig, ax = plt.subplots(figsize=(5, 5)); im = ax.imshow(Ms2, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto"); ax.set_yticks(range(len(VN)), list(VN.values()), fontsize=7)
ax.set_xticks(range(8), [f"PC{i + 1}" for i in range(8)]); fig.colorbar(im, shrink=0.6); ax.set_title("female speakers only"); save(fig, "pc_core_variable_spearman_within_female", sp2)
P = A7["pools"]; names = [("demographic_gender_age", "ridge", "gender + age labels"), ("core19", "rbf_krr", "19 core variables"), ("ecapa_reference", "rbf_krr", "ECAPA embedding"),
                          ("interpretable_minus_flagged", "rbf_krr", "interpretable minus flagged (121)"), ("interpretable", "rbf_krr", "interpretable variables (135)"), ("interpretable+mfcc+lfcc", "rbf_krr", "interpretable + MFCC + LFCC")]
fig, ax = plt.subplots(figsize=(5, 2.6)); v = [P[a][b]["r2_whole"] for a, b, _ in names]
ax.barh([n_ for *_, n_ in names], v, color=C["a"]); [ax.text(x_ + 0.01, i, f"{x_:.3f}", va="center", fontsize=7) for i, x_ in enumerate(v)]
ax.set_xlim(0, 1); ax.set_xlabel("out-of-fold $R^2$ of the tone-colour vector"); save(fig, "interpretable_measurements_predict_tone_colour", {n_: P[a][b] for a, b, n_ in names})
fig, ax = plt.subplots(figsize=(5, 2.6)); keys = [("lower_gender_mean_vector", "gender mean"), (None, "19 core variables"), (None, "interpretable (135)"), ("upper_own_half1_vector", "own half-1 vector")]
vals = [A7["lower_gender_mean_vector"]["top10"], P["core19"]["rbf_krr"]["convert"]["top10"], P["interpretable"]["rbf_krr"]["convert"]["top10"], A7["upper_own_half1_vector"]["top10"]]
ax.bar([k[1] for k in keys], vals, color=C["a"]); ax.set_ylabel("top-10 identification (2,408)"); ax.tick_params(axis="x", labelsize=7)
save(fig, "interpretable_prediction_identification", dict(zip([k[1] for k in keys], vals)))
fig, ax = plt.subplots(figsize=(5, 2.4)); vv = {"ECAPA, 19 core": H["primary_krr_r2"], "tone colour, 19 core": P["core19"]["rbf_krr"]["r2_whole"],
                                                  "ECAPA, 135 interpretable": H["amendments_A1_A2"]["interpretable_pool"]["krr_r2"], "tone colour, 135 interpretable": P["interpretable"]["rbf_krr"]["r2_whole"]}
ax.barh(list(vv)[::-1], list(vv.values())[::-1], color=[C["b"], C["a"], C["b"], C["a"]]); ax.axvline(0.70, color=C["r"], ls="--", lw=0.8); ax.set_xlim(0, 1)
ax.set_xlabel("out-of-fold $R^2$"); save(fig, "ecapa_versus_tone_colour_explained_by_variables", vv | {"gate": 0.70})
vn = T["D_vector_space_novelty"]
fig, ax = plt.subplots(figsize=(5, 2.6)); labs = [("real_same_speaker_half_cos", "real: own second half"), ("real_max_cos_to_other_speaker_whole", "real: nearest other key"),
                                                  ("D1_max_cos", "cell means"), ("D2_max_cos", "interpolations"), ("D3_max_cos", "Gaussian samples")]
for i, (k, lab) in enumerate(labs):
    qq = vn[k]; ax.plot([qq["5"], qq["95"]], [i, i], color=C["a"], lw=2); ax.plot([qq["25"], qq["75"]], [i, i], color=C["a"], lw=6); ax.plot(qq["50"], i, "w|", ms=8)
ax.set_yticks(range(len(labs)), [l for _, l in labs]); ax.set_xlabel("max cosine to real tone-colour vectors"); save(fig, "generated_vectors_novelty_in_vector_space", vn)
print("figures", len([f for f in os.listdir(FIG) if f.endswith('.png')]))
