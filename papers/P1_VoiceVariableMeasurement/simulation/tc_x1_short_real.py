"""Exploratory X1 (PAPER_PLAN.md execution log 2026-11-04 16:35; requested by review round 1; not pre-registered).
Duration- and domain-matched real reference for the novelty test: ECAPA of the first 5 s of each real key's first-half
speech, embedded in one call (as the converted clips are), compared with (a) the key's own whole-speech embedding and
(b) the whole-speech embeddings of all other keys. Thresholds theta / theta1 unchanged (results/tc_analysis.json D0).
CPU only (nice, 4 reader threads). Output: results/tc_x1_short_real.json."""
import io, json, os, sys, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SIM = os.path.join(HERE, "../../../simulation")
sys.path.insert(0, SIM); sys.path.insert(0, os.path.join(SIM, "stage1"))
import common_data as C, measure_voice as mv
import config
import torch; torch.set_num_threads(4)
from speechbrain.inference.speaker import EncoderClassifier
D_SEC = 5.0
T = json.load(open(os.path.join(HERE, "results/tc_analysis.json"))); th, th1 = T["D0_threshold"]["theta_eer"], T["D0_threshold"]["theta1_same_speaker_p1"]
df = C.load("primary"); spk = df.index.values; n = len(df)
Ew = np.vstack(df.ecapa.values); Ew /= np.linalg.norm(Ew, axis=1, keepdims=True)
Eh2 = np.vstack(df.ecapa_half2.values); Eh2 /= np.linalg.norm(Eh2, axis=1, keepdims=True)   # v2: disjoint from the first-half clip
ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=config.ECAPA_DIR, run_opts={"device": "cpu"})
zm = mv.zip_map()


def first5(row):
    import soundfile as sf
    zp, member = zm[row["file"]]
    x, _ = sf.read(io.BytesIO(zipfile.ZipFile(zp).read(member)), dtype="float32")
    segs = [s for s in json.loads(row["segments"]) if s["end"] <= len(x) / mv.SR]
    ch, _ = mv.pick_channel(x, segs)
    audio = [x[int(s["start"] * mv.SR):int(s["end"] * mv.SR), ch] for s in segs]
    return np.concatenate(audio[:len(audio) // 2])[:int(D_SEC * mv.SR)]


rows = df.reset_index()[["speaker", "file", "segments"]].to_dict("records")
Q = np.zeros((n, Ew.shape[1]))
with ThreadPoolExecutor(4) as ex:      # read all clips first (every first half is >= 20.8 s, so all clips are exactly 5 s)
    Y = list(ex.map(first5, rows))
L = np.array([len(y) / mv.SR for y in Y]); print("read", n, "min len", L.min(), flush=True)
full = np.where(L == D_SEC)[0]; short = np.where(L < D_SEC)[0]; print("full", len(full), "short", len(short), flush=True)
for j in range(0, len(full), 64):     # equal-length clips: batching gives the same embeddings as one-by-one encoding
    ix = full[j:j + 64]
    with torch.no_grad():
        e = ecapa.encode_batch(torch.tensor(np.stack([Y[i] for i in ix]))).squeeze(1).numpy()
    Q[ix] = e / np.linalg.norm(e, axis=1, keepdims=True)
    print(j, len(full), flush=True)
for i in short:                        # any shorter clip: encoded alone
    with torch.no_grad():
        e = ecapa.encode_batch(torch.tensor(Y[i])[None]).squeeze().numpy()
    Q[i] = e / np.linalg.norm(e)
np.save(os.path.join(HERE, "results/tc_x1_short_real_clip_ecapa.npy"), Q)
S2 = Q @ Eh2.T; own2 = S2[np.arange(n), np.arange(n)]; rank2 = (S2 > own2[:, None]).sum(1) + 1; So2 = S2.copy(); So2[np.arange(n), np.arange(n)] = -1; mo2 = So2.max(1)
S = Q @ Ew.T; own = S[np.arange(n), np.arange(n)]; rank = (S > own[:, None]).sum(1) + 1
So = S.copy(); So[np.arange(n), np.arange(n)] = -1; mo = So.max(1)
q = lambda v: {p: float(np.percentile(v, p)) for p in (1, 5, 25, 50, 75, 95, 99)}
R = {"note": "exploratory X1, not pre-registered", "duration_s": D_SEC, "n": int(n), "clip_seconds_min": float(L.min()), "n_shorter_than_duration": int((L < D_SEC).sum()), "theta": th, "theta1": th1,
     "own_whole": {"cos": q(own), "frac_ge_theta": float((own >= th).mean()), "frac_ge_theta1": float((own >= th1).mean()),
                   "top1": float((rank == 1).mean()), "top10": float((rank <= 10).mean()), "median_rank": float(np.median(rank))},
     "max_other_whole": {"cos": q(mo), "frac_ge_theta": float((mo >= th).mean()), "frac_ge_theta1": float((mo >= th1).mean())},
     "v2_vs_second_half": {"own_half2": {"cos": q(own2), "frac_ge_theta": float((own2 >= th).mean()), "frac_ge_theta1": float((own2 >= th1).mean()),
                                         "top1": float((rank2 == 1).mean()), "top10": float((rank2 <= 10).mean()), "median_rank": float(np.median(rank2))},
                           "max_other_half2": {"cos": q(mo2), "frac_ge_theta": float((mo2 >= th).mean()), "frac_ge_theta1": float((mo2 >= th1).mean())},
                           "max_any_half2": {"cos": q(S2.max(1)), "frac_ge_theta": float((S2.max(1) >= th).mean())}},
     "max_any_whole_including_own": {"cos": q(S.max(1)), "frac_ge_theta": float((S.max(1) >= th).mean())}}
json.dump(R, open(os.path.join(HERE, "results/tc_x1_short_real.json"), "w"), indent=1)
np.save(os.path.join(HERE, "results/tc_x1_short_real_maxcos.npy"), np.c_[own, mo, own2, mo2])   # cols: own whole, other whole, own half2, other half2
print(json.dumps(R, indent=1))
