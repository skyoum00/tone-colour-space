"""Balanced ECAPA gender meter (P3 metric 2) + the non-study speaker splits used by validate.py.

Speakers: AI Hub 71631 INDOOR speakers whose ID is NOT among the 2,525 eligible study speakers
(stage1/results/speaker_index.parquet, column `eligible`; matched on the AI Hub speaker ID so a study speaker can never
enter under another gender/age key). Only the parquet is read from stage1/results. For each speaker the parquet already
holds the best-SNR indoor conversation and the own, non-overlapping 1-15 s segments (build_speaker_index.own_segments).
Audio is read from the AI Hub zip on the NAS, the speaker's channel picked by energy (measure_voice.pick_channel),
segments outside a truncated wav dropped (as measure_voice.measure), concatenated, and the first 30 s embedded with the
study ECAPA (measure_voice.embed). Speakers with < 30 s in-bounds speech are skipped.

Split (seed 0, random order within gender): train = first 150 F + 150 M usable; held-out = next 50 F + 50 M usable.
The held-out 30-s clips are written to validation_clips/heldout/ (NAS only, never distributed) for validate.py.
Classifier: LogisticRegressionCV(Cs=10, inner 5-fold, class_weight balanced) on L2-normalised 192-d embeddings;
reported: outer StratifiedKFold(5, shuffle, seed 0) CV accuracy on the training speakers.
Outputs: gender_meter_balanced.joblib, gender_meter_balanced.json, splits.json.
"""
import io, json, os, sys, zipfile
import numpy as np, pandas as pd, soundfile as sf, joblib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "stage1")))
import measure_voice as mv  # noqa: E402
from sklearn.linear_model import LogisticRegressionCV  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402

SR = 16000
CLIP_S = 30
N_TRAIN, N_HELD = 150, 50
CLIPS = os.path.join(HERE, "validation_clips")


def non_study_speakers():
    idx = pd.read_parquet(os.path.join(mv.OUT, "speaker_index.parquet"))
    idx["id"] = idx.speaker.str.split("_").str[0]
    study_ids = set(idx.loc[idx.eligible, "id"])
    ns = idx[~idx.id.isin(study_ids)].copy()
    return ns, len(study_ids)


def own_audio(row, zm, _cache={}):
    """Concatenated own-channel speech (float32, 16 kHz) of one parquet row, or None."""
    if row["file"] not in zm:
        return None
    zp, member = zm[row["file"]]
    z = _cache.setdefault(zp, zipfile.ZipFile(zp))
    x, sr = sf.read(io.BytesIO(z.read(member)), dtype="float32")
    assert sr == SR and x.ndim == 2, (sr, x.shape)
    segs = [s for s in json.loads(row["segments"]) if s["end"] <= len(x) / SR]
    if not segs:
        return None
    ch, sep = mv.pick_channel(x, segs)
    return np.concatenate([x[int(s["start"] * SR):int(s["end"] * SR), ch] for s in segs]), ch, sep, segs


def main():
    ns, n_study = non_study_speakers()
    ns = ns[ns.speech_s >= CLIP_S]
    zm = mv.zip_map()
    rng = np.random.default_rng(0)
    mv.init_worker()
    os.makedirs(os.path.join(CLIPS, "heldout"), exist_ok=True)
    split = {"train": [], "heldout": []}
    X, y = [], []
    for g, lab in (("여성", 1), ("남성", 0)):
        cand = ns[ns.gender == g].to_dict("records")
        order = rng.permutation(len(cand))
        nt = nh = 0
        for i in order:
            if nh >= N_HELD:
                break
            r = cand[i]
            got = own_audio(r, zm)
            if got is None or len(got[0]) < CLIP_S * SR:
                continue
            a, ch, sep, segs = got
            a30 = a[:CLIP_S * SR]
            info = {"speaker": r["speaker"], "gender": g, "age": r["age"], "file": r["file"], "split_aihub": r["split"],
                    "snr_db": int(r["snr_db"]), "channel": ch, "channel_sep_db": round(sep, 2), "own_speech_s": round(len(a) / SR, 2)}
            if nt < N_TRAIN:
                X.append(mv.embed([a30])[0]); y.append(lab); split["train"].append(info); nt += 1
            else:
                fn = f"heldout/{r['speaker']}_30s.wav"
                sf.write(os.path.join(CLIPS, fn), a30, SR, subtype="FLOAT")
                info["clip"] = fn; split["heldout"].append(info); nh += 1
        print(g, "train", nt, "heldout", nh, flush=True)
    X = np.array(X); X = X / np.linalg.norm(X, axis=1, keepdims=True); y = np.array(y)
    clf = LogisticRegressionCV(Cs=10, cv=5, max_iter=5000, class_weight="balanced")
    acc = cross_val_score(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    clf.fit(X, y)
    joblib.dump(clf, os.path.join(HERE, "gender_meter_balanced.joblib"))
    tr = pd.DataFrame(split["train"])
    meta = {"n_female": int(y.sum()), "n_male": int((1 - y).sum()), "cv_accuracy_mean": float(acc.mean()),
            "cv_accuracy_sd": float(acc.std()), "cv_accuracy_folds": acc.tolist(), "C_selected": float(clf.C_[0]),
            "train_accuracy": float(clf.score(X, y)), "label": "1 = female (AI Hub Gender 여성), 0 = male (남성)",
            "source": "AI Hub 71631 indoor (TS_01/VS_01), speakers whose ID is not among the eligible study speakers",
            "n_study_ids_excluded": n_study, "n_non_study_indoor_with_30s": int(len(ns)),
            "features": "study ECAPA (measure_voice.embed, speechbrain/spkrec-ecapa-voxceleb) 192-d, L2-normalised, first 30 s own-channel speech",
            "classifier": "sklearn LogisticRegressionCV(Cs=10, cv=5, class_weight=balanced, max_iter=5000)",
            "outer_cv": "StratifiedKFold(5, shuffle=True, random_state=0)", "selection": "seed 0 random order within gender",
            "train_age_by_gender": tr.groupby("gender").age.value_counts().unstack(fill_value=0).to_dict("index"),
            "train_snr_db_median": float(tr.snr_db.median()), "train_snr_db_min": int(tr.snr_db.min())}
    json.dump(meta, open(os.path.join(HERE, "gender_meter_balanced.json"), "w"), ensure_ascii=False, indent=1)
    json.dump(split, open(os.path.join(HERE, "splits.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: meta[k] for k in ("n_female", "n_male", "cv_accuracy_mean", "cv_accuracy_folds")}))


if __name__ == "__main__":
    main()
