"""Shared loader for the three papers: measurements (retry lines replace error lines, last line per speaker wins),
speaker index, A6 primary-sample filter, tone-colour vectors."""
import json, os
import numpy as np, pandas as pd
import config  # paths (config.py; env PKV_STAGE1_RESULTS)
RES = config.STAGE1_RESULTS


def load(sample="primary", with_tone_colour=False):
    idx = pd.read_parquet(os.path.join(RES, "speaker_index.parquet")).set_index("speaker")
    rows = {}
    for f in ("measurements.jsonl", "measurements_retry.jsonl"):
        p = os.path.join(RES, f)
        if os.path.exists(p):
            for l in open(p):
                d = json.loads(l)
                if "error" not in d or d["speaker"] not in rows:
                    rows[d["speaker"]] = d
    ok = [d for d in rows.values() if "error" not in d]
    df = pd.DataFrame(ok).set_index("speaker").join(idx[["gender", "age", "device", "snr_db", "split", "file", "segments"]])
    if sample == "primary":
        chk = pd.read_csv(os.path.join(RES, "audio_label_duration_check.csv")).set_index("speaker")
        df = df[df.index.isin(chk.index[(chk.wav_over_label >= 0.99) & (chk.wav_over_label <= 1.01)])]
    if with_tone_colour:
        tc = {d["speaker"]: d for d in map(json.loads, open(os.path.join(RES, "tone_colour.jsonl"))) if "error" not in d}
        df = df[df.index.isin(tc)]
        for k in ("se", "se_half1", "se_half2"):
            df[k] = [np.array(tc[s][k]) for s in df.index]
    return df
