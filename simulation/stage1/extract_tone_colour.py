"""E-VC V3 target (PREREG A7): OpenVoice v2 tone-colour embedding (ref_enc, 256-d) of every eligible study speaker,
computed on exactly the segments measure_voice.py uses (A6 truncation rule, louder channel), for the whole speech and for
the first/second half of the segments. Audio is resampled 16 kHz -> 22.05 kHz (converter rate) before the linear spectrogram.
Output: results/tone_colour.jsonl (resumable). Reads audio and segment labels only; no measured variable is read."""
import argparse, io, json, os, sys, types, zipfile
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
import measure_voice as mv
import config  # noqa: E402  (on sys.path via measure_voice)

CK = config.OPENVOICE_CKPT
OUT = os.path.join(mv.OUT, "tone_colour.jsonl")
_C = None


def init():
    global _C
    import torch
    torch.set_num_threads(2)
    sys.modules["openvoice.text"] = types.SimpleNamespace(text_to_sequence=None)
    sys.path.insert(0, config.OPENVOICE_DIR)
    from openvoice.api import ToneColorConverter, OpenVoiceBaseClass
    c = ToneColorConverter.__new__(ToneColorConverter); OpenVoiceBaseClass.__init__(c, CK + "config.json", device="cpu")
    c.watermark_model = None; c.version = getattr(c.hps, "_version_", "v1"); c.load_ckpt(CK + "checkpoint.pth")
    _C = c


def se(y16):
    import torch, librosa
    from openvoice.mel_processing import spectrogram_torch
    h = _C.hps.data
    y = torch.FloatTensor(librosa.resample(y16, orig_sr=mv.SR, target_sr=h.sampling_rate))[None]
    s = spectrogram_torch(y, h.filter_length, h.sampling_rate, h.hop_length, h.win_length, center=False)
    with torch.no_grad():
        return _C.model.ref_enc(s.transpose(1, 2)).squeeze().numpy().round(6).tolist()


def run(task):
    row, (zp, member) = task
    import soundfile as sf
    try:
        x, sr = sf.read(io.BytesIO(zipfile.ZipFile(zp).read(member)), dtype="float32")
        segs = json.loads(row["segments"]); dur = len(x) / mv.SR
        segs = [s for s in segs if s["end"] <= dur]
        if sum(s["end"] - s["start"] for s in segs) < 60:
            return {"speaker": row["speaker"], "error": "insufficient in-bounds speech"}
        ch, _ = mv.pick_channel(x, segs)
        audio = [x[int(s["start"] * mv.SR):int(s["end"] * mv.SR), ch] for s in segs]; h = len(audio) // 2
        return {"speaker": row["speaker"], "se": se(np.concatenate(audio)), "se_half1": se(np.concatenate(audio[:h])),
                "se_half2": se(np.concatenate(audio[h:]))}
    except Exception as ex:
        return {"speaker": row["speaker"], "error": f"{type(ex).__name__}: {ex}"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=2); a = ap.parse_args()
    idx = pd.read_parquet(os.path.join(mv.OUT, "speaker_index.parquet")); idx = idx[idx.eligible]
    zm = mv.zip_map()
    done = {json.loads(l)["speaker"] for l in open(OUT)} if os.path.exists(OUT) else set()
    tasks = [(r, zm[r["file"]]) for r in idx.to_dict("records") if r["file"] in zm and r["speaker"] not in done]
    print(f"eligible {len(idx)}, done {len(done)}, to run {len(tasks)}", flush=True)
    with ProcessPoolExecutor(a.workers, initializer=init) as ex, open(OUT, "a") as f:
        for i, res in enumerate(ex.map(run, tasks, chunksize=1)):
            f.write(json.dumps(res) + "\n"); f.flush()
            if i % 100 == 0:
                print(i, res["speaker"], res.get("error", "ok"), flush=True)


if __name__ == "__main__":
    main()
