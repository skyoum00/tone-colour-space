"""P1 conversions (PAPER_PLAN.md 2026-11-04 14:42, sections C2, D1-D3, E demos). Sources: edge-tts s1 clips of the three
Korean voices only. OpenVoice v2 ToneColorConverter, tau 0.3, CPU (same call as P3 convert_gender.py).
Targets from results/tc_targets.npz. Resumable: outputs that already exist and read cleanly are kept (the first ~315
were made on CPU before the CPU pool was found to be overloaded; the rest on CUDA under the GPU lease, PKV_CONV_DEVICE=cuda);
the device of each output is recorded in the manifest. Outputs: outputs/<cond>/<src>__<tag>.wav (lab NAS only; C2 and D2 outputs carry real
speakers' tone colour and are evaluation-only), outputs/manifest.json for eval_runner."""
import json, os, sys, types
from concurrent.futures import ProcessPoolExecutor
import numpy as np, soundfile as sf, librosa
HERE = os.path.dirname(os.path.abspath(__file__)); SIM = os.path.join(HERE, "../../../simulation")
sys.path.append(SIM); import config  # noqa: E402
CK = config.OPENVOICE_CKPT; SR = 16000; OUTD = os.path.join(HERE, "outputs")
DEV = os.environ.get("PKV_CONV_DEVICE", "cpu"); NW = 1 if DEV == "cuda" else 4
_C = None


def init():
    global _C
    import torch; torch.set_num_threads(2)
    sys.modules["openvoice.text"] = types.SimpleNamespace(text_to_sequence=None); sys.path.insert(0, config.OPENVOICE_DIR)
    from openvoice.api import ToneColorConverter, OpenVoiceBaseClass
    c = ToneColorConverter.__new__(ToneColorConverter); OpenVoiceBaseClass.__init__(c, CK + "config.json", device=DEV)
    c.watermark_model = None; c.version = getattr(c.hps, "_version_", "v1"); c.load_ckpt(CK + "checkpoint.pth"); _C = c


def job(task):
    src, items = task          # all conversions of one source clip in one worker (source vector extracted once)
    import torch
    se_src = _C.extract_se([src["path"]]); out = []
    for it in items:
        p = os.path.join(OUTD, it["cond"], f"{src['id']}__{it['tag']}.wav"); meta = {k: v for k, v in it.items() if k not in ("vec", "delta")}
        if os.path.exists(p):
            try:
                if len(sf.read(p)[0]) > SR:
                    out.append(meta | {"file": p, "src_id": src["id"], "src_gender": src["gender"], "text": src["text"], "conv_device": "cpu"}); continue
            except Exception:
                pass
            os.makedirs(os.path.join(HERE, "../superseded/unreadable_outputs"), exist_ok=True); os.rename(p, os.path.join(HERE, "../superseded/unreadable_outputs", os.path.basename(p)))
        sv = se_src.squeeze().cpu().numpy()
        if it["cond"] == "N0":
            tgt = sv
        else:
            tgt = sv + np.asarray(it["delta"]) if "delta" in it else np.asarray(it["vec"])
        y = _C.convert(src["path"], se_src, torch.tensor(tgt, dtype=torch.float32).reshape(1, -1, 1).to(DEV), tau=0.3)
        y = librosa.resample(y.astype(np.float32), orig_sr=_C.hps.data.sampling_rate, target_sr=SR)
        os.makedirs(os.path.dirname(p), exist_ok=True); sf.write(p, y, SR)
        out.append(meta | {"file": p, "src_id": src["id"], "src_gender": src["gender"], "text": src["text"], "conv_device": DEV})
    return out


def main():
    T = np.load(os.path.join(HERE, "results/tc_targets.npz"), allow_pickle=True)
    clips = {c["id"]: c for c in json.load(open(os.path.join(SIM, "conversion_sources/tts_clips.json")))}
    for c in clips.values():   # release: tts_clips.json stores file names; the wavs live in config.TTS_SOURCES
        c["path"] = os.path.join(config.TTS_SOURCES, os.path.basename(c["path"]))
    srcs = [clips[f"tts_ko-KR-{v}Neural_s1"] for v in ("SunHi", "InJoon", "HyunsuMultilingual")]
    G = {"여성": "female", "남성": "male"}
    items = [{"cond": "N0", "tag": "own", "target_gender": None}]
    for k in (4, 16, 64, 256):
        for i, spk in enumerate(T["C2_speakers"]):
            items.append({"cond": f"C2_k{k}", "tag": f"t{i:02d}", "k": k, "target_speaker": str(spk), "target_gender": G[str(T["C2_gender"][i])], "vec": T[f"C2_k{k}"][i].tolist()})
    for i, cell in enumerate(T["D1_cells"]):
        gg, aa = str(cell).split("|")
        items.append({"cond": "D1", "tag": f"cell{i:02d}", "cell": str(cell), "target_gender": G[gg], "cell_n": int(T["D1_n"][i]), "vec": T["D1"][i].tolist()})
    al = [0.25, 0.5, 0.75]
    for i, (a, b) in enumerate(T["D2_pairs"]):
        for j, alpha in enumerate(al):
            items.append({"cond": "D2", "tag": f"p{i:02d}_a{int(alpha * 100)}", "pair": i, "alpha": alpha,
                          "endpoints": [str(s) for s in T["D2_pair_speakers"][i]], "target_gender": None, "vec": T["D2"][3 * i + j].tolist()})
    for i, gg in enumerate(T["D3_gender"]):
        items.append({"cond": "D3", "tag": f"s{i:02d}", "target_gender": G[str(gg)], "vec": T["D3"][i].tolist()})
    V, sd = T["pca_components"], T["pca_sd"]
    for pc in range(8):
        for s in (-2, 2):
            items.append({"cond": "E_pc", "tag": f"pc{pc + 1}_{'m' if s < 0 else 'p'}2", "pc": pc + 1, "sd_step": s, "target_gender": None,
                          "delta": (s * sd[pc] * V[pc]).tolist()})
    print(len(srcs), "sources x", len(items), "items", flush=True)
    rows = [{"file": s["path"], "cond": "SRC", "src_id": s["id"], "src_gender": s["gender"], "text": s["text"], "target_gender": None} for s in srcs]
    tasks = [(s, items[c::4]) for s in srcs for c in range(4)]
    with ProcessPoolExecutor(NW, initializer=init) as ex:
        for res in ex.map(job, tasks):
            rows += res; print("source done", res[0]["src_id"], len(res), flush=True)
    for r in rows:   # eval_runner remeasure needs a target gender for Praat settings: default to the source's gender
        r["target_gender_label"] = r["target_gender"]; r["target_gender"] = {"female": "여성", "male": "남성"}.get(r["target_gender"], r["src_gender"])
    json.dump(rows, open(os.path.join(OUTD, "manifest.json"), "w"), indent=1, ensure_ascii=False)
    print("rows", len(rows))


if __name__ == "__main__":
    main()
