"""Shared evaluation runner for P2/P3 conversion outputs (local models only, eval_toolkit).

Input manifest (json list): {"file", "cond", "src_id", "src_gender", "target_gender", "text", ...}.
--part gpu       : ECAPA + WavLM embeddings (saved as .npy per row index), gender_meter, gender_audeering, UTMOS, CER, spoof.
                   Takes the shared GPU lease (one job), released at exit.
--part remeasure : Praat/WORLD/opensauce re-measurement with the TARGET gender's settings, CPU pool.
Outputs (resumable jsonl, keyed by file): <out>_gpu.jsonl, <out>_remeasure.jsonl, <out>_emb.npz (written at end of gpu part)."""
import argparse, atexit, json, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "stage1"))
import config  # noqa: E402  (paths; see config.py)
# Optional lab GPU-lease helper (env PKV_GPU_LEASE = path of a script taking "acquire <job> <pid> <supervisor>" / "release <job> <pid>");
# unset = no lease, the evaluation runs directly.
LEASE = os.environ.get("PKV_GPU_LEASE", ""); SUPERVISOR = os.environ.get("PKV_GPU_LEASE_SUPERVISOR", "")


def done_keys(p):
    return {json.loads(l)["file"] for l in open(p)} if os.path.exists(p) else set()


def gpu(rows, out):
    name = "pkv-eval-" + os.path.basename(out)
    if LEASE:
        subprocess.run([LEASE, "acquire", name, str(os.getpid()), SUPERVISOR], check=True)
        atexit.register(lambda: subprocess.run([LEASE, "release", name, str(os.getpid())]))
    from eval_toolkit import metrics as M
    import torch, joblib
    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=config.ECAPA_DIR,
                                           run_opts={"device": "cuda"})   # same weights as the study path, on GPU
    meter = joblib.load(M.METER)

    def asv_ecapa_gpu(path):
        with torch.no_grad():
            e = ecapa.encode_batch(torch.tensor(M.load(path), dtype=torch.float32, device="cuda")[None]).squeeze().cpu().numpy()
        return e / np.linalg.norm(e)
    p = out + "_gpu.jsonl"; dk = done_keys(p); embp = out + "_emb"; os.makedirs(embp, exist_ok=True)
    with open(p, "a") as f:
        for i, r in enumerate(rows):
            if r["file"] in dk: continue
            try:
                e1, e2 = asv_ecapa_gpu(r["file"]), M.asv_wavlm(r["file"])
                key = os.path.relpath(r["file"], "/").replace("/", "__")[:-4]
                np.save(os.path.join(embp, key + "_ecapa.npy"), e1); np.save(os.path.join(embp, key + "_wavlm.npy"), e2)
                ga = M.gender_audeering(r["file"]); sp = M.spoof(r["file"])
                res = {"file": r["file"], "emb_key": key, "p_female_meter": float(meter.predict_proba(e1[None])[0, 1]),
                       "p_female_audeering": float(ga["p_female_adult"]), "p_male_audeering": float(ga["p_male"]),
                       "utmos": float(M.utmos(r["file"])), "spoof": float(sp["score"] if isinstance(sp, dict) else sp)}
                if r.get("text"):
                    c = M.cer(r["file"], r["text"]); res["cer"] = float(c["cer"] if isinstance(c, dict) else c)
            except Exception as ex:
                res = {"file": r["file"], "error": f"{type(ex).__name__}: {ex}"}
            f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
            if i % 200 == 0: print("gpu", i, len(rows), flush=True)


def _rm(r):
    from eval_toolkit import metrics as M
    try:
        v = M.remeasure(r["file"], r["target_gender"]); return {"file": r["file"], **{k: v[k] for k in v if np.isscalar(v[k])}}
    except Exception as ex:
        return {"file": r["file"], "error": f"{type(ex).__name__}: {ex}"}


def remeasure(rows, out, workers):
    from concurrent.futures import ProcessPoolExecutor
    p = out + "_remeasure.jsonl"; dk = done_keys(p); todo = [r for r in rows if r["file"] not in dk]
    with ProcessPoolExecutor(workers) as ex, open(p, "a") as f:
        for i, res in enumerate(ex.map(_rm, todo, chunksize=4)):
            f.write(json.dumps(res, ensure_ascii=False, default=float) + "\n"); f.flush()
            if i % 200 == 0: print("remeasure", i, len(todo), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--part", choices=["gpu", "remeasure"], required=True); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); rows = json.load(open(a.manifest))
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    gpu(rows, a.out) if a.part == "gpu" else remeasure(rows, a.out, a.workers)
