"""Build the Stage-1 speaker index from AI Hub 71631 labels (PREREG.md, fixed 2026-11-02).

Indoor recordings only. Per speaker (ID + gender + age band) take the conversation with the highest
labelled SNR; keep the speaker's utterances of 1.0-15 s that do not overlap the other speaker (0.2 s guard);
include the speaker if SNR >= 20 dB and the kept speech is >= 60 s. Keeps up to 120 s of segments in time order.
Output: results/speaker_index.parquet (+ .json summary).
"""
import glob, json, os, re, zipfile
import pandas as pd

import sys as _sys
_SIMDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # simulation/ (config.py)
if _SIMDIR not in _sys.path:
    _sys.path.append(_SIMDIR)
import config  # noqa: E402
RAW = config.DATA_ROOT
OUT = config.STAGE1_RESULTS
SNR_MIN, MIN_SPEECH, MAX_SPEECH, GUARD = 20, 60.0, 120.0, 0.2
HANGUL = re.compile(r"[가-힣]")


def snr_db(v):
    m = re.match(r"([+-]?\d+)dB", v or "")
    return int(m.group(1)) if m else None


def t(v):
    return float(str(v).replace(",", ""))


def own_segments(conv, me, other):
    mine = [u for u in conv if u["SpeakerNo"] == me]
    theirs = [(t(u["StartTime"]), t(u["EndTime"])) for u in conv if u["SpeakerNo"] == other]
    keep = []
    for u in mine:
        s, e = t(u["StartTime"]), t(u["EndTime"])
        if not 1.0 <= e - s <= 15.0:
            continue
        if any(s < te + GUARD and ts - GUARD < e for ts, te in theirs):
            continue
        keep.append({"start": s, "end": e, "syllables": len(HANGUL.findall(u.get("Text", "")))})
    return sorted(keep, key=lambda x: x["start"])


def main():
    os.makedirs(OUT, exist_ok=True)
    cands = {}
    for zp in sorted(glob.glob(os.path.join(RAW, "*/*/*/02.라벨링데이터/*L_01.실내.zip"))):
        split = "Validation" if "/Validation/" in zp else "Training"
        z = zipfile.ZipFile(zp)
        for n in z.namelist():
            if not n.endswith(".json"):
                continue
            d = json.loads(z.read(n))
            for me, other in (("Speaker1", "Speaker2"), ("Speaker2", "Speaker1")):
                sp = d[me]
                key = f'{sp["ID"]}_{sp["Gender"]}_{sp["Age"]}'
                snr = snr_db(d["Noise"].get(f"{me}SNR"))
                if snr is None:
                    continue
                segs = own_segments(d["Conversation"], me, other)
                rec = {"speaker": key, "file": d["File"]["FileName"], "split": split, "role": me, "snr_db": snr,
                       "gender": sp["Gender"], "age": sp["Age"], "device": sp["RecDevice"], "residence": sp.get("Residence"),
                       "sr": int(d["Wav"]["SamplingRate"]), "channels": int(d["Wav"]["NumberOfChannel"]),
                       "n_segments_all": len(segs), "speech_all_s": round(sum(s["end"] - s["start"] for s in segs), 2),
                       "segments": segs}
                best = cands.get(key)
                if best is None or (snr, rec["speech_all_s"]) > (best["snr_db"], best["speech_all_s"]):
                    cands[key] = rec
    rows = []
    for rec in cands.values():
        kept, tot = [], 0.0
        for s in rec["segments"]:
            if tot >= MAX_SPEECH:
                break
            kept.append(s); tot += s["end"] - s["start"]
        rec = dict(rec, segments=json.dumps(kept), speech_s=round(tot, 2),
                   eligible=bool(rec["snr_db"] >= SNR_MIN and rec["speech_all_s"] >= MIN_SPEECH))
        rows.append(rec)
    df = pd.DataFrame(rows).sort_values("speaker").reset_index(drop=True)
    df.to_parquet(os.path.join(OUT, "speaker_index.parquet"))
    el = df[df.eligible]
    summ = {"speakers_indoor": len(df), "snr_ge_20": int((df.snr_db >= SNR_MIN).sum()),
            "eligible": len(el), "eligible_by_gender": el.gender.value_counts().to_dict(),
            "eligible_by_age": el.age.value_counts().to_dict(), "eligible_by_device": el.device.value_counts().to_dict(),
            "eligible_by_split": el.split.value_counts().to_dict(), "eligible_files": int(el.file.nunique()),
            "speech_s_median_eligible": float(el.speech_s.median()) if len(el) else None,
            "rule": "indoor; best-SNR conversation per speaker; SNR>=20 dB; own non-overlapping 1-15 s segments >= 60 s; cap 120 s"}
    json.dump(summ, open(os.path.join(OUT, "speaker_index_summary.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
