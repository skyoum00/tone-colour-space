"""A6 data-integrity table (PREREG.md, Amendment A6): for every eligible speaker key, the length of the conversation wav
in the AI Hub zip against the label FileLength. common_data.load("primary"), analyze_h1.py and predict_tone_colour.py keep
the keys with 0.99 <= wav_over_label <= 1.01 (the primary sample, 2,410 keys).

Release note: in the study this table was produced by an ad-hoc command that was not saved as a script. This script was
written for the release from the A6 description and the columns of the study table, and was checked against the study
table on the full corpus: every column identical (max abs difference 2.2e-16) and the same 2,410 primary keys.
wav_s assumes the corpus wavs' canonical 44-byte PCM header (16 kHz, 2 channels, 16 bit), as in the study table.
Reads zip directory entries and label JSON only (no audio is decoded).
Output: results/audio_label_duration_check.csv (columns speaker, file, wav_s, last_end, ratio, label_len, wav_over_label).
"""
import glob, json, os, zipfile
import pandas as pd
import measure_voice as mv
import build_speaker_index as b


def main():
    zm = mv.zip_map()
    lab = {}
    for zp in glob.glob(os.path.join(b.RAW, "*/*/*/02.라벨링데이터/*L_01.실내.zip")):
        for n in zipfile.ZipFile(zp).namelist():
            if n.endswith(".json"):
                lab[os.path.basename(n)[:-5]] = (zp, n)
    idx = pd.read_parquet(os.path.join(mv.OUT, "speaker_index.parquet"))
    el = idx[idx.eligible].sort_values("speaker")
    zc, rows = {}, []
    for r in el.itertuples():
        zp, m = zm[r.file]
        size = zc.setdefault(zp, zipfile.ZipFile(zp)).getinfo(m).file_size
        lz, ln = lab[r.file]
        label_len = float(json.loads(zc.setdefault(lz, zipfile.ZipFile(lz)).read(ln))["File"]["FileLength"])
        wav_s = (size - 44) / (mv.SR * 2 * 2)
        last_end = max(s["end"] for s in json.loads(r.segments))
        rows.append({"speaker": r.speaker, "file": r.file, "wav_s": wav_s, "last_end": last_end, "ratio": last_end / wav_s,
                     "label_len": label_len, "wav_over_label": wav_s / label_len})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(mv.OUT, "audio_label_duration_check.csv"), index=False)
    ok = (out.wav_over_label >= 0.99) & (out.wav_over_label <= 1.01)
    print(f"eligible {len(out)}, wav within 1% of label length {int(ok.sum())}")


if __name__ == "__main__":
    main()
