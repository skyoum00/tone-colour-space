"""Generate the stock-TTS source set (P2 E2.2 / P3 source (i)): every ko-KR edge-tts voice x 5 fixed everyday sentences.

Only the sentence TEXT is sent to the Microsoft edge-tts service; no AI Hub audio leaves the NAS.
Output: tts_sources/<voice>_s<k>.wav (16 kHz mono PCM16) + tts_sources/sources.json.
Usage: ~/venvs/pkv/bin/python make_tts_sources.py
"""
import asyncio, io, json, os, subprocess, sys
import numpy as np, soundfile as sf, librosa
import edge_tts

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tts_sources")
# Fixed everyday sentences, standard Korean, no dialect, no names, no numerals (keeps CER references unambiguous).
SENTENCES = [
    "오늘은 날씨가 맑아서 오후에 공원을 천천히 걸어 볼 생각이에요.",
    "내일 아침에는 조금 일찍 일어나서 집 앞 시장에 다녀와야 해요.",
    "요즘 저녁마다 따뜻한 차를 마시면서 책을 읽는 게 제일 좋아요.",
    "버스가 생각보다 늦게 와서 약속 시간에 조금 늦을 것 같아요.",
    "주말에는 가족들과 함께 집에서 맛있는 음식을 만들어 먹었어요.",
]


async def synth(voice, text):
    buf = b""
    async for ch in edge_tts.Communicate(text, voice).stream():
        if ch["type"] == "audio":
            buf += ch["data"]
    return buf


async def main():
    os.makedirs(OUT, exist_ok=True)
    voices = [v for v in await edge_tts.list_voices() if v["Locale"] == "ko-KR"]
    rows = []
    for v in sorted(voices, key=lambda v: v["ShortName"]):
        for k, text in enumerate(SENTENCES, 1):
            mp3 = await synth(v["ShortName"], text)
            y, sr = librosa.load(io.BytesIO(mp3), sr=None, mono=True)   # edge-tts returns 24 kHz mono mp3
            y16 = librosa.resample(y, orig_sr=sr, target_sr=16000, res_type="soxr_hq")
            name = f'{v["ShortName"]}_s{k}.wav'
            sf.write(os.path.join(OUT, name), y16, 16000, subtype="PCM_16")
            rows.append({"file": name, "voice": v["ShortName"], "gender": v["Gender"], "sentence_id": k, "text": text,
                         "duration_s": round(len(y16) / 16000, 3), "orig_sr": int(sr)})
            print(name, rows[-1]["duration_s"], flush=True)
    meta = {"generator": f"edge-tts {edge_tts.__version__}", "date": "2026-11-03", "sample_rate": 16000, "channels": 1,
            "n_voices": len(voices), "voices": [{"name": v["ShortName"], "gender": v["Gender"]} for v in voices],
            "note": "all ko-KR voices listed by edge-tts at generation time; text-only requests to the edge-tts service",
            "sources": rows}
    json.dump(meta, open(os.path.join(OUT, "sources.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    asyncio.run(main())
