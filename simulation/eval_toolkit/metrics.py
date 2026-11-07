"""Local speech-evaluation metrics for P2 (parametric VC) and P3 (gender VC). Every function takes a wav path (any sample
rate, any channel count; mixed down to mono and resampled to 16 kHz inside) and returns plain numbers.
All models run locally; nothing is sent to any external service. Model names, revisions and licences: MODELS.md.

Device: env PKV_EVAL_DEVICE ("cpu" default, or "cuda"). ECAPA always runs on CPU through stage1/measure_voice.init_worker,
so it is byte-for-byte the study's embedding path. Take the GPU lease before setting PKV_EVAL_DEVICE=cuda.
"""
import os, re, sys, unicodedata
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE1 = os.path.normpath(os.path.join(HERE, "..", "stage1"))
if STAGE1 not in sys.path:
    sys.path.insert(0, STAGE1)
if os.path.dirname(STAGE1) not in sys.path:
    sys.path.append(os.path.dirname(STAGE1))   # simulation/ (config.py)
import config  # noqa: E402
MODELS = config.MODEL_CACHE
os.environ.setdefault("TORCH_HOME", os.path.join(MODELS, "torch_hub"))
HF_CACHE = os.path.join(MODELS, "hf", "hub")
WHISPER_CACHE = config.WHISPER_CACHE   # large-v3 was already cached here (rev 06f233fe)
SR = 16000

REV = {"wavlm": "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e",
       "audeering": "bad2737f529168bc3d3fcd97edfaba9583c7d36d",
       "whisper": "06f233fe06e710322aca913c1bc4249a0d71fce1"}
_M = {}


def device():
    return os.environ.get("PKV_EVAL_DEVICE", "cpu")


def load(path, sr=SR):
    """wav -> mono float32 at `sr` (soxr_hq resampling, as librosa default)."""
    import soundfile as sf, librosa
    try:
        y, s = sf.read(path, dtype="float32", always_2d=True)
        y = y.mean(1)
    except Exception:
        y, s = librosa.load(path, sr=None, mono=True)
    if s != sr:
        y = librosa.resample(y, orig_sr=s, target_sr=sr, res_type="soxr_hq")
    return np.ascontiguousarray(y, dtype=np.float32)


# ---------------------------------------------------------------- 1. ECAPA (study-identical)
def _ecapa():
    if "ecapa" not in _M:
        import torch, measure_voice as mv
        n = torch.get_num_threads()
        mv.init_worker()                 # loads speechbrain/spkrec-ecapa-voxceleb from ~/.cache/pkv_models/ecapa (+ openSMILE)
        torch.set_num_threads(n)         # init_worker pins torch to 1 thread (worker-pool setting); restore for the other models
        _M["ecapa"] = mv
    return _M["ecapa"]


def asv_ecapa(path):
    """192-d L2-normalised ECAPA-TDNN embedding (stage1 measure_voice.embed on the whole file as one segment)."""
    mv = _ecapa()
    e = mv.embed([load(path)])[0]
    return e / np.linalg.norm(e)


# ---------------------------------------------------------------- 2. WavLM-SV
def _wavlm():
    if "wavlm" not in _M:
        from transformers import AutoFeatureExtractor, WavLMForXVector
        r = "microsoft/wavlm-base-plus-sv"
        fe = AutoFeatureExtractor.from_pretrained(r, revision=REV["wavlm"], cache_dir=HF_CACHE)
        m = WavLMForXVector.from_pretrained(r, revision=REV["wavlm"], cache_dir=HF_CACHE).eval().to(device())
        _M["wavlm"] = (fe, m)
    return _M["wavlm"]


def asv_wavlm(path):
    """512-d L2-normalised x-vector from microsoft/wavlm-base-plus-sv (independent second ASV)."""
    import torch
    fe, m = _wavlm()
    x = fe(load(path), sampling_rate=SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        e = m(**{k: v.to(device()) for k, v in x.items()}).embeddings[0].float().cpu().numpy()
    return e / np.linalg.norm(e)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------- 3. UTMOS
def _utmos():
    if "utmos" not in _M:
        import torch
        _M["utmos"] = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval().to(device())
    return _M["utmos"]


def utmos(path):
    """Predicted naturalness MOS (1-5 scale), UTMOS22 strong learner via SpeechMOS v1.2.0."""
    import torch
    with torch.no_grad():
        return float(_utmos()(torch.from_numpy(load(path))[None].to(device()), SR).item())


# ---------------------------------------------------------------- 4. Whisper CER
_PUNCT = re.compile(r"\s+")


def normalize_ko(text):
    """CER text normalisation: Unicode NFKC, lower-case, then drop every whitespace character and every character whose
    Unicode category is punctuation (P*) or symbol (S*). Hangul syllables, jamo, digits and Latin letters are kept as is
    (no number verbalisation, no spelling normalisation)."""
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if not c.isspace() and unicodedata.category(c)[0] not in "PS")


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _whisper():
    if "whisper" not in _M:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        r = "openai/whisper-large-v3"
        dt = torch.float16 if device() == "cuda" else torch.float32
        p = WhisperProcessor.from_pretrained(r, revision=REV["whisper"], cache_dir=WHISPER_CACHE)
        m = WhisperForConditionalGeneration.from_pretrained(r, revision=REV["whisper"], cache_dir=WHISPER_CACHE,
                                                            dtype=dt).eval().to(device())
        _M["whisper"] = (p, m, dt)
    return _M["whisper"]


def transcribe(path):
    """Whisper large-v3, language forced to Korean, task transcribe, greedy decoding (num_beams=1, no sampling).
    Audio longer than 30 s is decoded with Whisper's sequential long-form algorithm."""
    import torch
    p, m, dt = _whisper()
    y = load(path)
    long = len(y) > 30 * SR
    x = p(y, sampling_rate=SR, return_tensors="pt", truncation=not long, padding="longest" if long else "max_length",
          return_attention_mask=True)
    feats = x.input_features.to(device(), dt)
    kw = dict(language="ko", task="transcribe", num_beams=1, do_sample=False)
    if long:
        kw.update(attention_mask=x.attention_mask.to(device()), return_timestamps=True)
    with torch.no_grad():
        ids = m.generate(feats, **kw)
    return p.batch_decode(ids, skip_special_tokens=True)[0].strip()


def cer(path, ref_text):
    """Korean character error rate = Levenshtein(norm(ref), norm(hyp)) / len(norm(ref)), characters = Hangul syllables
    after normalize_ko. Returns {"cer", "edits", "ref_len", "hyp"}."""
    hyp = transcribe(path)
    r, h = normalize_ko(ref_text), normalize_ko(hyp)
    d = _lev(r, h)
    return {"cer": d / max(len(r), 1), "edits": d, "ref_len": len(r), "hyp": hyp}


# ---------------------------------------------------------------- 5. AASIST anti-spoofing
AASIST_DIR = os.path.join(MODELS, "aasist_src", "aasist")
AASIST_NSAMP = 64600


def _aasist():
    if "aasist" not in _M:
        import importlib.util, json, torch
        spec = importlib.util.spec_from_file_location("aasist_model", os.path.join(AASIST_DIR, "models", "AASIST.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        cfg = json.load(open(os.path.join(AASIST_DIR, "config", "AASIST.conf")))["model_config"]
        m = mod.Model(cfg)
        m.load_state_dict(torch.load(os.path.join(AASIST_DIR, "models", "weights", "AASIST.pth"), map_location="cpu"))
        _M["aasist"] = m.eval().to(device())
    return _M["aasist"]


def spoof(path):
    """AASIST (clovaai, ASVspoof 2019 LA weights) countermeasure score. Score direction: HIGHER = MORE BONA FIDE
    (score = output logit of class 1 = bona fide, the official eval score in clovaai/aasist main.py).
    The official recipe scores the first 64,600 samples (4.04 s; shorter input tile-repeated, data_utils.pad). Here:
    "score" = mean of the official score over consecutive non-overlapping 64,600-sample windows (last partial window
    dropped if another full window exists; a file shorter than one window is tile-padded as in the recipe),
    "score_first" = the official first-window score, "p_bonafide" = mean softmax probability of class 1."""
    import torch
    y = load(path)
    if len(y) < AASIST_NSAMP:
        y = np.tile(y, AASIST_NSAMP // len(y) + 1)[:AASIST_NSAMP]
    n = len(y) // AASIST_NSAMP
    w = torch.from_numpy(y[:n * AASIST_NSAMP].reshape(n, AASIST_NSAMP)).to(device())
    with torch.no_grad():
        _, out = _aasist()(w)
    out = out.float().cpu()
    s = out[:, 1].numpy()
    return {"score": float(s.mean()), "score_first": float(s[0]), "p_bonafide": float(torch.softmax(out, 1)[:, 1].mean()),
            "n_windows": int(n)}


# ---------------------------------------------------------------- 6. audeering age-gender
def _audeering():
    if "aud" not in _M:
        import torch.nn as nn
        from transformers import AutoFeatureExtractor
        from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

        class ModelHead(nn.Module):            # verbatim from the model card
            def __init__(self, config, num_labels):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, num_labels)

            def forward(self, x):
                return self.out_proj(self.dropout(__import__("torch").tanh(self.dense(self.dropout(x)))))

        class AgeGenderModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.config = config
                self.wav2vec2 = Wav2Vec2Model(config)
                self.age = ModelHead(config, 1)
                self.gender = ModelHead(config, 3)
                self.post_init()

            def forward(self, input_values):
                import torch
                h = self.wav2vec2(input_values)[0].mean(1)
                return h, self.age(h), torch.softmax(self.gender(h), dim=1)

        r = "audeering/wav2vec2-large-robust-24-ft-age-gender"
        fe = AutoFeatureExtractor.from_pretrained(r, revision=REV["audeering"], cache_dir=HF_CACHE)
        m = AgeGenderModel.from_pretrained(r, revision=REV["audeering"], cache_dir=HF_CACHE).eval().to(device())
        _M["aud"] = (fe, m)
    return _M["aud"]


def gender_audeering(path):
    """audeering wav2vec2-large-robust-24-ft-age-gender. Returns the three-way softmax {p_female, p_male, p_child},
    p_female_adult = p_female / (p_female + p_male), and age_years = 100 x age output (model card: age 0..1 = 0..100 y)."""
    import torch
    fe, m = _audeering()
    x = fe(load(path), sampling_rate=SR, return_tensors="pt").input_values.to(device())
    with torch.no_grad():
        _, age, g = m(x)
    g = g[0].float().cpu().numpy()
    return {"p_female": float(g[0]), "p_male": float(g[1]), "p_child": float(g[2]),
            "p_female_adult": float(g[0] / (g[0] + g[1])), "age_years": float(age[0, 0]) * 100}


# ---------------------------------------------------------------- 7. balanced ECAPA gender meter
METER = os.path.join(HERE, "gender_meter_balanced.joblib")


def gender_meter(path):
    """P(female) from the balanced logistic regression on study-identical ECAPA embeddings (train_gender_meter.py)."""
    if "meter" not in _M:
        import joblib
        _M["meter"] = joblib.load(METER)
    return float(_M["meter"].predict_proba(asv_ecapa(path)[None])[0, 1])


# ---------------------------------------------------------------- 8. study-identical re-measurement
_G = {"남성": "남성", "male": "남성", "m": "남성", "여성": "여성", "female": "여성", "f": "여성"}


def remeasure(path, gender):
    """Re-measure the study variables on the whole file treated as ONE segment, with the stage-1 functions and settings
    (Praat pitch floor/ceiling 60-300 Hz male, 100-500 Hz female; Burg formant ceiling 5000/5500 Hz; 16 kHz).
    `gender` selects those settings ("male"/"female" or "남성"/"여성") and should be the gender the voice is meant to have.
    Returns the merged dicts of measure_voice.praat_measures, measure_voice.envelope_measures (WORLD env3oct_* 1/3-octave
    bands in dB re envelope mean, envERB*, MFCC/LFCC) and harmonic_measures.voice_quality_measures. Headline keys:
    f0_median_hz, f0_range_st (12 log2 p95/p5), f0_p5_hz, f0_p95_hz, F1_hz..F4_hz (medians over voiced frames), env3oct_<fc>hz_db."""
    import measure_voice as mv
    from harmonic_measures import voice_quality_measures
    g = _G[str(gender).strip().lower()] if str(gender).strip().lower() in _G else _G[gender]
    y = load(path)
    out = {}
    out.update(mv.praat_measures([y], g))
    out.update(mv.envelope_measures([y], g))
    out.update(voice_quality_measures([y], g))
    return out
