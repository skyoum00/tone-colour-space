"""Paths used by the tone-colour-space code.

Every path can be overridden with an environment variable. Defaults are relative to this repository or to the user's
home directory. Nothing else in the code holds an absolute path.

    PKV_DATA_ROOT        AI Hub 71631 download root: the directory that contains the dataset folder, under which the
                         code globs */*/*/01.<source-data>/*.zip (audio) and */*/*/02.<label-data>/*L_01.<indoor>.zip
                         (labels), exactly as the AI Hub download tool lays them out.
                         Default: <repo>/data/raw/freetalk71631
    PKV_STAGE1_RESULTS   stage-1 outputs (speaker index, per-speaker measurements, tone-colour vectors, h1/A7 results).
                         Default: <repo>/simulation/stage1/results
    PKV_STAGE1_FIGURES   figures written by the stage-1 scripts.            Default: <repo>/paper_figures
    PKV_MODEL_CACHE      model cache root.                                  Default: ~/.cache/pkv_models
    PKV_OPENVOICE_DIR    clone of github.com/myshell-ai/OpenVoice.          Default: ~/.local/share/OpenVoice
    PKV_OPENVOICE_CKPT   OpenVoice v2 converter checkpoint directory (config.json + checkpoint.pth).
                         Default: $PKV_MODEL_CACHE/openvoice_v2/converter
    PKV_ECAPA_DIR        SpeechBrain savedir for spkrec-ecapa-voxceleb.     Default: $PKV_MODEL_CACHE/ecapa
    PKV_OPENSAUCE_DIR    clone of github.com/voicesauce/opensauce-python.   Default: ~/.local/share/opensauce-python
    PKV_WHISPER_CACHE    Hugging Face hub cache holding whisper-large-v3.   Default: ~/.cache/huggingface/hub
    PKV_TTS_SOURCES      stock-TTS wavs made by eval_toolkit/make_tts_sources.py.
                         Default: <repo>/simulation/eval_toolkit/tts_sources
"""
import os

SIM = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SIM)


def _p(var, default):
    return os.path.abspath(os.path.expanduser(os.environ.get(var, default)))


DATA_ROOT = _p("PKV_DATA_ROOT", os.path.join(REPO, "data", "raw", "freetalk71631"))
STAGE1_RESULTS = _p("PKV_STAGE1_RESULTS", os.path.join(SIM, "stage1", "results"))
STAGE1_FIGURES = _p("PKV_STAGE1_FIGURES", os.path.join(REPO, "paper_figures"))
MODEL_CACHE = _p("PKV_MODEL_CACHE", "~/.cache/pkv_models")
OPENVOICE_DIR = _p("PKV_OPENVOICE_DIR", "~/.local/share/OpenVoice")
OPENVOICE_CKPT = _p("PKV_OPENVOICE_CKPT", os.path.join(MODEL_CACHE, "openvoice_v2", "converter")) + os.sep  # used as CK + "config.json"
ECAPA_DIR = _p("PKV_ECAPA_DIR", os.path.join(MODEL_CACHE, "ecapa"))
OPENSAUCE_DIR = _p("PKV_OPENSAUCE_DIR", "~/.local/share/opensauce-python")
WHISPER_CACHE = _p("PKV_WHISPER_CACHE", "~/.cache/huggingface/hub")
TTS_SOURCES = _p("PKV_TTS_SOURCES", os.path.join(SIM, "eval_toolkit", "tts_sources"))
