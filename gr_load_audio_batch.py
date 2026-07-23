"""
GRLoadAudioBatch
----------------
A ComfyUI node that replicates the behaviour of WAS Node Suite's
"Load Image Batch" node, but for audio files (.wav / .mp3, plus
anything else soundfile/ffmpeg can decode).

Modes:
  single_audio       - always returns the file at `index`
  incremental_audio  - advances one file per queue run (state persisted
                        to disk, keyed by path+pattern+label, same as
                        WAS's counter behaviour)
  random             - picks a file using `seed`

Drop this file into your ComfyUI custom_nodes package (e.g. alongside
your other GR nodes) and make sure it's picked up by your package's
NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS aggregation.
"""

import os
import glob
import json
import random
import subprocess
import tempfile

import torch
import soundfile as sf

try:
    from .gr_logger import GRLogger  # use your existing shared logger if present
    _log = GRLogger("GRLoadAudioBatch")
except Exception:
    class _FallbackLogger:
        def info(self, msg): print(f"[GRLoadAudioBatch] {msg}")
        def warn(self, msg): print(f"[GRLoadAudioBatch][WARN] {msg}")
        def error(self, msg): print(f"[GRLoadAudioBatch][ERROR] {msg}")
    _log = _FallbackLogger()

# Where we persist incremental-mode counters between runs/restarts.
_STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gr_audio_batch_state")
os.makedirs(_STATE_DIR, exist_ok=True)


def _state_key(path, pattern, label):
    raw = f"{os.path.abspath(path)}|{pattern}|{label}"
    return str(abs(hash(raw)))


def _state_file(path, pattern, label):
    return os.path.join(_STATE_DIR, f"{_state_key(path, pattern, label)}.json")


def _load_counter(path, pattern, label):
    fp = _state_file(path, pattern, label)
    if os.path.exists(fp):
        try:
            with open(fp, "r") as f:
                data = json.load(f)
            return data.get("counter", 0)
        except Exception:
            return 0
    return 0


def _save_counter(path, pattern, label, counter):
    fp = _state_file(path, pattern, label)
    with open(fp, "w") as f:
        json.dump({"counter": counter}, f)


def _list_audio_files(path, pattern):
    if not os.path.isdir(path):
        raise ValueError(f"GRLoadAudioBatch: path does not exist or is not a directory: {path}")
    files = sorted(glob.glob(os.path.join(path, pattern)))
    if not files:
        raise ValueError(f"GRLoadAudioBatch: no files matched pattern '{pattern}' in '{path}'")
    return files


def _decode_with_ffmpeg(filepath):
    """Fallback decoder for formats libsndfile can't read directly (notably mp3
    on many builds). Transcodes to a temp wav, then loads that with soundfile."""
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()
    try:
        cmd = [
            "ffmpeg", "-y", "-i", filepath,
            "-ar", "44100", "-ac", "2",
            tmp_wav.name,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="ignore"))
        waveform, sr = sf.read(tmp_wav.name, always_2d=True, dtype="float32")
        return waveform, sr
    finally:
        try:
            os.remove(tmp_wav.name)
        except OSError:
            pass


def _load_audio_file(filepath):
    """Returns (waveform [channels, samples] float32, sample_rate)."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".mp3":
        # libsndfile generally can't read mp3 - go straight to ffmpeg.
        waveform, sr = _decode_with_ffmpeg(filepath)
    else:
        try:
            waveform, sr = sf.read(filepath, always_2d=True, dtype="float32")
        except Exception as e:
            _log.warn(f"soundfile failed on {filepath} ({e}), falling back to ffmpeg")
            waveform, sr = _decode_with_ffmpeg(filepath)

    # soundfile gives [samples, channels] - ComfyUI AUDIO wants [channels, samples]
    waveform = waveform.T
    tensor = torch.from_numpy(waveform).unsqueeze(0)  # [batch, channels, samples]
    return tensor, sr


class GRLoadAudioBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "", "multiline": False}),
                "pattern": ("STRING", {"default": "*.wav"}),
                "mode": (["single_audio", "incremental_audio", "random"], {"default": "incremental_audio"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "label": ("STRING", {"default": "Batch 001"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", "INT")
    RETURN_NAMES = ("audio", "filename_text", "index_used")
    FUNCTION = "load_batch"
    CATEGORY = "GraftingRayman/Audio"

    def load_batch(self, path, pattern, mode, index, seed, label):
        files = _list_audio_files(path, pattern)
        count = len(files)

        if mode == "single_audio":
            idx = index % count

        elif mode == "random":
            rng = random.Random(seed)
            idx = rng.randrange(count)

        else:  # incremental_audio
            counter = _load_counter(path, pattern, label)
            # reset if the folder contents changed enough that the old counter
            # no longer makes sense (mirrors WAS resetting on path/pattern change,
            # which is implicitly handled here since the key is path+pattern+label)
            idx = counter % count
            _save_counter(path, pattern, label, counter + 1)

        filepath = files[idx]
        _log.info(f"Loading [{idx + 1}/{count}] {os.path.basename(filepath)} (mode={mode})")

        waveform, sr = _load_audio_file(filepath)
        audio_out = {"waveform": waveform, "sample_rate": sr}

        return (audio_out, os.path.basename(filepath), idx)


NODE_CLASS_MAPPINGS = {
    "GRLoadAudioBatch": GRLoadAudioBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRLoadAudioBatch": "GR Load Audio Batch",
}
