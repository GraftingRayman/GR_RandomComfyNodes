"""
GRLoadAudioImageBatch
----------------------
A ComfyUI node that replicates the behaviour of WAS Node Suite's
"Load Image Batch" node, but for audio files (.wav / .mp3, plus
anything else soundfile/ffmpeg can decode) - and additionally looks
for an image file with the same base filename in the same directory
(e.g. take_01.wav + take_01.png) and returns that image alongside
the audio.

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
import numpy as np
import soundfile as sf
from PIL import Image

try:
    from .gr_logger import GRLogger  # use your existing shared logger if present
    _log = GRLogger("GRLoadAudioImageBatch")
except Exception:
    class _FallbackLogger:
        def info(self, msg): print(f"[GRLoadAudioImageBatch] {msg}")
        def warn(self, msg): print(f"[GRLoadAudioImageBatch][WARN] {msg}")
        def error(self, msg): print(f"[GRLoadAudioImageBatch][ERROR] {msg}")
    _log = _FallbackLogger()

# Where we persist incremental-mode counters between runs/restarts.
_STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gr_audio_image_batch_state")
os.makedirs(_STATE_DIR, exist_ok=True)

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]


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
        raise ValueError(f"GRLoadAudioImageBatch: path does not exist or is not a directory: {path}")
    files = sorted(glob.glob(os.path.join(path, pattern)))
    if not files:
        raise ValueError(f"GRLoadAudioImageBatch: no files matched pattern '{pattern}' in '{path}'")
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
    """Returns (waveform [batch, channels, samples] float32, sample_rate)."""
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


def _find_matching_image(audio_filepath):
    """Look for an image file with the same base name (no extension) in the
    same directory as the audio file. Returns a filepath or None."""
    directory = os.path.dirname(audio_filepath)
    stem = os.path.splitext(os.path.basename(audio_filepath))[0]
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(directory, stem + ext)
        if os.path.exists(candidate):
            return candidate
        # also try uppercase extension, just in case
        candidate_upper = os.path.join(directory, stem + ext.upper())
        if os.path.exists(candidate_upper):
            return candidate_upper
    return None


def _load_image_file(filepath):
    """Returns an IMAGE tensor in ComfyUI format: [batch, H, W, C] float32 0-1."""
    img = Image.open(filepath).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W, C]
    return tensor


def _empty_image():
    """1x1 black placeholder image, used when no matching image is found."""
    return torch.zeros((1, 1, 1, 3), dtype=torch.float32)


class GRLoadAudioImageBatch:
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

    RETURN_TYPES = ("AUDIO", "IMAGE", "STRING", "STRING", "INT", "BOOLEAN")
    RETURN_NAMES = ("audio", "image", "filename_text", "image_filename_text", "index_used", "image_found")
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
            idx = counter % count
            _save_counter(path, pattern, label, counter + 1)

        filepath = files[idx]
        _log.info(f"Loading [{idx + 1}/{count}] {os.path.basename(filepath)} (mode={mode})")

        waveform, sr = _load_audio_file(filepath)
        audio_out = {"waveform": waveform, "sample_rate": sr}

        image_path = _find_matching_image(filepath)
        if image_path:
            _log.info(f"Found matching image: {os.path.basename(image_path)}")
            image_tensor = _load_image_file(image_path)
            image_filename = os.path.basename(image_path)
            image_found = True
        else:
            image_tensor = _empty_image()
            image_filename = ""
            image_found = False

        return (audio_out, image_tensor, os.path.basename(filepath), image_filename, idx, image_found)


NODE_CLASS_MAPPINGS = {
    "GRLoadAudioImageBatch": GRLoadAudioImageBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRLoadAudioImageBatch": "GR Load Audio Image Batch",
}
