"""
GRLoadTextBatchTest
----------------
A ComfyUI node that replicates the behaviour of WAS Node Suite's
"Load Image Batch" node, but for text files (.txt, or any pattern
you point it at).

Modes:
  single_text       - always returns the file at `index`
  incremental_text   - advances one file per queue run (state persisted
                        to disk, keyed by path+pattern+label, same as
                        WAS's counter behaviour)
  random             - picks a file using `seed`

Also exposes `remaining_count` and `batch_complete`, meant to be wired
into Impact Pack's `ImpactQueueTrigger` node the same way as the audio
loader - remaining_count hits 0 on the last file and stays there, and
incremental_text mode clamps on the last file instead of wrapping back
to file 0, so the one unavoidable extra trigger-tick is a harmless
repeat rather than a restart of the batch.

Drop this file into your ComfyUI custom_nodes package (e.g. alongside
your other GR nodes) and make sure it's picked up by your package's
NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS aggregation.
"""

import os
import glob
import json
import random

try:
    from .gr_logger import GRLogger  # use your existing shared logger if present
    _log = GRLogger("GRLoadTextBatchTest")
except Exception:
    class _FallbackLogger:
        def info(self, msg): print(f"[GRLoadTextBatchTest] {msg}")
        def warn(self, msg): print(f"[GRLoadTextBatchTest][WARN] {msg}")
        def error(self, msg): print(f"[GRLoadTextBatchTest][ERROR] {msg}")
    _log = _FallbackLogger()

# Where we persist incremental-mode counters between runs/restarts.
_STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gr_text_batch_test_state")
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


def _list_text_files(path, pattern):
    if not os.path.isdir(path):
        raise ValueError(f"GRLoadTextBatchTest: path does not exist or is not a directory: {path}")
    files = sorted(glob.glob(os.path.join(path, pattern)))
    if not files:
        raise ValueError(f"GRLoadTextBatchTest: no files matched pattern '{pattern}' in '{path}'")
    return files


def _read_text_file(filepath):
    # Try utf-8 first, fall back to latin-1 so odd/legacy-encoded files
    # don't blow up the whole batch run.
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        _log.warn(f"utf-8 decode failed on {filepath}, retrying as latin-1")
        with open(filepath, "r", encoding="latin-1") as f:
            return f.read()


class GRLoadTextBatchTest:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "", "multiline": False}),
                "pattern": ("STRING", {"default": "*.txt"}),
                "mode": (["single_text", "incremental_text", "random"], {"default": "incremental_text"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "label": ("STRING", {"default": "Batch 001"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("text", "filename_text", "index_used", "remaining_count", "batch_complete")
    FUNCTION = "load_batch"
    CATEGORY = "GraftingRayman/Text"

    def load_batch(self, path, pattern, mode, index, seed, label):
        files = _list_text_files(path, pattern)
        count = len(files)

        batch_complete = False

        if mode == "single_text":
            idx = index % count

        elif mode == "random":
            rng = random.Random(seed)
            idx = rng.randrange(count)

        else:  # incremental_text
            counter = _load_counter(path, pattern, label)
            if counter >= count:
                idx = count - 1
                batch_complete = True
            else:
                idx = counter
                _save_counter(path, pattern, label, counter + 1)

        filepath = files[idx]
        _log.info(f"Loading [{idx + 1}/{count}] {os.path.basename(filepath)} (mode={mode})")

        text_content = _read_text_file(filepath)

        if mode == "incremental_text":
            remaining_count = 0 if batch_complete else max(0, count - (idx + 1))
        else:
            remaining_count = max(0, count - (idx + 1))

        return (text_content, os.path.basename(filepath), idx, remaining_count, batch_complete)


NODE_CLASS_MAPPINGS = {
    "GRLoadTextBatchTest": GRLoadTextBatchTest,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRLoadTextBatchTest": "GR Load Text Batch [TEST]",
}
