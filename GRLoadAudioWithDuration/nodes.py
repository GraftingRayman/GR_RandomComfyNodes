import os
import math

import numpy as np
import torch
import soundfile as sf

import folder_paths
import node_helpers


def _duration_to_int(duration_seconds: float) -> int:
    """
    Converts a duration in seconds to an integer:
      - floor(duration) + 1, normally
      - floor(duration) + 2, if the fractional part is > 0.9

    Examples:
      40.0   -> 41
      40.01  -> 41
      40.9   -> 41
      40.91  -> 42
      40.99  -> 42
    """
    floor_val = math.floor(duration_seconds)
    frac = duration_seconds - floor_val
    return floor_val + (2 if frac > 0.9 else 1)


class GRLoadAudioWithDuration:
    """
    Loads an uploaded audio file and returns:
      1. AUDIO - the standard ComfyUI audio dict ({"waveform": tensor, "sample_rate": int})
      2. INT   - floor(duration_seconds) + 1, or +2 if the fractional part exceeds 0.9
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
            and f.lower().endswith((".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aiff", ".aif"))
        ]
        if not files:
            files = [""]
        return {
            "required": {
                "audio": (sorted(files),),
            },
            "optional": {
                "override_duration_int": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0x7fffffff,
                    "step": 1,
                    "tooltip": "Set to a value > 0 to override the computed duration integer. Leave at 0 to use the automatic calculation.",
                }),
                "override_longest_side": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "Set to a value > 0 to override the automatic longest-side selection. Leave at 0 for automatic (duration_seconds_int > 40 -> 1024, else 1280).",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT", "INT")
    RETURN_NAMES = ("audio", "duration_seconds_int", "longest_side")
    FUNCTION = "load"
    CATEGORY = "GraftingRayman/Audio"

    def load(self, audio, override_duration_int=0, override_longest_side=0):
        audio_path = folder_paths.get_annotated_filepath(audio)

        # soundfile instead of torchaudio.load, per established Windows-compat pattern
        data, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        # data shape from soundfile: (samples, channels) -> convert to (channels, samples)
        waveform = torch.from_numpy(data.T).unsqueeze(0)  # (1, channels, samples)

        num_samples = data.shape[0]
        duration_seconds = num_samples / float(sample_rate)

        if override_duration_int and override_duration_int > 0:
            duration_int = override_duration_int
        else:
            duration_int = _duration_to_int(duration_seconds)

        if override_longest_side and override_longest_side > 0:
            longest_side = override_longest_side
        else:
            longest_side = 1024 if duration_int > 40 else 1280

        audio_out = {"waveform": waveform, "sample_rate": sample_rate}

        return (audio_out, duration_int, longest_side)

    @classmethod
    def IS_CHANGED(cls, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        m = os.path.getmtime(audio_path)
        return m

    @classmethod
    def VALIDATE_INPUTS(cls, audio):
        if not folder_paths.exists_annotated_filepath(audio):
            return "Invalid audio file: {}".format(audio)
        return True


NODE_CLASS_MAPPINGS = {
    "GRLoadAudioWithDuration": GRLoadAudioWithDuration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRLoadAudioWithDuration": "GR Load Audio (+Duration Int)",
}
