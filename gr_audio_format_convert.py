"""
GRAudioFormatConvert
--------------------
Takes a ComfyUI AUDIO input (any waveform/sample_rate it was loaded with) and
writes it out in a chosen container/codec: wav, mp3, flac, ogg, m4a (aac).

Handles any input shape ComfyUI AUDIO nodes commonly produce:
    waveform: torch.Tensor of shape [B, C, T], [C, T], or [T]
    sample_rate: int

Output:
    AUDIO  - the converted audio re-loaded from disk, so downstream nodes
             see exactly what was written (e.g. any resampling / lossy
             artifacts the codec introduced)
    STRING - full path to the saved file (last file if batch > 1)

Encoding strategy:
    - wav / flac  -> written directly with soundfile (lossless, no ffmpeg dep)
    - mp3 / ogg / m4a -> written to a temp wav via soundfile, then transcoded
      with ffmpeg (must be on PATH - same dependency GRCaptionOverlay uses)

Drop this file in your custom_nodes package alongside your other GR nodes
and add the mappings below to your package's __init__.py (or leave the
mappings in this file if your loader auto-discovers them per-file).
"""

import os
import subprocess
import tempfile
import shutil

import numpy as np
import soundfile as sf
import torch

try:
    import folder_paths
    OUTPUT_DIR = folder_paths.get_output_directory()
except Exception:
    OUTPUT_DIR = os.path.join(os.getcwd(), "output")

FORMAT_CHOICES = ["wav", "mp3", "flac", "ogg", "m4a"]

# soundfile subtype per format (only used for the wav/flac direct path)
SF_SUBTYPE = {
    "wav": "PCM_16",
    "flac": "PCM_16",
}

# ffmpeg args per lossy/compressed target (applied to the transcode step)
FFMPEG_CODEC_ARGS = {
    "mp3": ["-c:a", "libmp3lame"],
    "ogg": ["-c:a", "libvorbis"],
    "m4a": ["-c:a", "aac"],
}


class GRAudioFormatConvert:
    """
    Converts an AUDIO input to a selected file format (mp3/wav/flac/ogg/m4a)
    and returns it re-loaded as AUDIO, plus the saved path.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "format": (FORMAT_CHOICES, {"default": "mp3"}),
                "filename_prefix": ("STRING", {"default": "GR_audio"}),
                "mp3_bitrate_kbps": ("INT", {"default": 192, "min": 64, "max": 320, "step": 32}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "file_path")
    FUNCTION = "convert"
    CATEGORY = "GraftingRayman/audio"
    OUTPUT_NODE = True

    # ---- helpers -----------------------------------------------------

    @staticmethod
    def _to_numpy_2d(waveform_item):
        """
        Normalize a single audio item to a 2D numpy array shaped [T, C]
        (what soundfile.write expects), from a tensor that may be
        [C, T], [T], or already numpy.
        """
        if isinstance(waveform_item, torch.Tensor):
            arr = waveform_item.detach().cpu().float().numpy()
        else:
            arr = np.asarray(waveform_item, dtype=np.float32)

        if arr.ndim == 1:
            # [T] -> mono
            arr = arr[:, None]
        elif arr.ndim == 2:
            # assume [C, T] (ComfyUI convention) -> transpose to [T, C]
            arr = arr.T
        else:
            raise ValueError(f"Unsupported waveform shape: {arr.shape}")

        return np.clip(arr, -1.0, 1.0)

    @staticmethod
    def _ffmpeg_available():
        return shutil.which("ffmpeg") is not None

    def _transcode_with_ffmpeg(self, src_wav_path, dst_path, fmt, mp3_bitrate_kbps):
        if not self._ffmpeg_available():
            raise RuntimeError(
                "ffmpeg not found on PATH - required for mp3/ogg/m4a output. "
                "Install ffmpeg or choose wav/flac output instead."
            )

        cmd = ["ffmpeg", "-y", "-i", src_wav_path]
        cmd += FFMPEG_CODEC_ARGS[fmt]
        if fmt == "mp3":
            cmd += ["-b:a", f"{mp3_bitrate_kbps}k"]
        cmd += [dst_path]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed converting to {fmt}:\n{result.stderr}")

    def _save_one(self, arr_2d, sample_rate, fmt, out_path, mp3_bitrate_kbps):
        if fmt in ("wav", "flac"):
            sf.write(out_path, arr_2d, sample_rate, subtype=SF_SUBTYPE[fmt])
            return out_path

        # mp3 / ogg / m4a: write a temp wav first, then transcode
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav_path = tmp.name
        try:
            sf.write(tmp_wav_path, arr_2d, sample_rate, subtype="PCM_16")
            self._transcode_with_ffmpeg(tmp_wav_path, out_path, fmt, mp3_bitrate_kbps)
        finally:
            if os.path.exists(tmp_wav_path):
                os.remove(tmp_wav_path)
        return out_path

    # ---- main ----------------------------------------------------------

    def convert(self, audio, format, filename_prefix, mp3_bitrate_kbps, output_dir=""):
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])

        target_dir = output_dir.strip() or OUTPUT_DIR
        os.makedirs(target_dir, exist_ok=True)

        # Normalize to a batch list of [T, C] arrays regardless of input rank
        if isinstance(waveform, torch.Tensor):
            if waveform.dim() == 3:
                items = [waveform[i] for i in range(waveform.shape[0])]
            elif waveform.dim() in (1, 2):
                items = [waveform]
            else:
                raise ValueError(f"Unsupported AUDIO waveform rank: {waveform.dim()}")
        else:
            items = [waveform]

        saved_paths = []
        counter = 1
        for item in items:
            arr_2d = self._to_numpy_2d(item)
            # find a free filename
            while True:
                fname = f"{filename_prefix}_{counter:05d}.{format}"
                out_path = os.path.join(target_dir, fname)
                if not os.path.exists(out_path):
                    break
                counter += 1
            self._save_one(arr_2d, sample_rate, format, out_path, mp3_bitrate_kbps)
            saved_paths.append(out_path)
            counter += 1

        # Reload the last saved file so downstream nodes see the true
        # converted audio (post-resample / post-codec).
        reloaded_waveforms = []
        reload_sr = sample_rate
        for p in saved_paths:
            data, sr = sf.read(p, dtype="float32", always_2d=True)  # [T, C]
            reload_sr = sr
            tensor = torch.from_numpy(data.T)  # -> [C, T]
            reloaded_waveforms.append(tensor)

        max_len = max(t.shape[-1] for t in reloaded_waveforms)
        max_c = max(t.shape[0] for t in reloaded_waveforms)
        batch = torch.zeros((len(reloaded_waveforms), max_c, max_len), dtype=torch.float32)
        for i, t in enumerate(reloaded_waveforms):
            batch[i, :t.shape[0], :t.shape[1]] = t

        out_audio = {"waveform": batch, "sample_rate": reload_sr}
        return (out_audio, saved_paths[-1])


NODE_CLASS_MAPPINGS = {
    "GRAudioFormatConvert": GRAudioFormatConvert,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRAudioFormatConvert": "GR Audio Format Convert",
}
