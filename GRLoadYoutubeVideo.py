"""
GRLoadYoutubeVideo
-------------------
Downloads a YouTube video from a URL (via yt-dlp) and returns it as an
IMAGE batch (frames) + AUDIO, following the same conventions as the rest
of the GR package:

  - content-hash keyed working folder (hash of the URL + options)
  - .done marker resume logic so re-queuing the same URL doesn't re-download
  - GRLogger-style logging
  - ProgressBar-style progress reporting during frame extraction
  - duration_seconds_int / longest_side outputs, matching
    GRLoadAudioWithDuration / GRLoadAudioImageBatchWithDuration

Requires: yt-dlp, ffmpeg (on PATH or set FFMPEG_PATH env var), soundfile.

    pip install yt-dlp soundfile
"""

import os
import io
import json
import hashlib
import subprocess
import shutil

import numpy as np
import torch
import soundfile as sf

try:
    import folder_paths
    COMFY_OUTPUT_DIR = folder_paths.get_output_directory()
except Exception:
    COMFY_OUTPUT_DIR = os.path.join(os.getcwd(), "output")


# ---------------------------------------------------------------------------
# Shared GR conventions
# ---------------------------------------------------------------------------

class GRLogger:
    """Minimal shared logger matching the rest of the GR package."""
    PREFIX = "[GR]"

    @classmethod
    def info(cls, msg):
        print(f"{cls.PREFIX} {msg}")

    @classmethod
    def warn(cls, msg):
        print(f"{cls.PREFIX} [WARN] {msg}")

    @classmethod
    def error(cls, msg):
        print(f"{cls.PREFIX} [ERROR] {msg}")


class ProgressBar:
    """Thin wrapper so this works whether or not comfy.utils is available."""

    def __init__(self, total):
        self.total = max(total, 1)
        self._impl = None
        try:
            from comfy.utils import ProgressBar as _CPB
            self._impl = _CPB(total)
        except Exception:
            self._impl = None
        self.current = 0

    def update(self, value):
        self.current = value
        if self._impl is not None:
            self._impl.update(value)
        else:
            GRLogger.info(f"progress {value}/{self.total}")


def _content_hash(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
    return h.hexdigest()[:16]


def _working_dir(url, fmt, max_height):
    key = _content_hash(url, fmt, max_height)
    base = os.path.join(COMFY_OUTPUT_DIR, "gr_youtube_cache", key)
    os.makedirs(base, exist_ok=True)
    return base, key


def _ffmpeg_bin():
    return os.environ.get("FFMPEG_PATH", "ffmpeg")


def _ffprobe_bin():
    return os.environ.get("FFPROBE_PATH", "ffprobe")


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class GRLoadYoutubeVideo:
    """
    Downloads a YouTube video by URL and returns frames as an IMAGE batch
    plus AUDIO, alongside duration/longest_side metadata.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "", "multiline": False}),
                "max_height": ("INT", {"default": 1080, "min": 144, "max": 4320, "step": 1}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "force_redownload": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "cookies_file": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "INT", "STRING")
    RETURN_NAMES = ("images", "audio", "duration_seconds_int", "longest_side", "video_title")
    FUNCTION = "load"
    CATEGORY = "GraftingRayman/video"

    # -- download -----------------------------------------------------------

    def _download(self, url, max_height, cookies_file, work_dir):
        done_marker = os.path.join(work_dir, ".done")
        video_path = os.path.join(work_dir, "source.mp4")
        info_path = os.path.join(work_dir, "info.json")

        if os.path.exists(done_marker) and os.path.exists(video_path):
            GRLogger.info(f"cache hit, skipping download: {video_path}")
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            return video_path, info

        GRLogger.info(f"downloading via yt-dlp: {url}")

        try:
            import yt_dlp
        except ImportError as e:
            raise RuntimeError(
                "yt-dlp is not installed. Run: pip install yt-dlp"
            ) from e

        ydl_opts = {
            "format": f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]",
            "outtmpl": os.path.join(work_dir, "source.%(ext)s"),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        if cookies_file:
            ydl_opts["cookiefile"] = cookies_file

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        # yt-dlp may not name the merged file exactly source.mp4 depending on codecs
        produced = os.path.join(work_dir, f"source.{info.get('ext', 'mp4')}")
        if produced != video_path and os.path.exists(produced):
            shutil.move(produced, video_path)
        elif not os.path.exists(video_path):
            # fallback: find whatever got produced
            candidates = [f for f in os.listdir(work_dir) if f.startswith("source.")]
            if not candidates:
                raise RuntimeError("yt-dlp reported success but no output file was found")
            shutil.move(os.path.join(work_dir, candidates[0]), video_path)

        slim_info = {
            "title": info.get("title", ""),
            "duration": info.get("duration", 0),
            "width": info.get("width", 0),
            "height": info.get("height", 0),
        }
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(slim_info, f)

        with open(done_marker, "w") as f:
            f.write("ok")

        GRLogger.info(f"download complete: {video_path}")
        return video_path, slim_info

    # -- probing --------------------------------------------------------

    def _probe(self, video_path):
        cmd = [
            _ffprobe_bin(), "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")
        data = json.loads(result.stdout)

        v_stream = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
        duration = float(data.get("format", {}).get("duration", 0.0))
        width = int(v_stream.get("width", 0)) if v_stream else 0
        height = int(v_stream.get("height", 0)) if v_stream else 0
        fps_raw = v_stream.get("r_frame_rate", "30/1") if v_stream else "30/1"
        num, den = fps_raw.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 30.0

        return duration, width, height, fps

    # -- frame extraction -------------------------------------------------

    def _extract_frames(self, video_path, work_dir, skip_first_frames,
                         select_every_nth, frame_load_cap, fps):
        frames_dir = os.path.join(work_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        existing = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
        if existing and os.path.exists(os.path.join(work_dir, ".frames_done")):
            GRLogger.info(f"cache hit, {len(existing)} frames already extracted")
        else:
            for f in existing:
                os.remove(os.path.join(frames_dir, f))

            vf_parts = []
            if select_every_nth > 1:
                vf_parts.append(f"select='not(mod(n\\,{select_every_nth}))'")
            vf = ",".join(vf_parts) if vf_parts else None

            cmd = [_ffmpeg_bin(), "-y", "-i", video_path]
            if skip_first_frames > 0:
                skip_seconds = skip_first_frames / fps if fps else 0
                cmd = [_ffmpeg_bin(), "-y", "-ss", str(skip_seconds), "-i", video_path]
            if vf:
                cmd += ["-vf", vf, "-vsync", "vfr"]
            if frame_load_cap > 0:
                cmd += ["-frames:v", str(frame_load_cap)]
            cmd += [os.path.join(frames_dir, "frame_%06d.png")]

            GRLogger.info(f"extracting frames: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg frame extraction failed: {result.stderr}")

            with open(os.path.join(work_dir, ".frames_done"), "w") as f:
                f.write("ok")

        frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
        if frame_load_cap > 0:
            frame_files = frame_files[:frame_load_cap]
        if not frame_files:
            raise RuntimeError("no frames were extracted from the video")

        pb = ProgressBar(len(frame_files))
        from PIL import Image
        tensors = []
        for i, fname in enumerate(frame_files):
            img = Image.open(os.path.join(frames_dir, fname)).convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(arr))
            pb.update(i + 1)

        return torch.stack(tensors, dim=0)

    # -- audio extraction -------------------------------------------------

    def _extract_audio(self, video_path, work_dir):
        audio_path = os.path.join(work_dir, "audio.wav")
        if not os.path.exists(audio_path):
            cmd = [
                _ffmpeg_bin(), "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                audio_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                GRLogger.warn(f"audio extraction failed, returning silence: {result.stderr}")
                waveform = torch.zeros((1, 2, 44100))
                return {"waveform": waveform, "sample_rate": 44100}

        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T).unsqueeze(0)  # (1, channels, samples)
        return {"waveform": waveform, "sample_rate": sr}

    # -- main ---------------------------------------------------------------

    def load(self, url, max_height, frame_load_cap, skip_first_frames,
              select_every_nth, force_redownload, cookies_file=""):

        if not url.strip():
            raise ValueError("GRLoadYoutubeVideo: url is empty")

        work_dir, cache_key = _working_dir(url, "mp4", max_height)
        GRLogger.info(f"cache key: {cache_key}")

        if force_redownload and os.path.exists(work_dir):
            shutil.rmtree(work_dir)
            os.makedirs(work_dir, exist_ok=True)

        video_path, info = self._download(url, max_height, cookies_file, work_dir)
        duration, width, height, fps = self._probe(video_path)

        images = self._extract_frames(
            video_path, work_dir, skip_first_frames, select_every_nth,
            frame_load_cap, fps,
        )
        audio = self._extract_audio(video_path, work_dir)

        longest_side = max(width, height)
        duration_seconds_int = int(round(duration))
        title = info.get("title", "")

        GRLogger.info(
            f"done: {images.shape[0]} frames, {duration_seconds_int}s, "
            f"{width}x{height}, title='{title}'"
        )

        return (images, audio, duration_seconds_int, longest_side, title)


NODE_CLASS_MAPPINGS = {
    "GRLoadYoutubeVideo": GRLoadYoutubeVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRLoadYoutubeVideo": "GR Load YouTube Video",
}
