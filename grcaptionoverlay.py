"""
GRCaptionOverlay - ComfyUI custom node
Accepts an IMAGE batch tensor (B,H,W,C) and AUDIO from upstream nodes
(e.g. VHS Load Video), transcribes speech/lyrics via faster-whisper,
generates a timed ASS subtitle file with configurable font/style/glow,
burns it in via ffmpeg subprocess, and returns the captioned frames as
an IMAGE batch + the original AUDIO separately (ready for VHS Video Combine).
"""

import os
import hashlib
import subprocess
import json
import tempfile
import shutil
import time

import numpy as np
import torch

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None  # fallback: console only

try:
    import folder_paths
    _COMFY_BASE = folder_paths.base_path
except Exception:
    _COMFY_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

THIS_NODE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Logger - prints to ComfyUI console with timing, drives ProgressBar in UI
# ---------------------------------------------------------------------------

class GRLogger:
    PREFIX = "[GRCaptionOverlay]"

    def __init__(self, total_steps: int):
        self._step       = 0
        self._total      = total_steps
        self._t0         = time.time()
        self._step_t     = time.time()
        self._pbar       = ProgressBar(total_steps) if ProgressBar else None

    def _elapsed(self):
        return time.time() - self._t0

    def _step_elapsed(self):
        dt = time.time() - self._step_t
        self._step_t = time.time()
        return dt

    def log(self, msg: str):
        print(f"{self.PREFIX} [{self._elapsed():.1f}s] {msg}", flush=True)

    def step(self, msg: str):
        self._step += 1
        dt = self._step_elapsed()
        tag = f"step {self._step}/{self._total}"
        print(f"{self.PREFIX} [{self._elapsed():.1f}s] ({tag}) {msg}", flush=True)
        if self._pbar:
            self._pbar.update(1)

    def done(self, msg: str = ""):
        total_t = time.time() - self._t0
        print(f"{self.PREFIX} finished in {total_t:.1f}s. {msg}", flush=True)


# ---------------------------------------------------------------------------
# Auto-derived paths - no hardcoded drives or OS-specific locations
# ---------------------------------------------------------------------------

def _resolve_working_root():
    env = os.environ.get("GR_CAPTION_WORK_DIR")
    if env:
        return env
    try:
        base = folder_paths.get_temp_directory()
    except Exception:
        base = os.path.join(_COMFY_BASE, "temp")
    return os.path.join(base, "gr_caption_work")


WORKING_ROOT = _resolve_working_root()
os.makedirs(WORKING_ROOT, exist_ok=True)

NODE_LOCAL_FONTS_DIR = os.path.join(THIS_NODE_DIR, "fonts")
os.makedirs(NODE_LOCAL_FONTS_DIR, exist_ok=True)
DEFAULT_FONTSDIR = os.environ.get("GR_FONTS_DIR", NODE_LOCAL_FONTS_DIR)

WHISPER_MODEL_SIZE = "large-v3"
WHISPER_DEVICE     = "cuda"
WHISPER_COMPUTE    = "float16"

FONT_EXTENSIONS = (".ttf", ".otf", ".ttc")

SYSTEM_FONT_DIRS = [
    r"C:\Windows\Fonts",
    os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\Fonts") if os.name == "nt" else "",
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    os.path.expanduser("~/.fonts"),
    os.path.expanduser("~/.local/share/fonts"),
    "/System/Library/Fonts",
    "/Library/Fonts",
    os.path.expanduser("~/Library/Fonts"),
    DEFAULT_FONTSDIR,
]

def _ffmpeg_filter_path(p):
    """Escape a file path for use inside an ffmpeg filtergraph string on any OS.
    ffmpeg's filter parser uses ':' as an option separator and '\\' as an escape
    character, so both must be escaped. Forward slashes are used throughout so
    Windows drive letters don't introduce a second unescaped colon."""
    return p.replace("\\", "/").replace(":", "\\:")


ALIGNMENT_MAP = {
    "bottom_center": 2,
    "bottom_left":   1,
    "bottom_right":  3,
    "middle_center": 5,
    "middle_left":   4,
    "middle_right":  6,
    "top_center":    8,
    "top_left":      7,
    "top_right":     9,
}

MODE_LIST = ["per_word", "per_line", "karaoke_fill"]


# ---------------------------------------------------------------------------
# Font scanning
# ---------------------------------------------------------------------------

def _read_font_family_name(path):
    try:
        from fontTools.ttLib import TTFont
        tt = TTFont(path, lazy=True, fontNumber=0)
        name_table = tt["name"]
        for name_id in (16, 1):
            rec = name_table.getName(name_id, 3, 1, 0x409)
            if rec is None:
                rec = name_table.getName(name_id, 1, 0, 0)
            if rec is not None:
                return str(rec)
    except Exception:
        pass
    return os.path.splitext(os.path.basename(path))[0]


def _scan_system_fonts():
    found = {}
    for d in SYSTEM_FONT_DIRS:
        if not d or not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            for fname in files:
                if fname.lower().endswith(FONT_EXTENSIONS):
                    fpath = os.path.join(root, fname)
                    family = _read_font_family_name(fpath)
                    if family and family not in found:
                        found[family] = fpath
    return found


_FONT_CACHE = None


def _get_font_cache():
    global _FONT_CACHE
    if _FONT_CACHE is None:
        _FONT_CACHE = _scan_system_fonts()
    return _FONT_CACHE


# ---------------------------------------------------------------------------
# ASS helpers
# ---------------------------------------------------------------------------

def _color_to_ass(hex_color, alpha=0):
    """'#RRGGBB' -> ASS '&HAABBGGRR'"""
    h = hex_color.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b.upper()}{g.upper()}{r.upper()}"


def _fmt_ts(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:01d}:{m:02d}:{s:05.2f}"


def _build_style_block(p, width, height):
    align   = ALIGNMENT_MAP[p["placement"]]
    primary = _color_to_ass(p["font_color"])
    outline = _color_to_ass(p["outline_color"])
    bold    = 1 if p["bold"] else 0
    italic  = 1 if p["italic"] else 0

    lines = [
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
    ]

    if p["glow_enabled"]:
        glow_col = _color_to_ass(p["glow_color"])
        lines.append(
            f"Style: Glow,{p['font_name']},{p['font_size']},"
            f"{glow_col},&H00000000,{glow_col},&H00000000,"
            f"{bold},{italic},0,0,100,100,0,0,1,"
            f"{p['glow_outline_width']},0,{align},"
            f"{p['margin_lr']},{p['margin_lr']},{p['margin_v']},1"
        )

    lines.append(
        f"Style: Crisp,{p['font_name']},{p['font_size']},"
        f"{primary},&H00000000,{outline},&H00000000,"
        f"{bold},{italic},0,0,100,100,0,0,1,"
        f"{p['outline_width']},{p['shadow_depth']},{align},"
        f"{p['margin_lr']},{p['margin_lr']},{p['margin_v']},1"
    )
    return "\n".join(lines)


def _build_events(transcript, p):
    lines     = ["[Events]", "Format: Layer, Start, End, Style, Text"]
    glow      = p["glow_enabled"]
    blur_tag  = f"{{\\blur{p['glow_blur']}}}" if glow else ""

    if p["mode"] == "per_word":
        for seg in transcript:
            words = seg["words"] or [{"word": seg["text"], "start": seg["start"], "end": seg["end"]}]
            for w in words:
                if not w["word"]:
                    continue
                s, e, t = _fmt_ts(w["start"]), _fmt_ts(w["end"]), w["word"]
                if glow:
                    lines.append(f"Dialogue: 0,{s},{e},Glow,{blur_tag}{t}")
                lines.append(f"Dialogue: 1,{s},{e},Crisp,{t}")

    elif p["mode"] == "per_line":
        for seg in transcript:
            if not seg["text"]:
                continue
            s, e, t = _fmt_ts(seg["start"]), _fmt_ts(seg["end"]), seg["text"]
            if glow:
                lines.append(f"Dialogue: 0,{s},{e},Glow,{blur_tag}{t}")
            lines.append(f"Dialogue: 1,{s},{e},Crisp,{t}")

    elif p["mode"] == "karaoke_fill":
        for seg in transcript:
            if not seg["words"]:
                continue
            s, e = _fmt_ts(seg["start"]), _fmt_ts(seg["end"])
            k_text = "".join(
                f"{{\\k{max(1, int(round((w['end'] - w['start']) * 100)))}}}{w['word']}"
                for w in seg["words"]
            )
            if glow:
                lines.append(f"Dialogue: 0,{s},{e},Glow,{blur_tag}{k_text}")
            lines.append(f"Dialogue: 1,{s},{e},Crisp,{k_text}")

    return "\n".join(lines)


def _write_ass(path, params, transcript, width, height):
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "ScaledBorderAndShadow: yes\n\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(_build_style_block(params, width, height))
        f.write("\n\n")
        f.write(_build_events(transcript, params))
        f.write("\n")


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class GRCaptionOverlay:
    """
    Inputs:
      images  - IMAGE batch tensor (B, H, W, C) float32  from upstream node
      audio   - AUDIO dict {"waveform": tensor, "sample_rate": int}
      fps     - frame rate of the incoming image batch
      + style/font/glow widgets

    Outputs:
      images  - IMAGE batch tensor (B, H, W, C) with captions burned in
      audio   - AUDIO dict (pass-through, unchanged)
    """

    @classmethod
    def INPUT_TYPES(cls):
        font_cache   = _get_font_cache()
        font_choices = sorted(font_cache.keys()) or ["Arial"]
        return {
            "required": {
                "images":             ("IMAGE",),
                "audio":              ("AUDIO",),
                "fps":                ("FLOAT",  {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "mode":               (MODE_LIST, {"default": "per_word"}),
                "font_name":          (font_choices, {"default": font_choices[0]}),
                "font_size":          ("INT",    {"default": 64,  "min": 12, "max": 300}),
                "font_color":         ("STRING", {"default": "#FFFFFF"}),
                "outline_color":      ("STRING", {"default": "#000000"}),
                "outline_width":      ("INT",    {"default": 3,   "min": 0,  "max": 20}),
                "shadow_depth":       ("INT",    {"default": 1,   "min": 0,  "max": 20}),
                "bold":               ("BOOLEAN",{"default": True}),
                "italic":             ("BOOLEAN",{"default": False}),
                "glow_enabled":       ("BOOLEAN",{"default": False}),
                "glow_color":         ("STRING", {"default": "#FFA500"}),
                "glow_blur":          ("INT",    {"default": 8,   "min": 0,  "max": 30}),
                "glow_outline_width": ("INT",    {"default": 8,   "min": 0,  "max": 40}),
                "placement":          (list(ALIGNMENT_MAP.keys()), {"default": "bottom_center"}),
                "margin_v":           ("INT",    {"default": 80,  "min": 0,  "max": 500}),
                "margin_lr":          ("INT",    {"default": 40,  "min": 0,  "max": 500}),
                "whisper_language":   ("STRING", {"default": "auto"}),
                "whisper_device":     (["cuda", "cpu", "auto"], {"default": "auto"}),
                "whisper_compute":    (["float16", "int8_float16", "int8", "float32"], {"default": "int8_float16"}),
            },
            "optional": {
                "font_file_override": ("STRING", {"default": ""}),
                "extra_fontsdir":     ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE", "AUDIO")
    RETURN_NAMES  = ("images", "overlay", "audio")
    FUNCTION      = "run"
    CATEGORY      = "GraftingRayman/Video"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _content_hash(self, images_tensor, audio, log: GRLogger):
        """Hash the content of the incoming batch so we can key a work dir."""
        B, H, W, C = images_tensor.shape
        log.log(f"hashing input — {B} frames @ {W}x{H}")
        h    = hashlib.sha1()
        step = max(1, B // 30)
        sample = images_tensor[::step].contiguous().cpu().numpy()
        h.update(sample.tobytes())
        if audio is not None:
            h.update(str(audio.get("sample_rate", 0)).encode())
            h.update(audio["waveform"].cpu().numpy().tobytes()[:1_000_000])
        digest = h.hexdigest()
        log.log(f"content hash: {digest[:12]}… (work dir keyed)")
        return digest

    def _get_work_dir(self, content_hash):
        d = os.path.join(WORKING_ROOT, content_hash)
        os.makedirs(d, exist_ok=True)
        return d

    def _style_hash(self, params):
        """Short hash of the visual style params so style changes bust the render cache
        without affecting the transcript/audio cache."""
        key = json.dumps(params, sort_keys=True)
        return hashlib.sha1(key.encode()).hexdigest()[:12]

    def _frames_to_video(self, images, fps, work_dir, log: GRLogger):
        """Write IMAGE batch tensor -> silent .mp4 in work_dir."""
        raw_path    = os.path.join(work_dir, "input_frames.mp4")
        done_marker = raw_path + ".done"
        if os.path.exists(done_marker) and os.path.exists(raw_path):
            log.log("input_frames.mp4 cached — skipping frame write")
            return raw_path

        B, H, W, C = images.shape
        raw_bytes   = (images.cpu().numpy() * 255).clip(0, 255).astype(np.uint8).tobytes()
        log.log(f"writing {B} frames to temp video ({W}x{H} @ {fps}fps, "
                f"{len(raw_bytes) / 1_048_576:.1f} MB raw)…")

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-video_size", f"{W}x{H}", "-framerate", str(fps),
            "-i", "pipe:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            raw_path,
        ]
        # communicate() feeds all stdin bytes then reads stderr concurrently,
        # avoiding the deadlock caused by ffmpeg's stderr buffer filling up
        # while we are still blocked writing to stdin.
        proc   = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        _, err = proc.communicate(input=raw_bytes)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg frame write failed:\n{err.decode()}")

        size_mb = os.path.getsize(raw_path) / 1_048_576
        log.log(f"frame write done → {raw_path} ({size_mb:.1f} MB)")
        open(done_marker, "w").close()
        return raw_path

    def _audio_to_wav(self, audio, work_dir, log: GRLogger):
        """Write AUDIO dict -> wav file."""
        wav_path = os.path.join(work_dir, "audio.wav")
        if os.path.exists(wav_path):
            log.log("audio.wav cached — skipping audio write")
            return wav_path
        if sf is None:
            raise RuntimeError("soundfile is not installed.")
        waveform    = audio["waveform"]
        sample_rate = audio["sample_rate"]
        arr = waveform.squeeze().cpu().numpy()
        if arr.ndim == 2:
            arr = arr.T
        duration = arr.shape[0] / sample_rate
        log.log(f"writing audio — {sample_rate}Hz, {duration:.1f}s, "
                f"{arr.shape[-1] if arr.ndim == 2 else 1}ch")
        sf.write(wav_path, arr, sample_rate)
        log.log(f"audio written → {wav_path}")
        return wav_path

    def _transcribe(self, wav_path, work_dir, language, log: GRLogger,
                    device="auto", compute="int8_float16"):
        transcript_path = os.path.join(work_dir, "transcript.json")
        if os.path.exists(transcript_path):
            with open(transcript_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            word_count = sum(len(s["words"]) for s in data)
            log.log(f"transcript cached — {len(data)} segments, {word_count} words")
            return data

        if WhisperModel is None:
            raise RuntimeError("faster_whisper is not installed.")

        # resolve device
        if device == "auto":
            try:
                import torch
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                resolved_device = "cpu"
        else:
            resolved_device = device

        # on CPU, float16 is not supported — downgrade silently
        resolved_compute = compute
        if resolved_device == "cpu" and resolved_compute in ("float16", "int8_float16"):
            resolved_compute = "int8"
            log.log(f"CPU device: compute type downgraded to int8")

        def _load_and_run(dev, comp):
            log.log(f"loading Whisper model ({WHISPER_MODEL_SIZE}, device={dev}, compute={comp})…")
            m = WhisperModel(WHISPER_MODEL_SIZE, device=dev, compute_type=comp)
            log.log("model loaded — starting transcription…")
            return m

        try:
            model = _load_and_run(resolved_device, resolved_compute)
        except Exception as e:
            if resolved_device == "cuda":
                log.log(f"CUDA load failed ({e}) — falling back to CPU int8")
                resolved_device  = "cpu"
                resolved_compute = "int8"
                model = _load_and_run(resolved_device, resolved_compute)
            else:
                raise

        kwargs = {"word_timestamps": True}
        if language and language.lower() != "auto":
            kwargs["language"] = language
            log.log(f"language forced: {language}")
        else:
            log.log("language: auto-detect")

        segments, info = model.transcribe(wav_path, **kwargs)

        result       = []
        seg_count    = 0
        word_count   = 0
        last_print_t = 0.0

        for seg in segments:
            words = [{"word": w.word.strip(), "start": w.start, "end": w.end}
                     for w in (seg.words or [])]
            result.append({
                "start": seg.start, "end": seg.end,
                "text": seg.text.strip(), "words": words,
            })
            seg_count  += 1
            word_count += len(words)
            if seg.end - last_print_t >= 5.0:
                log.log(f"  transcribed to {seg.end:.1f}s — "
                        f"{seg_count} segs, {word_count} words so far | "
                        f"last: \"{seg.text.strip()[:60]}\"")
                last_print_t = seg.end

        log.log(f"transcription complete — {seg_count} segments, {word_count} words")
        if hasattr(info, "language"):
            log.log(f"detected language: {info.language} "
                    f"(confidence {getattr(info, 'language_probability', 0):.2f})")

        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        log.log(f"transcript saved → {transcript_path}")
        return result

    def _resolve_fontsdir(self, font_name, font_file_override, extra_fontsdir, log: GRLogger):
        resolved_dir = os.path.join(WORKING_ROOT, "_resolved_fonts")
        os.makedirs(resolved_dir, exist_ok=True)

        if font_file_override and os.path.isfile(font_file_override):
            src = font_file_override
            log.log(f"font override: {src}")
        else:
            cache = _get_font_cache()
            src   = cache.get(font_name)
            log.log(f"font lookup: '{font_name}' → {src or 'NOT FOUND'}")
            if not src and extra_fontsdir and os.path.isdir(extra_fontsdir):
                for fname in os.listdir(extra_fontsdir):
                    if fname.lower().endswith(FONT_EXTENSIONS):
                        src = os.path.join(extra_fontsdir, fname)
                        log.log(f"font found in extra_fontsdir: {src}")
                        break
            if not src:
                raise RuntimeError(
                    f"Could not resolve font '{font_name}'. "
                    "Provide font_file_override or extra_fontsdir."
                )

        dest = os.path.join(resolved_dir, os.path.basename(src))
        if not os.path.exists(dest):
            shutil.copy2(src, dest)
            log.log(f"font copied to resolved cache → {dest}")
        else:
            log.log(f"font already in resolved cache: {dest}")
        return resolved_dir

    def _burn_captions(self, silent_video, ass_path, fontsdir, fps, work_dir, style_hash, log: GRLogger):
        out_path    = os.path.join(work_dir, f"captioned_{style_hash}.mp4")
        done_marker = out_path + ".done"
        if os.path.exists(done_marker) and os.path.exists(out_path):
            log.log("captioned.mp4 cached — skipping burn-in")
            return out_path

        log.log("burning captions into video…")
        ass_esc   = _ffmpeg_filter_path(ass_path)
        fonts_esc = _ffmpeg_filter_path(fontsdir)
        vf        = f"ass='{ass_esc}':fontsdir='{fonts_esc}'"
        log.log(f"filter string: {vf}")

        cmd = [
            "ffmpeg", "-y", "-i", silent_video,
            "-vf", vf,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-an", out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg caption burn-in failed:\n{result.stderr}")

        size_mb = os.path.getsize(out_path) / 1_048_576
        log.log(f"burn-in done → {out_path} ({size_mb:.1f} MB)")
        open(done_marker, "w").close()
        return out_path

    def _burn_overlay_only(self, ass_path, fontsdir, fps, B, W, H, work_dir, style_hash, log: GRLogger):
        out_path    = os.path.join(work_dir, f"overlay_{style_hash}.mp4")
        done_marker = out_path + ".done"

        if not (os.path.exists(done_marker) and os.path.exists(out_path)):
            duration  = B / fps
            log.log(f"rendering text-only overlay ({W}x{H}, {duration:.2f}s, transparent bg)…")
            ass_esc   = _ffmpeg_filter_path(ass_path)
            fonts_esc = _ffmpeg_filter_path(fontsdir)
            vf        = f"ass='{ass_esc}':fontsdir='{fonts_esc}':alpha=1"
            log.log(f"overlay filter string: {vf}")
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"color=black@0.0:size={W}x{H}:rate={fps}:duration={duration}",
                "-vf", vf,
                "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p",
                "-auto-alt-ref", "0", "-an", out_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg overlay render failed:\n{result.stderr}")
            size_mb = os.path.getsize(out_path) / 1_048_576
            log.log(f"overlay render done → {out_path} ({size_mb:.1f} MB)")
            open(done_marker, "w").close()
        else:
            log.log("overlay_only.mp4 cached — skipping overlay render")

        log.log("reading overlay frames back (RGBA)…")
        cmd = [
            "ffmpeg", "-i", out_path,
            "-vf", "format=rgba",
            "-f", "rawvideo",
            "-pix_fmt", "rgba",
            "pipe:1",
        ]
        proc     = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw, err = proc.communicate()

        frame_size = W * H * 4
        n_frames   = min(len(raw) // frame_size, B)
        log.log(f"overlay: {len(raw) / 1_048_576:.1f} MB received, "
                f"{n_frames} RGBA frames decoded (expected {B})")
        frames_np = (
            np.frombuffer(raw[: n_frames * frame_size], dtype=np.uint8)
            .reshape(n_frames, H, W, 4)
            .astype(np.float32) / 255.0
        )
        if n_frames < B:
            log.log(f"overlay: padding {B - n_frames} missing frames with zeros")
            pad = np.zeros((B - n_frames, H, W, 4), dtype=np.float32)
            frames_np = np.concatenate([frames_np, pad], axis=0)

        return torch.from_numpy(frames_np)

    def _video_to_tensor(self, video_path, log: GRLogger):
        """Read captioned video back into an IMAGE batch tensor (B,H,W,3) float32."""
        log.log(f"reading captioned frames from {os.path.basename(video_path)}…")
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
            "-of", "json", video_path,
        ]
        probe = subprocess.run(probe_cmd, capture_output=True, text=True)
        info  = json.loads(probe.stdout)
        W  = info["streams"][0]["width"]
        H  = info["streams"][0]["height"]
        nb = info["streams"][0].get("nb_frames", "?")
        log.log(f"captioned video: {W}x{H}, ~{nb} frames")

        # -vf format=rgb24 forces the colour-space conversion pipeline explicitly;
        # -pix_fmt rgb24 tells the rawvideo muxer the output pixel format.
        # Using both ensures we never get YUV bytes when we expect RGB.
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", "format=rgb24",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1",
        ]
        proc     = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw, err = proc.communicate()

        frame_size     = W * H * 3
        expected_total = int(nb) * frame_size if str(nb).isdigit() else None
        log.log(f"received {len(raw) / 1_048_576:.1f} MB raw "
                f"(expected ~{expected_total / 1_048_576:.1f} MB for {nb} frames @ {W}x{H} RGB24)"
                if expected_total else
                f"received {len(raw) / 1_048_576:.1f} MB raw")

        if len(raw) == 0:
            raise RuntimeError(
                f"ffmpeg returned 0 bytes reading {video_path}.\n"
                f"stderr: {err.decode()}"
            )

        if frame_size > 0 and len(raw) % frame_size != 0:
            log.log(f"WARNING: raw byte count {len(raw)} is not a multiple of "
                    f"frame_size {frame_size} ({W}x{H}x3) — possible pixel format mismatch")

        n_frames  = len(raw) // frame_size
        log.log(f"decoded {n_frames} RGB frames")
        frames_np = (
            np.frombuffer(raw[: n_frames * frame_size], dtype=np.uint8)
            .reshape(n_frames, H, W, 3)
            .astype(np.float32) / 255.0
        )
        return torch.from_numpy(frames_np)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(
        self, images, audio, fps,
        mode, font_name, font_size, font_color,
        outline_color, outline_width, shadow_depth,
        bold, italic,
        glow_enabled, glow_color, glow_blur, glow_outline_width,
        placement, margin_v, margin_lr,
        whisper_language, whisper_device, whisper_compute,
        font_file_override="", extra_fontsdir="",
    ):
        B, H, W, C = images.shape

        # 8 steps total for the progress bar
        log = GRLogger(total_steps=8)
        log.log(f"=== GR Caption Overlay start ===")
        log.log(f"input: {B} frames, {W}x{H}, {C}ch | fps={fps} | mode={mode}")
        log.log(f"font: {font_name} {font_size}pt | color={font_color} | "
                f"outline={outline_color} w={outline_width} | "
                f"glow={'on blur=' + str(glow_blur) if glow_enabled else 'off'}")
        log.log(f"whisper: model={WHISPER_MODEL_SIZE} device={whisper_device} compute={whisper_compute} lang={whisper_language}")
        log.log(f"work root: {WORKING_ROOT}")

        # 1. Hash content → work dir
        content_hash = self._content_hash(images, audio, log)
        work_dir     = self._get_work_dir(content_hash)
        log.step(f"work dir: {work_dir}")

        params = dict(
            mode=mode, font_name=font_name, font_size=font_size,
            font_color=font_color, outline_color=outline_color,
            outline_width=outline_width, shadow_depth=shadow_depth,
            bold=bold, italic=italic,
            glow_enabled=glow_enabled, glow_color=glow_color,
            glow_blur=glow_blur, glow_outline_width=glow_outline_width,
            placement=placement, margin_v=margin_v, margin_lr=margin_lr,
        )

        # style hash keys the render outputs independently of the content hash
        # so changing font/size/placement regenerates captioned + overlay
        # without redoing the transcript or frame write
        s_hash = self._style_hash(params)
        log.log(f"style hash: {s_hash} | font={font_name} {font_size}pt "
                f"| placement={placement} | color={font_color} "
                f"| outline={outline_color} w={outline_width} "
                f"| glow={'on blur=' + str(glow_blur) if glow_enabled else 'off'}")

        # 2. Frames → silent temp video
        silent_video = self._frames_to_video(images, fps, work_dir, log)
        log.step("frames written to temp video")

        # 3. Audio → wav
        wav_path = self._audio_to_wav(audio, work_dir, log)
        log.step("audio extracted to wav")

        # 4. Transcribe
        transcript   = self._transcribe(wav_path, work_dir, whisper_language, log,
                                        device=whisper_device, compute=whisper_compute)
        word_count   = sum(len(s["words"]) for s in transcript)
        log.step(f"transcription done — {len(transcript)} segments, {word_count} words")

        # 5. Write ASS file (style-hashed filename so each style gets its own)
        ass_path = os.path.join(work_dir, f"captions_{s_hash}.ass")
        _write_ass(ass_path, params, transcript, W, H)
        ass_size = os.path.getsize(ass_path)
        log.log(f"ASS subtitle file written ({ass_size} bytes) → {ass_path}")
        log.step("ASS subtitle file ready")

        # 6. Resolve font, burn captions into main video
        fontsdir       = self._resolve_fontsdir(font_name, font_file_override, extra_fontsdir, log)
        captioned_path = self._burn_captions(silent_video, ass_path, fontsdir, fps, work_dir, s_hash, log)
        log.step("captions burned into main video")

        # 7. Read captioned frames back as tensor
        out_images = self._video_to_tensor(captioned_path, log)
        log.step(f"main output tensor ready: {tuple(out_images.shape)}")

        # 8. Render text-only overlay (transparent RGBA)
        overlay = self._burn_overlay_only(ass_path, fontsdir, fps, B, W, H, work_dir, s_hash, log)
        log.step(f"overlay tensor ready: {tuple(overlay.shape)}")

        log.done(f"outputs → images {tuple(out_images.shape)} | "
                 f"overlay {tuple(overlay.shape)} | audio pass-through")

        return (out_images, overlay, audio)


NODE_CLASS_MAPPINGS = {
    "GRCaptionOverlay": GRCaptionOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRCaptionOverlay": "GR Caption Overlay",
}