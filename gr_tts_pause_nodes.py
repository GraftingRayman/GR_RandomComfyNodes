"""
GRTTSTextPauseSplitter / GRTTSAudioPauseConcat
------------------------------------------------
Drop-in replacement for ComfyUI-Qwen-TTS's QwenTTSConfigNode (removed in v1.0.7
"due to voice inconsistency"). Instead of feeding pause markers into the model's
generation path, these two nodes work as pure pre/post-processing:

    1. GRTTSTextPauseSplitter splits your input text into segments on
       punctuation/linebreaks and emits a matching list of "silence to insert
       after this segment" durations.
    2. Feed the segment list straight into the `text` input of VoiceCloneNode /
       CustomVoiceNode / VoiceDesignNode / DialogueInferenceNode (or any other
       TTS node). ComfyUI's native list-expansion runs that node once per
       segment automatically, reusing every other input (voice/seed/reference
       audio) unchanged each time -- so voice identity stays fixed across
       segments instead of drifting.
    3. GRTTSAudioPauseConcat collects the resulting AUDIO list + the pause
       list and stitches them back into one AUDIO with real silence gaps.

Because the model itself never sees pause tokens, this can't reproduce the
"voice inconsistency" bug the upstream node hit -- generation is untouched,
only the finished audio is spliced.

Wiring in a workflow:

    [Text] -> GRTTSTextPauseSplitter -> segments (list) -> VoiceCloneNode.text
                                     +-> pauses (list)  -> GRTTSAudioPauseConcat.pauses
                          VoiceCloneNode.AUDIO (list, auto-batched) -> GRTTSAudioPauseConcat.audio
                                     GRTTSAudioPauseConcat -> AUDIO (final, single clip)
"""

import re
import torch


class GRTTSTextPauseSplitter:
    """
    Splits text into TTS-friendly segments on punctuation/linebreaks and
    outputs a matching list of pause durations (seconds) to insert after
    each segment. Punctuation is kept attached to its segment so the model
    still hears it (natural intonation), the pause is purely an audio-level
    gap added afterward.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "pause_linebreak": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 5.0, "step": 0.05}),
                "period_pause": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.05}),
                "comma_pause": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 5.0, "step": 0.05}),
                "question_pause": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.05}),
                "hyphen_pause": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 5.0, "step": 0.05}),
                "trailing_pause": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.05,
                                              "tooltip": "Silence appended after the FINAL segment."}),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("segments", "pauses")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "split"
    CATEGORY = "GraftingRayman/Audio/TTS"

    # Matches: an optional run of non-terminal text, followed by ONE of the
    # punctuation marks we care about, OR a linebreak. Keeps the punctuation
    # with the preceding text.
    _PUNCT_RE = re.compile(r'([^\n]*?[,\-\.\?])(?=\s|$)|([^\n]+)')

    def split(self, text, pause_linebreak, period_pause, comma_pause,
              question_pause, hyphen_pause, trailing_pause):

        pause_map = {
            ".": period_pause,
            ",": comma_pause,
            "?": question_pause,
            "-": hyphen_pause,
        }

        segments = []
        pauses = []

        lines = text.split("\n")
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            pos = 0
            length = len(line)
            chunk_start = 0
            while pos < length:
                ch = line[pos]
                if ch in pause_map and (pos + 1 == length or line[pos + 1].isspace()):
                    seg = line[chunk_start:pos + 1].strip()
                    if seg:
                        segments.append(seg)
                        pauses.append(pause_map[ch])
                    pos += 1
                    chunk_start = pos
                else:
                    pos += 1

            remainder = line[chunk_start:].strip()
            if remainder:
                segments.append(remainder)
                # No explicit punctuation at end of line -> treat as a soft
                # line-level pause so rhythm isn't lost.
                pauses.append(pause_linebreak)
            elif segments and line_idx < len(lines) - 1:
                # Line ended exactly on punctuation; still respect the
                # linebreak by adding to (not replacing) the punctuation pause.
                pauses[-1] = max(pauses[-1], pause_linebreak)

        if not segments:
            segments = [text.strip() or " "]
            pauses = [0.0]

        # Last segment's pause is "trailing_pause" (usually 0 so you don't
        # get dead air at the very end of the clip).
        pauses[-1] = trailing_pause

        return (segments, pauses)


class GRTTSAudioPauseConcat:
    """
    Gathers a list of AUDIO clips (e.g. one per text segment from a TTS node
    that auto-iterated over GRTTSTextPauseSplitter's segment list) plus a
    matching list of pause durations, and concatenates them into a single
    AUDIO with real silence inserted between segments.

    INPUT_IS_LIST is used here (rather than relying on auto per-item
    execution) because this node needs to see the WHOLE list at once to
    concatenate it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "pauses": ("FLOAT",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    INPUT_IS_LIST = True
    FUNCTION = "concat"
    CATEGORY = "GraftingRayman/Audio/TTS"

    def concat(self, audio, pauses):
        # With INPUT_IS_LIST=True every input arrives wrapped in a list,
        # including ones that aren't semantically lists -- audio/pauses here
        # really are the per-segment lists we want.
        if len(audio) == 0:
            raise ValueError("GRTTSAudioPauseConcat received no audio segments.")

        if len(pauses) != len(audio):
            # Be forgiving: pad/truncate rather than hard-fail on a mismatched
            # wiring, since this is easy to get off-by-one on.
            if len(pauses) < len(audio):
                pauses = list(pauses) + [0.0] * (len(audio) - len(pauses))
            else:
                pauses = list(pauses)[:len(audio)]

        sample_rate = None
        for clip in audio:
            if clip is not None and clip.get("waveform") is not None:
                sample_rate = int(clip["sample_rate"])
                break
        if sample_rate is None:
            raise ValueError("GRTTSAudioPauseConcat: no valid audio segments found.")

        max_channels = 1
        waveforms = []
        for clip in audio:
            wf = clip["waveform"]
            sr = int(clip["sample_rate"])
            if sr != sample_rate:
                wf = self._resample(wf, sr, sample_rate)
            # Normalize shape to (channels, samples); we collapse batch dim
            # (assume batch size 1 per TTS call, which is standard here).
            if wf.dim() == 3:
                wf = wf[0]
            elif wf.dim() == 1:
                wf = wf.unsqueeze(0)
            max_channels = max(max_channels, wf.shape[0])
            waveforms.append(wf)

        pieces = []
        for wf, pause_s in zip(waveforms, pauses):
            if wf.shape[0] < max_channels:
                wf = wf.expand(max_channels, -1).contiguous()
            pieces.append(wf)
            pause_s = float(pause_s)
            if pause_s > 0:
                silence_len = int(round(pause_s * sample_rate))
                if silence_len > 0:
                    pieces.append(torch.zeros(
                        (max_channels, silence_len), dtype=wf.dtype, device=wf.device))

        combined = torch.cat(pieces, dim=1)  # (channels, total_samples)
        combined = combined.unsqueeze(0)     # -> (batch=1, channels, samples)

        return ({"waveform": combined, "sample_rate": sample_rate},)

    @staticmethod
    def _resample(waveform, orig_sr, target_sr):
        # Lightweight linear-interpolation resample so we don't need
        # torchaudio (kept optional/absent-friendly per this pipeline's
        # existing soundfile-over-torchaudio convention).
        if orig_sr == target_sr:
            return waveform
        orig_shape = waveform.dim()
        if orig_shape == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif orig_shape == 2:
            waveform = waveform.unsqueeze(0)
        # waveform: (batch, channels, samples)
        new_len = int(round(waveform.shape[-1] * target_sr / orig_sr))
        resampled = torch.nn.functional.interpolate(
            waveform, size=new_len, mode="linear", align_corners=False)
        if orig_shape == 1:
            return resampled[0, 0]
        elif orig_shape == 2:
            return resampled[0]
        return resampled


NODE_CLASS_MAPPINGS = {
    "GRTTSTextPauseSplitter": GRTTSTextPauseSplitter,
    "GRTTSAudioPauseConcat": GRTTSAudioPauseConcat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRTTSTextPauseSplitter": "GR TTS Text Pause Splitter",
    "GRTTSAudioPauseConcat": "GR TTS Audio Pause Concat",
}
