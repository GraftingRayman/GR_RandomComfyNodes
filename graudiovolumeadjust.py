import torch

class GRAudioVolumeAdjust:
    """
    Increases (or decreases) the volume of an AUDIO input by a gain value,
    then returns the modified AUDIO.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "volume_db": ("FLOAT", {
                    "default": 6.0,
                    "min": -80.0,
                    "max": 40.0,
                    "step": 0.5,
                    "display": "number",
                    "tooltip": "Gain in decibels. Positive = louder, negative = quieter."
                }),
                "allow_clipping": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If disabled, hard-clips the waveform to [-1, 1] after gain. If enabled, leaves samples as-is even if they exceed that range."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "adjust_volume"
    CATEGORY = "GraftingRayman/Audio"

    def adjust_volume(self, audio, volume_db, allow_clipping):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        gain = 10 ** (volume_db / 20.0)
        out_waveform = waveform.clone() * gain

        if not allow_clipping:
            out_waveform = torch.clamp(out_waveform, -1.0, 1.0)

        return ({"waveform": out_waveform, "sample_rate": sample_rate},)


NODE_CLASS_MAPPINGS = {
    "GRAudioVolumeAdjust": GRAudioVolumeAdjust,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRAudioVolumeAdjust": "GR Audio Volume Adjust",
}