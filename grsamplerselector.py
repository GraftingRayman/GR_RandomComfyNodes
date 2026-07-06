import comfy.samplers


class GRSamplerSelector:
    """
    Dropdown selector over every sampler ComfyUI currently knows about
    (comfy.samplers.KSampler.SAMPLERS — this stays in sync automatically if a
    ComfyUI update adds/removes samplers, no hardcoded list to maintain).

    Outputs a SAMPLER object, ready to feed into SamplerCustom /
    SamplerCustomAdvanced (or anything else that expects the SAMPLER type),
    plus the sampler name as a STRING for logging / preset_info-style display.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": comfy.samplers.KSampler.SAMPLERS[0]}),
            }
        }

    RETURN_TYPES = ("SAMPLER", "STRING")
    RETURN_NAMES = ("sampler", "sampler_name")
    FUNCTION     = "get_sampler"
    CATEGORY     = "GraftingRayman/Sigmas"

    def get_sampler(self, sampler_name):
        sampler = comfy.samplers.sampler_object(sampler_name)
        return (sampler, sampler_name)


NODE_CLASS_MAPPINGS = {
    "GR Sampler Selector": GRSamplerSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GR Sampler Selector": "GR Sampler Selector"
}