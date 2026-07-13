class GRLongestSideSelector:
    """
    Takes an integer input and passes it through unchanged.
    Also outputs a 'longest side' value:
      - if override_longest_side is enabled (non-zero), that value is used
      - otherwise: integer > 40 -> 1024, integer <= 40 -> 1280
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "integer": ("INT", {
                    "default": 0,
                    "min": -0x7fffffff,
                    "max": 0x7fffffff,
                    "step": 1,
                }),
            },
            "optional": {
                "override_longest_side": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "Set to a value > 0 to override the automatic longest-side selection. Leave at 0 to use automatic behavior.",
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("integer", "longest_side")
    FUNCTION = "select"
    CATEGORY = "GraftingRayman/Utils"

    def select(self, integer, override_longest_side=0):
        if override_longest_side and override_longest_side > 0:
            longest_side = override_longest_side
        else:
            longest_side = 1024 if integer > 40 else 1280

        return (integer, longest_side)


NODE_CLASS_MAPPINGS = {
    "GRLongestSideSelector": GRLongestSideSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRLongestSideSelector": "GR Longest Side Selector",
}
