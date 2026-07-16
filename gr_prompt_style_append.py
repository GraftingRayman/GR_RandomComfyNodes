"""
GRPromptStyleAppend
--------------------
Takes a text prompt, lets you pick an animation/art style from a dropdown,
and returns the style text prepended to the front of the original prompt.

Style phrasing follows "The image is a <style> style image." - kept short and
plain, since Z-Image (bf16 base, Lumina2-type text encoder) responded far
better to this simple form than to longer descriptive sentences or comma-
separated tag lists, even with a character LoRA active.

Drop this file into your custom_nodes package alongside your other GR nodes
and make sure NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS from this file
get merged into your package's __init__.py (same pattern as your other nodes).
"""

NO_STYLE = "None"
STYLE_PREFIX = "The image is "
STYLE_SUFFIX = " style image."

STYLE_PRESETS = {
    "Anime": "an anime",
    "Arcane Inspired": "an Arcane-inspired painterly animation",
    "CGI Realistic Animation": "a photorealistic CGI animation",
    "Chibi Animation": "a chibi animation",
    "Classic Cartoon": "a classic 2D cartoon",
    "Clay Animation": "a clay animation",
    "Comic Animation": "a comic book animation",
    "Cutout Animation": "a paper cutout animation",
    "Disney Style": "a Disney-inspired animation",
    "DreamWorks Style": "a DreamWorks-inspired 3D animation",
    "Flash Animation": "a Flash animation",
    "Isometric Diorama Animation": "an isometric diorama animation",
    "Modern Mobile Animation": "a stylized mobile game animation",
    "Pixar Style": "a Pixar-inspired 3D animation",
    "Puppet Animation": "a puppet animation",
    "Retro Vaporwave Animation": "a retro vaporwave animation",
    "Rotoscope Animation": "a rotoscope animation",
    "Rubber Hose Cartoon": "a 1930s rubber hose cartoon",
    "Saturday Morning Cartoon": "a 1980s Saturday morning cartoon",
    "Spider-Verse Inspired": "a Spider-Verse-inspired comic animation",
    "Studio Ghibli Inspired": "a Studio Ghibli-inspired animation",
}


class GRPromptStyleAppend:
    """Prepends 'The image is a <style> style image.' to an incoming prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        style_choices = [NO_STYLE] + list(STYLE_PRESETS.keys())
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True, "multiline": True, "default": ""}),
                "style": (style_choices, {"default": NO_STYLE}),
                "separator": ("STRING", {"default": " \n\n", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "append_style"
    CATEGORY = "GraftingRayman/Prompt"

    def append_style(self, prompt, style, separator):
        prompt = prompt or ""

        if style == NO_STYLE:
            return (prompt,)

        style_desc = STYLE_PRESETS.get(style, "")

        if not style_desc:
            result = prompt
        else:
            style_text = f"{STYLE_PREFIX}{style_desc}{STYLE_SUFFIX}"
            if not prompt.strip():
                result = style_text
            else:
                result = f"{style_text}{separator}{prompt}"

        return (result,)


NODE_CLASS_MAPPINGS = {
    "GRPromptStyleAppend": GRPromptStyleAppend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRPromptStyleAppend": "GR Prompt Style Append",
}