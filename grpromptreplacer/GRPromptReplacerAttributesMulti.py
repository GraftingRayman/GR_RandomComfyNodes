import random

try:
    from .GRPromptReplacerAttributesBasic import GRPromptReplacerAttributesBasic
except ImportError:
    # fallback for flat/non-package layouts
    from GRPromptReplacerAttributesBasic import GRPromptReplacerAttributesBasic


class GRPromptReplacerAttributesMulti(GRPromptReplacerAttributesBasic):
    """
    Same behaviour as GRPromptReplacerAttributesBasic, but every category is
    a tick-box multi-select instead of a single dropdown. The checklist UI
    is provided by a custom JS widget (see web/js/gr_multi_select_widget.js)
    that stores the ticked options as a comma-separated string on a plain
    STRING input, so the node also works fine from the API without the JS
    (just pass a comma-separated list of option names).

    Resolution rules per category, each run:
      - 0 ticked  -> category is skipped (same as "none"), UNLESS
        "randomize" is enabled and the category is randomizable, in which
        case a value is picked from the category's FULL option list.
      - 1 ticked  -> that value is always used.
      - 2+ ticked -> ONE value is picked at random from just the ticked
        options, using "seed" (0 = fresh random each run, otherwise
        deterministic). This happens regardless of the "randomize" toggle,
        since ticking several boxes is itself a request for variety.
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for cat_key, (options, mapping_default) in cls.CATEGORIES.items():
            label = cat_key.replace("_", " ").title()
            tooltip = "Tick as many as you like - one is picked per run."
            if cat_key == "photograph_type":
                tooltip = ("Intelligently inserts distance/shot type "
                           "(e.g. 'close up photograph'). Tick as many as you like.")

            optional[cat_key] = ("STRING", {
                "default": "",
                "multiline": False,
                "multi_select": True,
                "options": [o for o in options if o != "none"],
                "label": label,
                "tooltip": tooltip,
                "placeholder": "none selected",
            })

        optional.update({
            "case_sensitive": ("BOOLEAN", {
                "default": False,
                "label": "Case Sensitive Matching",
            }),
            "match_whole_words": ("BOOLEAN", {
                "default": True,
                "label": "Match Whole Words Only",
                "tooltip": "When enabled, 'man' won't match 'woman'",
            }),
            "sort_by_length": ("BOOLEAN", {
                "default": True,
                "label": "Sort Rules by Length (Longest First)",
                "tooltip": "Prevents shorter phrases from replacing parts of longer phrases",
            }),
            "preserve_case": ("BOOLEAN", {
                "default": True,
                "label": "Preserve Original Case Pattern",
            }),
            "highlight_format": (["markdown", "html", "plain"], {
                "default": "markdown",
                "label": "Highlight Format",
            }),
            "randomize": ("BOOLEAN", {
                "default": False,
                "label": "Randomize Empty Categories",
                "tooltip": "When enabled, categories with nothing ticked get a random value from their full option list",
            }),
            "seed": ("INT", {
                "default": 0,
                "min": 0,
                "max": 0xFFFFFFFF,
                "label": "Seed",
                "tooltip": "Controls which value gets picked whenever a category has more than one box ticked (0 = random each run)",
            }),
        })

        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter text to process...",
                }),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "highlighted_text", "rules_applied")
    FUNCTION = "replace_attributes"
    CATEGORY = "GR Utilities"

    # ---- helpers --------------------------------------------------------

    @staticmethod
    def _parse_selection(raw_value, options):
        """Turn the widget's comma-separated string into a de-duplicated
        list of valid options, preserving canonical casing/order from the
        category's option list."""
        if not raw_value:
            return []
        wanted = {v.strip().lower() for v in raw_value.split(",") if v.strip()}
        wanted.discard("none")
        if not wanted:
            return []
        return [opt for opt in options if opt.lower() in wanted]

    def _resolve_value(self, cat_key, raw_value, randomize, random_gen):
        options = self.CATEGORIES[cat_key][0]
        selected = self._parse_selection(raw_value, options)

        if not selected:
            if randomize and cat_key in self.RANDOMIZABLE_CATEGORIES:
                return self.get_random_value(cat_key, random_gen)
            return "none"

        if len(selected) == 1:
            return selected[0]

        return random_gen.choice(selected)

    # ---- overrides --------------------------------------------------------

    def build_rules_text(self, **kwargs):
        randomize = kwargs.get('randomize', False)
        seed = kwargs.get('seed', 0)
        random_gen = random.Random(seed) if seed else random.Random()

        lines = []
        for cat_key in self.CATEGORIES:
            if cat_key in self.SPECIAL_CATEGORIES:
                continue

            raw_value = kwargs.get(cat_key) or ""
            actual_value = self._resolve_value(cat_key, raw_value, randomize, random_gen)

            if actual_value == "none":
                continue

            mapping_text = self.CATEGORIES[cat_key][1]
            for raw_line in mapping_text.strip().split("\n"):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = [p.strip() for p in line.split(",", 1)]
                if len(parts) != 2:
                    continue

                old_phrase, template = parts
                if not old_phrase or not template:
                    continue

                new_phrase = template.replace("{value}", actual_value)
                if old_phrase.lower() == new_phrase.lower():
                    continue

                lines.append(f"{old_phrase},{new_phrase}")

        return "\n".join(lines)

    def replace_attributes(self, text, case_sensitive=False, match_whole_words=True,
                            sort_by_length=True, preserve_case=True,
                            highlight_format="markdown", randomize=False, seed=0, **kwargs):
        if not text:
            return (text, text, "No text provided.")

        random_gen = random.Random(seed) if seed else random.Random()

        all_changes = []
        result_text = text

        photograph_raw = kwargs.get('photograph_type', "")
        photograph_type_value = self._resolve_value(
            'photograph_type', photograph_raw, randomize, random_gen
        )

        if photograph_type_value != 'none':
            result_text, photo_changes = self.apply_photograph_type_fallback(
                result_text, photograph_type_value, preserve_case
            )
            all_changes.extend(photo_changes)

        rules_text = self.build_rules_text(randomize=randomize, seed=seed, **kwargs)

        if rules_text:
            result_text, highlighted = self.replace_words(
                result_text, rules_text,
                case_sensitive=case_sensitive,
                match_whole_words=match_whole_words,
                sort_by_length=sort_by_length,
                preserve_case=preserve_case,
                highlight_format=highlight_format,
            )
        else:
            highlighted = result_text

        rules_report = rules_text or "No standard rules applied."
        if all_changes:
            rules_report += f"\n\nSpecial category changes: {len(all_changes)}"

        return (result_text, highlighted, rules_report)


NODE_CLASS_MAPPINGS = {
    "GRPromptReplacerAttributesMulti": GRPromptReplacerAttributesMulti
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRPromptReplacerAttributesMulti": "GR Prompt Replacer Attributes (Multi-Select)"
}
