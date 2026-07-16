import re


class GRPromptAttributeReplacer:
    """
    A ComfyUI node that replaces specific facial/appearance attributes (hair colour,
    skin tone, hair style, eye colour, smile, and overall facial expression) in a
    prompt using dropdown selectors, instead of hand-typing "old phrase,new phrase"
    rules every time.

    Each category has:
      - a dropdown of common target values (plus "custom")
      - an optional custom text field, used when the dropdown is set to "custom"
      - an editable multiline mapping box: "source phrase,template"
        where {value} in the template gets replaced with the chosen target value.

    Set a category's dropdown to "none" to leave that attribute untouched.
    Leave the mapping boxes as-is for sensible defaults, or edit/extend them to
    catch whatever phrasing shows up in your prompts.
    """

    # ---- default mapping tables -------------------------------------------------
    # Each line: source_phrase,template   (template uses {value} as a placeholder)

    HAIR_COLOR_OPTIONS = [
        "none", "blonde", "brunette", "black", "dark brown", "light brown",
        "red", "auburn", "gray", "white", "green", "blue", "pink", "purple",
        "turquoise", "silver", "caramel",
        "custom",
    ]
    HAIR_COLOR_MAPPING_DEFAULT = (
        "dark brown hair,{value} hair\n"
        "blonde hair,{value} hair\n"
        "blond hair,{value} hair\n"
        "light brown hair,{value} hair\n"
        "reddish-brown hair,{value} hair\n"
        "reddish-orange hair,{value} hair\n"
        "dark hair,{value} hair\n"
        "brunette hair,{value} hair\n"
        "black hair,{value} hair\n"
        "brown hair,{value} hair\n"
        "red hair,{value} hair\n"
        "auburn hair,{value} hair\n"
        "gray hair,{value} hair\n"
        "grey hair,{value} hair\n"
        "gray-streaked hair,{value} hair\n"
        "blue-streaked hair,{value} hair\n"
        "white hair,{value} hair\n"
        "green hair,{value} hair\n"
        "blue hair,{value} hair\n"
        "pink hair,{value} hair\n"
        "purple hair,{value} hair\n"
        "turquoise hair,{value} hair\n"
        "silver hair,{value} hair\n"
        "caramel-colored hair,{value} hair\n"
        "light blonde,{value}\n"
        "dark blonde,{value}\n"
        "light brunette,{value}\n"
        "dark brunette,{value}\n"
        "blonde,{value}\n"
        "brunette,{value}\n"
        "redhead,{value}\n"
        "ginger,{value}"
    )

    SKIN_TONE_OPTIONS = [
        "none", "pale", "fair", "light", "medium", "tan", "olive", "dark",
        "deep", "freckled",
        "custom",
    ]
    SKIN_TONE_MAPPING_DEFAULT = (
        "light skin,{value} skin\n"
        "pale skin,{value} skin\n"
        "fair skin,{value} skin\n"
        "dark skin,{value} skin\n"
        "tan skin,{value} skin\n"
        "olive skin,{value} skin\n"
        "medium skin,{value} skin\n"
        "freckled skin,{value} skin\n"
        "medium-brown skin,{value} skin\n"
        "medium-tan skin,{value} skin\n"
        "medium-dark skin,{value} skin\n"
        "fair complexion,{value} complexion\n"
        "dark complexion,{value} complexion\n"
        "medium complexion,{value} complexion\n"
        "skin tone is light,skin tone is {value}\n"
        "skin tone is pale,skin tone is {value}\n"
        "skin tone is fair,skin tone is {value}\n"
        "skin tone is dark,skin tone is {value}\n"
        "skin tone is tan,skin tone is {value}\n"
        "skin tone is medium,skin tone is {value}\n"
        "skin tone is medium-brown,skin tone is {value}\n"
        "skin tone is medium-tan,skin tone is {value}\n"
        "skin tone is medium-dark,skin tone is {value}"
    )

    HAIR_STYLE_OPTIONS = [
        "none", "long straight", "long wavy", "long curly", "shoulder-length",
        "short", "bob", "pixie", "ponytail", "high ponytail", "low ponytail",
        "braided", "braids", "bun", "messy bun", "updo", "sleek updo",
        "bangs", "fringe", "afro", "pigtails", "buzz cut",
        "custom",
    ]
    HAIR_STYLE_MAPPING_DEFAULT = (
        "shoulder-length,{value}\n"
        "long straight,{value}\n"
        "long wavy,{value}\n"
        "long curly,{value}\n"
        "loose waves,{value}\n"
        "loose curls,{value}\n"
        "short hair,{value} hair\n"
        "bob haircut,{value} haircut\n"
        "bob cut,{value} cut\n"
        "pixie cut,{value}\n"
        "ponytail,{value}\n"
        "high ponytail,{value}\n"
        "low ponytail,{value}\n"
        "long ponytail,{value}\n"
        "braided hair,{value} hair\n"
        "braids,{value}\n"
        "two braids,{value}\n"
        "bun,{value}\n"
        "messy bun,{value}\n"
        "neat bun,{value}\n"
        "updo,{value}\n"
        "sleek updo,{value}\n"
        "elegant updo,{value}\n"
        "messy updo,{value}\n"
        "bangs,{value}\n"
        "fringe,{value}\n"
        "afro,{value}\n"
        "pigtails,{value}\n"
        "ponytail style haircut,{value}\n"
        "ponytail style,{value}\n"
        "ponytail hairstyle,{value}\n"
        "bun style haircut,{value}\n"
        "updo style haircut,{value}\n"
        "braided style haircut,{value}"
    )

    EYE_COLOR_OPTIONS = [
        "none", "blue", "green", "brown", "hazel", "gray", "amber", "violet",
        "custom",
    ]
    EYE_COLOR_MAPPING_DEFAULT = (
        "blue eyes,{value} eyes\n"
        "green eyes,{value} eyes\n"
        "brown eyes,{value} eyes\n"
        "hazel eyes,{value} eyes\n"
        "gray eyes,{value} eyes\n"
        "grey eyes,{value} eyes\n"
        "dark eyes,{value} eyes\n"
        "light-colored eyes,{value} eyes\n"
        "eyes are blue,eyes are {value}\n"
        "eyes are green,eyes are {value}\n"
        "eyes are brown,eyes are {value}\n"
        "eyes are hazel,eyes are {value}\n"
        "eyes are gray,eyes are {value}\n"
        "eyes are grey,eyes are {value}\n"
        "eyes are dark,eyes are {value}"
    )

    SMILE_OPTIONS = [
        "none", "big smile", "subtle smile", "slight smile", "warm smile",
        "gentle smile", "confident smile", "cheerful smile", "playful smile",
        "friendly smile", "radiant smile", "mischievous smile", "broad smile",
        "no smile", "smirk", "closed-mouth smile", "toothy grin",
        "custom",
    ]
    SMILE_MAPPING_DEFAULT = (
        "big smile,{value}\n"
        "subtle smile,{value}\n"
        "slight smile,{value}\n"
        "warm smile,{value}\n"
        "gentle smile,{value}\n"
        "confident smile,{value}\n"
        "cheerful smile,{value}\n"
        "playful smile,{value}\n"
        "friendly smile,{value}\n"
        "bright smile,{value}\n"
        "radiant smile,{value}\n"
        "inviting smile,{value}\n"
        "mischievous smile,{value}\n"
        "broad smile,{value}\n"
        "wide smile,{value}\n"
        "crooked smile,{value}\n"
        "closed-mouth smile,{value}\n"
        "no smile,{value}\n"
        "smirk,{value}\n"
        "toothy grin,{value}"
    )

    EXPRESSION_OPTIONS = [
        "none", "neutral", "serious", "relaxed", "playful", "confident",
        "cheerful", "peaceful", "serene", "focused", "surprised", "curious",
        "content", "calm", "thoughtful", "contemplative", "happy", "friendly",
        "joyful", "amused", "pensive", "sultry", "seductive", "alluring",
        "sensual",
        "custom",
    ]
    EXPRESSION_MAPPING_DEFAULT = (
        "neutral expression,{value} expression\n"
        "serious expression,{value} expression\n"
        "relaxed expression,{value} expression\n"
        "playful expression,{value} expression\n"
        "confident expression,{value} expression\n"
        "cheerful expression,{value} expression\n"
        "peaceful expression,{value} expression\n"
        "serene expression,{value} expression\n"
        "focused expression,{value} expression\n"
        "surprised expression,{value} expression\n"
        "curious expression,{value} expression\n"
        "content expression,{value} expression\n"
        "calm expression,{value} expression\n"
        "thoughtful expression,{value} expression\n"
        "contemplative expression,{value} expression\n"
        "happy expression,{value} expression\n"
        "friendly expression,{value} expression\n"
        "joyful expression,{value} expression\n"
        "mischievous expression,{value} expression\n"
        "amused expression,{value} expression\n"
        "pensive expression,{value} expression\n"
        "shocked expression,{value} expression\n"
        "sultry expression,{value} expression\n"
        "seductive expression,{value} expression\n"
        "alluring expression,{value} expression\n"
        "sensual expression,{value} expression\n"
        "facial expression,{value} expression"
    )

    # category_key -> (dropdown options, default mapping text)
    CATEGORIES = {
        "hair_color": (HAIR_COLOR_OPTIONS, HAIR_COLOR_MAPPING_DEFAULT),
        "skin_tone": (SKIN_TONE_OPTIONS, SKIN_TONE_MAPPING_DEFAULT),
        "hair_style": (HAIR_STYLE_OPTIONS, HAIR_STYLE_MAPPING_DEFAULT),
        "eye_color": (EYE_COLOR_OPTIONS, EYE_COLOR_MAPPING_DEFAULT),
        "smile": (SMILE_OPTIONS, SMILE_MAPPING_DEFAULT),
        "expression": (EXPRESSION_OPTIONS, EXPRESSION_MAPPING_DEFAULT),
    }

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for cat_key, (options, mapping_default) in cls.CATEGORIES.items():
            label = cat_key.replace("_", " ").title()
            optional[cat_key] = (options, {"default": "none", "label": label})
            optional[f"{cat_key}_custom"] = ("STRING", {
                "default": "",
                "label": f"{label} (custom, used when dropdown = custom)",
            })
            optional[f"{cat_key}_mapping"] = ("STRING", {
                "multiline": True,
                "default": mapping_default,
                "label": f"{label} Mapping (source phrase,template with {{value}})",
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
                "tooltip": "Attempt to maintain the case pattern of the original text",
            }),
            "highlight_format": (["markdown", "html", "plain"], {
                "default": "markdown",
                "label": "Highlight Format",
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

    # ---- self-contained replacement engine (no dependency on GRPromptReplacer.py) --

    def parse_rules(self, rules_text):
        """Parse the multi-line replacement rules into a list of (old, new) tuples."""
        rules = []
        lines = rules_text.strip().split('\n')

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = [part.strip() for part in line.split(',', 1)]

            if len(parts) == 2:
                old_phrase, new_phrase = parts
                if old_phrase:
                    old_phrase = ' '.join(old_phrase.split())
                    new_phrase = ' '.join(new_phrase.split())
                    rules.append((old_phrase, new_phrase))
            else:
                print(f"Warning: Line {line_num} in replacement rules is malformed: '{line}'")

        return rules

    def escape_for_regex(self, text):
        """Escape text for regex but preserve word boundaries for phrases."""
        escaped = re.escape(text)
        escaped = escaped.replace(r'\ ', r'\s+')
        return escaped

    def create_whole_word_pattern(self, phrase):
        """Create a pattern that matches whole words, handling multi-word phrases."""
        words = phrase.split()
        if len(words) == 1:
            return r'\b' + re.escape(words[0]) + r'\b'
        else:
            pattern_parts = []
            for i, word in enumerate(words):
                escaped = re.escape(word)
                if i == 0:
                    pattern_parts.append(r'\b' + escaped)
                elif i == len(words) - 1:
                    pattern_parts.append(escaped + r'\b')
                else:
                    pattern_parts.append(escaped)
            return r'\s+'.join(pattern_parts)

    def preserve_case_pattern(self, matched_text, replacement):
        """Attempt to preserve the case pattern of the matched text."""
        if not matched_text or not replacement:
            return replacement

        if matched_text.isupper():
            return replacement.upper()
        elif matched_text.islower():
            return replacement.lower()
        elif matched_text[0].isupper() and not matched_text[1:].isupper():
            if len(replacement) > 1:
                return replacement[0].upper() + replacement[1:].lower()
            else:
                return replacement.upper()
        elif matched_text[0].isupper() and matched_text[1:].isupper():
            return replacement.upper()
        else:
            result = []
            replacement_chars = list(replacement)
            replacement_index = 0

            for char in matched_text:
                if replacement_index >= len(replacement_chars):
                    break
                if char.isupper():
                    result.append(replacement_chars[replacement_index].upper())
                else:
                    result.append(replacement_chars[replacement_index].lower())
                replacement_index += 1

            if replacement_index < len(replacement_chars):
                result.extend(replacement_chars[replacement_index:])

            return ''.join(result)

    def replace_words_batch(self, text, rules, case_sensitive, match_whole_words, preserve_case):
        """
        Batch replacement using a single regex for all rules.
        Returns both the replaced text and a list of changes made.
        """
        if not rules:
            return text, []

        rules.sort(key=lambda x: len(x[0]), reverse=True)

        pattern_parts = []
        for old_phrase, new_phrase in rules:
            if match_whole_words:
                pattern = self.create_whole_word_pattern(old_phrase)
            else:
                pattern = self.escape_for_regex(old_phrase)
            pattern_parts.append(f'({pattern})')

        combined_pattern = '|'.join(pattern_parts)
        flags = 0 if case_sensitive else re.IGNORECASE

        changes = []
        result_parts = []
        last_end = 0

        def replace_func(match):
            nonlocal last_end

            result_parts.append(text[last_end:match.start()])

            for i, group in enumerate(match.groups()):
                if group is not None:
                    matched_text = group

                    for old_phrase, new_phrase in rules:
                        match_found = False

                        if not case_sensitive:
                            if matched_text.lower() == old_phrase.lower():
                                match_found = True
                            elif re.match(self.create_whole_word_pattern(old_phrase) if match_whole_words else self.escape_for_regex(old_phrase),
                                        matched_text, re.IGNORECASE):
                                match_found = True
                        else:
                            if matched_text == old_phrase:
                                match_found = True
                            elif re.match(self.create_whole_word_pattern(old_phrase) if match_whole_words else self.escape_for_regex(old_phrase),
                                        matched_text):
                                match_found = True

                        if match_found:
                            if preserve_case:
                                replacement = self.preserve_case_pattern(matched_text, new_phrase)
                            else:
                                replacement = new_phrase

                            changes.append({
                                'old': matched_text,
                                'new': replacement,
                                'start': match.start(),
                                'end': match.end(),
                                'rule_used': f"{old_phrase} → {new_phrase}"
                            })

                            result_parts.append(replacement)
                            last_end = match.end()
                            return replacement

                    result_parts.append(matched_text)
                    last_end = match.end()
                    return matched_text

            result_parts.append(match.group(0))
            last_end = match.end()
            return match.group(0)

        try:
            re.sub(combined_pattern, replace_func, text, flags=flags)

            if last_end < len(text):
                result_parts.append(text[last_end:])

            result = ''.join(result_parts)

        except Exception as e:
            print(f"Regex error: {e}")
            result = text
            changes = []
            for old_phrase, new_phrase in rules:
                if match_whole_words:
                    pattern = self.create_whole_word_pattern(old_phrase)
                    rflags = 0 if case_sensitive else re.IGNORECASE

                    def fallback_replacer(match):
                        matched = match.group(0)
                        if preserve_case:
                            replacement = self.preserve_case_pattern(matched, new_phrase)
                        else:
                            replacement = new_phrase

                        changes.append({
                            'old': matched,
                            'new': replacement,
                            'start': match.start(),
                            'end': match.end(),
                            'rule_used': f"{old_phrase} → {new_phrase}"
                        })
                        return replacement

                    result = re.sub(pattern, fallback_replacer, result, flags=rflags)
                else:
                    if case_sensitive:
                        pos = 0
                        while True:
                            idx = result.find(old_phrase, pos)
                            if idx == -1:
                                break

                            changes.append({
                                'old': old_phrase,
                                'new': new_phrase,
                                'start': idx,
                                'end': idx + len(old_phrase),
                                'rule_used': f"{old_phrase} → {new_phrase}"
                            })

                            result = result[:idx] + new_phrase + result[idx + len(old_phrase):]
                            pos = idx + len(new_phrase)
                    else:
                        pattern = re.escape(old_phrase)

                        def fallback_replacer_ci(match):
                            matched = match.group(0)
                            if preserve_case:
                                replacement = self.preserve_case_pattern(matched, new_phrase)
                            else:
                                replacement = new_phrase

                            changes.append({
                                'old': matched,
                                'new': replacement,
                                'start': match.start(),
                                'end': match.end(),
                                'rule_used': f"{old_phrase} → {new_phrase}"
                            })
                            return replacement

                        result = re.sub(pattern, fallback_replacer_ci, result, flags=re.IGNORECASE)

        return result, changes

    def format_changes_report(self, changes, format_type):
        """Create a human-readable report of all changes made."""
        if not changes:
            return "No changes were made."

        if format_type == "markdown":
            report = "### Changes Made:\n\n"
            for i, change in enumerate(changes, 1):
                report += f"{i}. **{change['old']}** → **{change['new']}**  \n"
            return report

        elif format_type == "html":
            report = "<h3>Changes Made:</h3>\n<ul>\n"
            for change in changes:
                report += f'  <li><b>{change["old"]}</b> → <b>{change["new"]}</b></li>\n'
            report += "</ul>"
            return report

        else:
            report = "Changes Made:\n"
            for i, change in enumerate(changes, 1):
                report += f"{i}. {change['old']} -> {change['new']}\n"
            return report

    def replace_words(self, text, replacement_rules, case_sensitive=False,
                       match_whole_words=True, sort_by_length=True, preserve_case=True,
                       highlight_format="markdown"):
        """Apply replacement rules to the input text and return both plain and highlighted versions."""
        if not text or not replacement_rules:
            return (text, text)

        rules = self.parse_rules(replacement_rules)

        if not rules:
            return (text, text)

        if sort_by_length:
            rules.sort(key=lambda x: len(x[0]), reverse=True)

        result, changes = self.replace_words_batch(
            text, rules, case_sensitive, match_whole_words, preserve_case
        )

        if changes:
            changes_sorted = sorted(changes, key=lambda x: x['start'], reverse=True)

            highlighted = text
            for change in changes_sorted:
                old_text = change['old']
                new_text = change['new']

                if highlight_format == "markdown":
                    highlighted = highlighted[:change['start']] + \
                                 f"**{new_text}**" + \
                                 highlighted[change['end']:]
                elif highlight_format == "html":
                    highlighted = highlighted[:change['start']] + \
                                 f'<span style="background-color: #ffff00; font-weight: bold;">{new_text}</span>' + \
                                 highlighted[change['end']:]
                else:
                    highlighted = highlighted[:change['start']] + \
                                 f"[{new_text}]" + \
                                 highlighted[change['end']:]

            changes_report = self.format_changes_report(changes, highlight_format)

            if highlight_format == "markdown":
                highlighted = f"{highlighted}\n\n{changes_report}"
            elif highlight_format == "html":
                highlighted = f"{highlighted}<br><br>{changes_report}"
            else:
                highlighted = f"{highlighted}\n\n{changes_report}"
        else:
            highlighted = text

        return (result, highlighted)

    def build_rules_text(self, **kwargs):
        """Turn the active dropdown selections + mapping tables into a
        newline-delimited 'old,new' rules string."""
        lines = []
        for cat_key in self.CATEGORIES:
            dropdown_val = (kwargs.get(cat_key) or "none").strip()
            if not dropdown_val or dropdown_val == "none":
                continue

            if dropdown_val == "custom":
                actual_value = (kwargs.get(f"{cat_key}_custom") or "").strip()
                if not actual_value:
                    # No custom value supplied, nothing to do for this category
                    continue
            else:
                actual_value = dropdown_val

            mapping_text = kwargs.get(f"{cat_key}_mapping")
            if not mapping_text:
                # Fall back to the built-in default if the widget value is
                # missing or blank (e.g. node called outside ComfyUI).
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

                # Skip no-op rules (source already equals the target text)
                if old_phrase.lower() == new_phrase.lower():
                    continue

                lines.append(f"{old_phrase},{new_phrase}")

        return "\n".join(lines)

    def replace_attributes(self, text, case_sensitive=False, match_whole_words=True,
                            sort_by_length=True, preserve_case=True,
                            highlight_format="markdown", **kwargs):
        rules_text = self.build_rules_text(**kwargs)

        if not text or not rules_text:
            return (text, text, rules_text or "No categories selected.")

        result, highlighted = self.replace_words(
            text, rules_text,
            case_sensitive=case_sensitive,
            match_whole_words=match_whole_words,
            sort_by_length=sort_by_length,
            preserve_case=preserve_case,
            highlight_format=highlight_format,
        )

        return (result, highlighted, rules_text)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GRPromptAttributeReplacer": GRPromptAttributeReplacer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRPromptAttributeReplacer": "GR Prompt Attribute Replacer"
}