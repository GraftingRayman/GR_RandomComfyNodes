import re
import time
from datetime import datetime

# Dropdown label -> strftime pattern ("UNIX" is handled as a special case)
DATE_FORMATS = {
    "YYYY-MM-DD": "%Y-%m-%d",
    "DD-MM-YYYY": "%d-%m-%Y",
    "MM-DD-YYYY": "%m-%d-%Y",
    "YYYYMMDD": "%Y%m%d",
    "DDMMYYYY": "%d%m%Y",
    "MMDDYYYY": "%m%d%Y",
    "YY-MM-DD": "%y-%m-%d",
    "YYYY-MM-DD_HH-MM-SS": "%Y-%m-%d_%H-%M-%S",
    "YYYYMMDD_HHMMSS": "%Y%m%d_%H%M%S",
    "HH-MM-SS": "%H-%M-%S",
    "Month DD YYYY": "%B %d %Y",
    "DD Month YYYY": "%d %B %Y",
    "Day, DD Month YYYY": "%A, %d %B %Y",
    "UNIX Timestamp": "UNIX",
}


def sanitize_filename(value):
    """Strip all whitespace, then strip anything that isn't filename-safe.
    No slashes allowed - use this for the date portion on its own."""
    value = value.strip()
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[^A-Za-z0-9_\-]", "", value)
    return value


def sanitize_path(value):
    """Strip all whitespace, then strip anything that isn't filename/path-safe,
    but keep / and \\ so downstream nodes can still split into folders."""
    value = value.strip()
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[^A-Za-z0-9_\-/\\]", "", value)
    return value


class GRDateAppendText:
    """
    Pick a date format from a dropdown, get today's date in that format
    sanitized to be filename-safe (no spaces/special chars), then prepend
    it to a multiline text input. Returns the date, the appended result,
    and the original multiline text unchanged - all as STRING.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prefix_text": ("STRING", {"multiline": True, "default": ""}),
                "date_format": (list(DATE_FORMATS.keys()), {"default": "YYYY-MM-DD"}),
                "text": ("STRING", {"multiline": True, "default": ""}),
                "separator": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prefix_text", "date_string", "appended_text", "original_text")
    FUNCTION = "process"
    CATEGORY = "GraftingRayman/Text"

    def process(self, prefix_text, date_format, text, separator=""):
        fmt = DATE_FORMATS.get(date_format, "%Y-%m-%d")

        if fmt == "UNIX":
            raw_date = str(int(time.time()))
        else:
            raw_date = datetime.now().strftime(fmt)

        safe_date = sanitize_filename(raw_date)
        safe_separator = sanitize_filename(separator) if separator else ""

        appended_text = sanitize_path(f"{prefix_text}{safe_date}{safe_separator}{text}")

        return (prefix_text, safe_date, appended_text, text)


NODE_CLASS_MAPPINGS = {
    "GRDateAppendText": GRDateAppendText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRDateAppendText": "GR Date Append Text",
}