import os
import json


class GRIntIncrement:
    """
    Simple incrementing counter node.

    - start_value: value used the first time the node runs, or whenever
      start_value itself is changed (also honoured by the manual reset toggle)
    - increment_by: amount added to the current value on every subsequent run
    - use_stop_value / stop_value: once the running value reaches stop_value
      it holds there (doesn't overshoot, doesn't wrap) until start_value changes
    - Outputs the current value as both INT and FLOAT

    State persists across queue runs using a small JSON state file, so it
    survives ComfyUI restarts as well as normal "run node again" behaviour.
    """

    CATEGORY = "GraftingRayman/Utils"
    FUNCTION = "increment"
    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("INT", "FLOAT")

    _STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gr_state")
    _STATE_FILE = os.path.join(_STATE_DIR, "gr_int_increment_state.json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_value": ("INT", {"default": 0, "min": -1_000_000_000, "max": 1_000_000_000, "step": 1}),
                "increment_by": ("INT", {"default": 1, "min": -1_000_000_000, "max": 1_000_000_000, "step": 1}),
                "use_stop_value": ("BOOLEAN", {"default": False}),
                "stop_value": ("INT", {"default": 100, "min": -1_000_000_000, "max": 1_000_000_000, "step": 1}),
                "reset": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "unique_id": ("STRING", {"default": "default", "multiline": False}),
                "video_id": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
            },
        }

    @classmethod
    def _load_state(cls):
        if not os.path.exists(cls._STATE_FILE):
            return {}
        try:
            with open(cls._STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    @classmethod
    def _save_state(cls, state):
        os.makedirs(cls._STATE_DIR, exist_ok=True)
        with open(cls._STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f)

    def increment(self, start_value, increment_by, use_stop_value=False, stop_value=100,
                  reset=False, unique_id="default", video_id=""):
        state = self._load_state()
        key = str(unique_id)
        entry = state.get(key)

        # (Re)start whenever there's no prior state, reset is ticked,
        # start_value itself has changed, or the video_id input has changed
        # (e.g. a new video/file was loaded upstream).
        if (
            reset
            or entry is None
            or entry.get("start_value") != start_value
            or entry.get("video_id") != video_id
        ):
            current = start_value
        else:
            current = entry["current"] + increment_by

            if use_stop_value:
                if increment_by >= 0:
                    if current >= stop_value:
                        current = stop_value
                else:
                    if current <= stop_value:
                        current = stop_value

        state[key] = {"start_value": start_value, "current": current, "video_id": video_id}
        self._save_state(state)

        return (int(current), float(current))

    @classmethod
    def IS_CHANGED(cls, start_value, increment_by, use_stop_value=False, stop_value=100,
                    reset=False, unique_id="default", video_id=""):
        # Force this node to re-run every queue execution rather than being
        # cached as identical, since its output changes with no input change.
        return float("nan")


NODE_CLASS_MAPPINGS = {
    "GRIntIncrement": GRIntIncrement,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRIntIncrement": "GR Int Increment",
}