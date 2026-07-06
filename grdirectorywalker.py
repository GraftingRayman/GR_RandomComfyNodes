import os
import json
import time
import numpy as np
import torch
from PIL import Image, ImageOps

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gr_directory_walker_state.json")


def _load_state():
    if os.path.isfile(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        print(f"GRDirectoryWalker: failed to save state file: {e}")


def _fit_longest_side_letterbox(img, max_side_length, pad_color=(0, 0, 0)):
    """
    Resize preserving aspect ratio so the longest side == max_side_length,
    then center it on a max_side_length x max_side_length canvas padded
    with pad_color. Lets landscape and portrait images share one batch
    without distortion or cropping.
    """
    w, h = img.size
    scale = max_side_length / float(max(w, h))
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (max_side_length, max_side_length), pad_color)
    paste_x = (max_side_length - new_w) // 2
    paste_y = (max_side_length - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def _collect_image_files(sub_path):
    """
    Recursively collect every supported image file inside sub_path,
    including any nested subfolders within it -- not just files sitting
    directly at the top level.
    """
    found = []
    for root, _dirs, files in os.walk(sub_path):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                found.append(os.path.join(root, fname))
    found.sort()
    return found


def _load_folder_images(sub_path, resize, resize_mode, target_width, target_height,
                         fit_longest_side=False, max_side_length=1024):
    """
    Returns a list of individual IMAGE tensors, each shaped [1, H, W, C].
    Images are NOT stacked into a single batch, so there is no requirement
    for them to share the same dimensions -- each keeps its own size
    (subject to whatever resize option is enabled). ComfyUI's list-output
    mechanism fans these out to downstream nodes one at a time.
    """
    files = _collect_image_files(sub_path)

    tensors = []
    first_size = None

    for fpath in files:
        try:
            img = Image.open(fpath)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
        except Exception as e:
            print(f"GRDirectoryWalker: skipping unreadable image '{fpath}': {e}")
            continue

        if resize:
            if fit_longest_side:
                # Takes priority over resize_mode -- handles mixed
                # landscape/portrait images cleanly via letterbox padding.
                img = _fit_longest_side_letterbox(img, max_side_length)
            elif resize_mode == "match_first_in_folder":
                if first_size is None:
                    first_size = img.size
                elif img.size != first_size:
                    img = img.resize(first_size, Image.LANCZOS)
            elif resize_mode == "stretch_to_size":
                img = img.resize((target_width, target_height), Image.LANCZOS)
            elif resize_mode == "none_skip_mismatched":
                if first_size is None:
                    first_size = img.size
                elif img.size != first_size:
                    print(f"GRDirectoryWalker: skipping mismatched-size image '{fpath}'")
                    continue
        # resize == False: leave the image at its native size, as-is.
        # No uniformity is required -- each image is returned individually.

        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W, C]
        tensors.append(tensor)

    return tensors


class GRDirectoryWalker:
    """
    Point this at a main directory containing multiple subdirectories.

    Each time you queue the prompt, this node outputs the images and
    folder name for ONE subdirectory, advancing to the next subdirectory
    on every subsequent queue. When it reaches the last subdirectory, the
    next queue wraps back around to the first.

    Images are returned as a LIST of individual image tensors rather than
    one stacked batch, so images of different sizes within a subdirectory
    are all preserved as-is -- nothing gets skipped or forced to match.
    Any downstream node runs once per image, with folder_name/index/
    total_folders repeated (broadcast) for each one.

    The index always advances by exactly 1 per queue, in strict order --
    it never silently substitutes a different folder. If a folder turns
    out to have no usable images, this raises a clear error naming that
    folder rather than quietly falling back to an earlier one.

    Position is remembered per directory_path in a small state file next
    to this node, so it persists across ComfyUI restarts. Use the
    reset_index toggle to force it back to the first subdirectory.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "", "multiline": False}),
                "sort_order": (
                    ["name_asc", "name_desc", "date_asc", "date_desc"],
                    {"default": "name_asc"},
                ),
                "resize": ("BOOLEAN", {"default": True}),
                "resize_mode": (
                    ["match_first_in_folder", "stretch_to_size", "none_skip_mismatched"],
                    {"default": "match_first_in_folder"},
                ),
                "target_width": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "target_height": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "fit_longest_side": ("BOOLEAN", {"default": False}),
                "max_side_length": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "reset_index": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("images", "folder_name", "index", "total_folders")
    OUTPUT_IS_LIST = (True, False, False, False)
    FUNCTION = "load"
    CATEGORY = "GraftingRayman/IO"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always report "changed" so ComfyUI re-runs this node on every
        # queue instead of serving a cached output for identical inputs.
        return time.time()

    def load(self, directory_path, sort_order, resize, resize_mode,
              target_width, target_height, fit_longest_side, max_side_length,
              reset_index):

        if not directory_path or not os.path.isdir(directory_path):
            raise ValueError(
                f"GRDirectoryWalker: '{directory_path}' is not a valid directory"
            )

        subdirs = [
            d for d in os.listdir(directory_path)
            if os.path.isdir(os.path.join(directory_path, d))
        ]

        if sort_order == "name_asc":
            subdirs.sort()
        elif sort_order == "name_desc":
            subdirs.sort(reverse=True)
        elif sort_order == "date_asc":
            subdirs.sort(key=lambda d: os.path.getmtime(os.path.join(directory_path, d)))
        elif sort_order == "date_desc":
            subdirs.sort(
                key=lambda d: os.path.getmtime(os.path.join(directory_path, d)),
                reverse=True,
            )

        if not subdirs:
            raise ValueError(
                f"GRDirectoryWalker: no subdirectories found in '{directory_path}'"
            )

        total = len(subdirs)
        state = _load_state()
        key = os.path.abspath(directory_path)

        idx = 0 if reset_index else state.get(key, 0)
        if idx >= total:
            idx = 0

        sub = subdirs[idx]
        sub_path = os.path.join(directory_path, sub)

        # Advance the counter for next run BEFORE attempting to load, so a
        # bad/empty folder doesn't get stuck retrying the same slot forever.
        next_idx = (idx + 1) % total
        state[key] = next_idx
        _save_state(state)

        print(
            f"GRDirectoryWalker: subdirectories in order: {subdirs}\n"
            f"GRDirectoryWalker: this run -> '{sub}' (index {idx + 1}/{total}); "
            f"next queue will use '{subdirs[next_idx]}' (index {next_idx + 1}/{total})"
        )

        images = _load_folder_images(
            sub_path, resize, resize_mode, target_width, target_height,
            fit_longest_side=fit_longest_side, max_side_length=max_side_length,
        )

        if not images:
            raise ValueError(
                f"GRDirectoryWalker: subdirectory '{sub}' contains no supported "
                f"images (checked extensions: {sorted(IMAGE_EXTS)}). The index has "
                f"already advanced -- the next queue will try '{subdirs[next_idx]}' instead."
            )

        return (images, sub, idx, total)


NODE_CLASS_MAPPINGS = {
    "GRDirectoryWalker": GRDirectoryWalker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRDirectoryWalker": "GR Directory Walker (sequential per queue)",
}
