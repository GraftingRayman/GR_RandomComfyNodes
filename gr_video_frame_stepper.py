import os
import json
import glob

import cv2
import numpy as np
import torch


VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


class GRVideoFrameStepper:
    """
    Steps through frames of the videos in a folder, one run at a time.

    - path: folder containing video files. Files are sorted alphabetically
      and processed in that order.
    - start_value: frame index used at the start of each video (and on reset)
    - increment_by: how many frames to advance per run
    - When the frame index would run past the end of the current video, it
      moves on to the next video file in the folder and resets to start_value.
    - loop_folder: when the last video is exhausted, start over from the
      first video instead of holding on the final frame.

    Outputs the current frame index as INT and FLOAT, the frame itself as an
    IMAGE, and the filename of the video that frame came from.
    """

    CATEGORY = "GraftingRayman/Video"
    FUNCTION = "step"
    RETURN_TYPES = ("INT", "FLOAT", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("frame_index", "frame_index_float", "image", "filename", "folder_name")

    _STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gr_state")
    _STATE_FILE = os.path.join(_STATE_DIR, "gr_video_frame_stepper_state.json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "", "multiline": False}),
                "start_value": ("INT", {"default": 0, "min": 0, "max": 1_000_000_000, "step": 1}),
                "increment_by": ("INT", {"default": 1, "min": 1, "max": 1_000_000, "step": 1}),
                "loop_folder": ("BOOLEAN", {"default": True}),
                "reset": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "unique_id": ("STRING", {"default": "default", "multiline": False}),
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

    @staticmethod
    def _scan_videos(path):
        files = []
        for ext in VIDEO_EXTENSIONS:
            files.extend(glob.glob(os.path.join(path, f"*{ext}")))
            files.extend(glob.glob(os.path.join(path, f"*{ext.upper()}")))
        # de-dupe (case-insensitive glob can double up on some filesystems) and sort
        files = sorted(set(files), key=lambda p: os.path.basename(p).lower())
        return files

    @staticmethod
    def _read_frame(video_path, frame_index):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"GRVideoFrameStepper: could not open video '{video_path}'")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = cap.read()
        cap.release()

        if not ok:
            return None, total_frames

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_float = frame_rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(frame_float)[None, ...]  # (1, H, W, 3)
        return tensor, total_frames

    def step(self, path, start_value, increment_by, loop_folder=True, reset=False, unique_id="default"):
        state = self._load_state()
        key = str(unique_id)
        entry = state.get(key)

        videos = self._scan_videos(path)
        if not videos:
            raise RuntimeError(f"GRVideoFrameStepper: no video files found in '{path}'")

        fresh_start = (
            reset
            or entry is None
            or entry.get("path") != path
            or entry.get("start_value") != start_value
        )

        if fresh_start:
            video_index = 0
            frame_index = start_value
        else:
            video_index = entry["video_index"]
            frame_index = entry["frame_index"] + increment_by
            # if the folder contents shrank since last run, clamp back onto it
            if video_index >= len(videos):
                video_index = 0
                frame_index = start_value

        video_path = videos[video_index]
        tensor, total_frames = self._read_frame(video_path, frame_index)

        # ran past the end of this video -> advance to the next file
        while tensor is None:
            video_index += 1
            if video_index >= len(videos):
                if not loop_folder:
                    # hold on the last valid frame of the last video
                    video_index = len(videos) - 1
                    video_path = videos[video_index]
                    tensor, total_frames = self._read_frame(video_path, max(0, total_frames - 1))
                    frame_index = max(0, total_frames - 1)
                    break
                video_index = 0
            frame_index = start_value
            video_path = videos[video_index]
            tensor, total_frames = self._read_frame(video_path, frame_index)

        state[key] = {
            "path": path,
            "start_value": start_value,
            "video_index": video_index,
            "frame_index": frame_index,
        }
        self._save_state(state)

        filename = os.path.basename(video_path)
        folder_name = os.path.basename(os.path.normpath(path))
        return (int(frame_index), float(frame_index), tensor, filename, folder_name)

    @classmethod
    def IS_CHANGED(cls, path, start_value, increment_by, loop_folder=True, reset=False, unique_id="default"):
        return float("nan")


NODE_CLASS_MAPPINGS = {
    "GRVideoFrameStepper": GRVideoFrameStepper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRVideoFrameStepper": "GR Video Frame Stepper",
}