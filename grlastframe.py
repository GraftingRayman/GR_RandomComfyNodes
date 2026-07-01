import torch
import numpy as np
import cv2
import os


class GRLastFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "images": ("IMAGE",),
                "video_path": ("STRING", {"default": ""}),
                "frame_index": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 99999,
                    "step": 1,
                    "tooltip": "Frame to extract. -1 = last frame (default behaviour)."
                }),
                "frame_start": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99999,
                    "step": 1,
                    "tooltip": "Start frame for range output (inclusive)."
                }),
                "frame_end": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 99999,
                    "step": 1,
                    "tooltip": "End frame for range output (inclusive). -1 = last frame."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "image_range")
    FUNCTION = "get_last_frame"
    CATEGORY = "GR Nodes"

    def _resolve_index(self, frame_index: int, total: int) -> int:
        if frame_index < 0:
            return total - 1
        return min(frame_index, total - 1)

    def get_last_frame(self, images=None, video_path="", frame_index=-1, frame_start=0, frame_end=-1):

        # -----------------------------------
        # CASE 1: IMAGE batch input
        # -----------------------------------
        if images is not None:
            total = images.shape[0]

            # Single frame output
            idx = self._resolve_index(frame_index, total)
            single = images[idx : idx + 1].clone()

            # Range output
            start = max(0, frame_start)
            end = self._resolve_index(frame_end, total)
            if start > end:
                start = end
            image_range = images[start : end + 1].clone()

            return (single, image_range)

        # -----------------------------------
        # CASE 2: MP4 video input
        # -----------------------------------
        if video_path and os.path.isfile(video_path):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file.")

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0:
                cap.release()
                raise ValueError("Video contains no frames.")

            # Resolve indices
            idx = self._resolve_index(frame_index, frame_count)
            start = max(0, frame_start)
            end = self._resolve_index(frame_end, frame_count)
            if start > end:
                start = end

            def read_frame(cap, pos):
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f"Failed to read frame {pos}.")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                return torch.from_numpy(frame)[None, ...]

            # Single frame
            single = read_frame(cap, idx)

            # Range frames
            range_frames = []
            for i in range(start, end + 1):
                range_frames.append(read_frame(cap, i))

            cap.release()

            image_range = torch.cat(range_frames, dim=0)
            return (single, image_range)

        # -----------------------------------
        # Nothing provided
        # -----------------------------------
        raise ValueError("Provide either images or a valid video_path.")


NODE_CLASS_MAPPINGS = {
    "GRLastFrame": GRLastFrame
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRLastFrame": "GR Last Frame"
}