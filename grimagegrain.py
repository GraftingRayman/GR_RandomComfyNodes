import torch
import numpy as np
from comfy.utils import ProgressBar


class GRLogger:
    """Minimal logger matching existing GR node conventions."""
    PREFIX = "[GRImageGrain]"

    @classmethod
    def info(cls, msg):
        print(f"{cls.PREFIX} {msg}")

    @classmethod
    def warn(cls, msg):
        print(f"{cls.PREFIX} WARNING: {msg}")

    @classmethod
    def error(cls, msg):
        print(f"{cls.PREFIX} ERROR: {msg}")


class GRImageGrain:
    """
    Adds film-grain style noise to an image batch after generation.
    Useful for restoring texture lost to over-smoothing (e.g. Qwen Image Edit
    skin-smoothing artifacts), or for general stylistic grain.

    - Seed-driven, deterministic per image index (mirrors GRLoraLoader's
      seed-based determinism pattern).
    - Supports monochrome (luminance-only) or per-channel color grain.
    - Optional highlight/shadow rolloff so grain doesn't blow out pure
      white/black regions (matches how real film grain behaves).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 0.12, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "For masking artifacts before a denoising upscaler, "
                               "use higher values (0.12-0.25) than a stylistic-grain pass"
                }),
                "grain_size": ("FLOAT", {
                    "default": 1.0, "min": 0.25, "max": 8.0, "step": 0.05,
                    "tooltip": "Higher = coarser/larger grain clumps"
                }),
                "monochrome": ("BOOLEAN", {"default": True}),
                "highlight_shadow_rolloff": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Leave off if masking midtone wrinkles/creases - "
                               "rolloff protects exactly that tonal range"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_grain"
    CATEGORY = "GraftingRayman/Image"

    def _generate_grain(self, h, w, channels, grain_size, rng):
        """Generate noise at a downscaled resolution then upsample, to
        control apparent grain size (mirrors how real film grain scales)."""
        small_h = max(1, int(h / grain_size))
        small_w = max(1, int(w / grain_size))

        noise = rng.standard_normal((small_h, small_w, channels)).astype(np.float32)

        if grain_size != 1.0:
            noise_t = torch.from_numpy(noise).permute(2, 0, 1).unsqueeze(0)
            noise_t = torch.nn.functional.interpolate(
                noise_t, size=(h, w), mode="bilinear", align_corners=False
            )
            noise = noise_t.squeeze(0).permute(1, 2, 0).numpy()

        return noise

    def apply_grain(self, image, strength, grain_size, monochrome,
                     highlight_shadow_rolloff, seed, mask=None):
        if strength <= 0.0:
            GRLogger.info("Strength is 0, returning image unchanged.")
            return (image,)

        batch, h, w, c = image.shape

        mask_np_batch = None
        if mask is not None:
            if mask.shape[0] not in (1, batch):
                GRLogger.warn(
                    f"Mask batch size ({mask.shape[0]}) doesn't match image "
                    f"batch size ({batch}); mask index will be clamped."
                )
            mask_np_batch = mask.cpu().numpy()

        out = torch.empty_like(image)
        pbar = ProgressBar(batch)

        for i in range(batch):
            img_seed = (seed + i) & 0xffffffff
            rng = np.random.default_rng(img_seed)

            channels = 1 if monochrome else c
            noise = self._generate_grain(h, w, channels, grain_size, rng)

            if monochrome:
                noise = np.repeat(noise, c, axis=2)

            img_np = image[i].cpu().numpy()

            if highlight_shadow_rolloff:
                # Reduce grain visibility near pure black/white, like real film
                luminance = img_np.mean(axis=2, keepdims=True)
                rolloff = 1.0 - np.abs(luminance - 0.5) * 2.0
                rolloff = np.clip(rolloff, 0.15, 1.0)
                noise = noise * rolloff

            noise = noise * strength

            if mask_np_batch is not None:
                m_idx = min(i, mask_np_batch.shape[0] - 1)
                m = mask_np_batch[m_idx]
                if m.shape != (h, w):
                    m_t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0)
                    m_t = torch.nn.functional.interpolate(
                        m_t, size=(h, w), mode="bilinear", align_corners=False
                    )
                    m = m_t.squeeze(0).squeeze(0).numpy()
                m = m[..., None]  # broadcast over channels
                noise = noise * m

            grained = np.clip(img_np + noise, 0.0, 1.0)

            out[i] = torch.from_numpy(grained.astype(np.float32))
            pbar.update(1)

        GRLogger.info(
            f"Applied grain to {batch} image(s): strength={strength}, "
            f"grain_size={grain_size}, monochrome={monochrome}, "
            f"masked={mask is not None}, seed={seed}"
        )

        return (out,)


NODE_CLASS_MAPPINGS = {
    "GRImageGrain": GRImageGrain,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRImageGrain": "GR Image Grain",
}