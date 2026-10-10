"""
Iris-3B ComfyUI Nodes
======================
Four nodes wired as:

  [Iris3BModelLoader] ──model──► [Iris3BGenerate] ──IMAGE──► [Iris3BDepth]
                                                   └─────────► [Iris3BUpscale]

ComfyUI image tensor convention: [B, H, W, C]  float32  0..1
Iris-3B internal convention:     [B, C, H, W]  float32  -1..1
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .model_utils import (
    Iris3BHandle,
    apply_lora,
    free_vram,
    get_device,
    get_subfolder,
    patch_text_encoder_path,
    resolve_source,
)

# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def _pil_to_comfy(image: Image.Image) -> torch.Tensor:
    """PIL RGB → [1, H, W, C] float32 0..1"""
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _comfy_to_pil(tensor: torch.Tensor) -> Image.Image:
    """[1, H, W, C] float32 0..1 → PIL RGB"""
    arr = (tensor[0].clamp(0.0, 1.0).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def iris_to_comfy(tensor: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] float32 -1..1 → [B, H, W, C] float32 0..1"""
    scaled = (tensor.float().clamp(-1, 1) + 1.0) * 0.5
    return scaled.permute(0, 2, 3, 1).cpu()


def comfy_to_iris(tensor: torch.Tensor) -> torch.Tensor:
    """[B, H, W, C] float32 0..1 → [B, C, H, W] float32 -1..1"""
    chw = tensor.permute(0, 3, 1, 2)
    return chw.float() * 2.0 - 1.0


# Supported generation size buckets
_SIZES: dict[str, tuple[int, int]] = {
    "256×256":   (256,  256),
    "384×384":   (384,  384),
    "512×512":   (512,  512),
    "512×768":   (512,  768),
    "768×512":   (768,  512),
    "768×1344":  (768,  1344),
    "832×1280":  (832,  1280),
    "896×1152":  (896,  1152),
    "1024×1024": (1024, 1024),
    "1152×896":  (1152, 896),
    "1280×832":  (1280, 832),
    "1344×768":  (1344, 768),
}


# ---------------------------------------------------------------------------
# Node: Iris3BModelLoader
# ---------------------------------------------------------------------------

class Iris3BModelLoader:
    """
    Resolves model paths and parses config. Does NOT hold weights in VRAM.
    Outputs an Iris3BHandle passed to the Generate node on a wire.
    """

    CATEGORY = "Iris-3B"
    FUNCTION = "load"
    RETURN_TYPES = ("IRIS3B_MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("STRING", {
                    "default": "speridlabs/iris-3b",
                    "multiline": False,
                    "tooltip": "HF repo id or absolute path to a local model directory",
                }),
            },
            "optional": {
                "lora_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to an AI-Toolkit / Ostris LoRA .safetensors (leave blank for none)",
                }),
                "lora_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "LoRA merge strength",
                }),
            },
        }

    def load(self, source: str, lora_path: str = "", lora_scale: float = 1.0):
        from iris3b.config import inference_config
        from iris3b.sampling import load_for_inference

        resolved = resolve_source(source.strip())
        lp = Path(lora_path.strip()) if lora_path.strip() else None
        if lp and not lp.exists():
            raise FileNotFoundError(f"[Iris3B] LoRA not found: {lp}")

        raw, _ = load_for_inference(resolved)
        cfg = inference_config(raw, [])
        patch_text_encoder_path(cfg)

        handle = Iris3BHandle(
            source=resolved,
            lora_path=lp,
            lora_scale=lora_scale,
            raw_cfg=raw,
            sample_cfg=cfg,
        )
        print(f"[Iris3B Loader] source={resolved}  lora={lp}  scale={lora_scale}")
        return (handle,)


# ---------------------------------------------------------------------------
# Node: Iris3BGenerate
# ---------------------------------------------------------------------------

class Iris3BGenerate:
    """
    Text-to-image generation. Loads IrisDiT + Qwen3-VL, generates,
    then offloads both from VRAM before returning.
    """

    CATEGORY = "Iris-3B"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":     ("IRIS3B_MODEL", {}),
                "prompt":    ("STRING", {"multiline": True, "default": ""}),
                "size":      (list(_SIZES.keys()), {"default": "512×512"}),
                "steps":     ("INT",   {"default": 50,  "min": 10,  "max": 150}),
                "cfg_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "seed":      ("INT",   {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    def generate(
        self,
        model: Iris3BHandle,
        prompt: str,
        size: str,
        steps: int,
        cfg_scale: float,
        seed: int,
        negative_prompt: str = "",
    ):
        import transformers
        import iris3b.text  # noqa: F401 – registers encoders
        from iris3b.models.dit import IrisDiT
        from iris3b.registry import TEXT_ENCODERS
        from iris3b.sampling import generate as _generate, load_for_inference

        device = get_device()
        width, height = _SIZES[size]
        cfg = model.sample_cfg

        print(f"[Iris3B Generate] {width}×{height}  steps={steps}  cfg={cfg_scale}  seed={seed}")

        # Load generation model
        _, weights = load_for_inference(model.source)
        with torch.device("meta"):
            dit = IrisDiT(cfg.model)
        dit.load_state_dict(weights, strict=True, assign=True)
        dit = dit.eval().to(device=device, dtype=torch.float32)
        del weights

        if model.lora_path:
            apply_lora(dit, model.lora_path, model.lora_scale)

        # Monkey-patch from_pretrained to use local path while keeping
        # cfg.text_encoder.pretrained as the HF slug (used for cache filenames)
        _real_tok   = transformers.AutoTokenizer.from_pretrained
        _real_model = transformers.Qwen3VLForConditionalGeneration.from_pretrained
        local_te    = getattr(cfg.text_encoder, '_local_path', None)
        if local_te:
            transformers.AutoTokenizer.from_pretrained = \
                lambda p, **kw: _real_tok(local_te, **kw)
            transformers.Qwen3VLForConditionalGeneration.from_pretrained = \
                lambda p, **kw: _real_model(local_te, **kw)

        text_encoder = TEXT_ENCODERS.build(cfg.text_encoder.name, cfg.text_encoder, device=device)

        if local_te:
            transformers.AutoTokenizer.from_pretrained = _real_tok
            transformers.Qwen3VLForConditionalGeneration.from_pretrained = _real_model

        generator = torch.Generator(device=device).manual_seed(seed)
        images = _generate(
            dit, text_encoder,
            [prompt],
            height=height, width=width,
            steps=steps,
            order=cfg.sample.order,
            cfg_scale=cfg_scale,
            cfg_interval=tuple(cfg.sample.cfg_interval),
            shift=cfg.flow.shift,
            negative_prompt=negative_prompt,
            generator=generator,
            device=device,
            num_train_timesteps=cfg.flow.num_train_timesteps,
            prediction=cfg.flow.prediction,
        )

        out = iris_to_comfy(images)
        free_vram(dit, text_encoder)
        print("[Iris3B Generate] done – VRAM released")
        return (out,)


# ---------------------------------------------------------------------------
# Node: Iris3BDepth
# ---------------------------------------------------------------------------

class Iris3BDepth:
    """
    Monocular depth estimation. Returns a colourised depth IMAGE and a
    single-channel MASK (0=near, 1=far) normalised to 0..1.
    """

    CATEGORY = "Iris-3B"
    FUNCTION = "depth"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("depth_image", "depth_mask")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":    ("IMAGE", {}),
                "max_side": ("INT", {
                    "default": 1024, "min": 0, "max": 4096,
                    "tooltip": "Cap long side before depth pass (0 = native resolution)",
                }),
            },
        }

    def depth(self, image: torch.Tensor, max_side: int = 1024):
        from iris3b.downstream.depth import DepthPredictor, colorize

        device = get_device()
        pil = _comfy_to_pil(image)
        print(f"[Iris3B Depth] input={pil.width}×{pil.height}")

        source = get_subfolder("depth")
        predictor = DepthPredictor(source, device=device)
        depth_array = predictor(pil, max_side=max_side)

        colourised   = colorize(depth_array)
        depth_comfy  = _pil_to_comfy(colourised)

        d = depth_array.astype(np.float32)
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
        mask_tensor = torch.from_numpy(d).unsqueeze(0)

        free_vram(predictor.model)
        print("[Iris3B Depth] done – VRAM released")
        return (depth_comfy, mask_tensor)


# ---------------------------------------------------------------------------
# Node: Iris3BUpscale
# ---------------------------------------------------------------------------

class Iris3BUpscale:
    """
    Iris-3B one-step 4× restoration and upscaler.

    Accepts ANY image — not just Iris-3B generated outputs. Useful as a
    general-purpose upscaler for any source material.

    The upscaler was fine-tuned on 1024×1024 crops (4× of 256×256 inputs)
    and processes via overlapping 1024×1024 tiles with a Gaussian blend window.
    Compute scales with output area — a 512×512 input → 2048×2048 output = 9 tiles.

    fit_budget optionally downscales the input before upscaling to cap tile count.
    Disable it if you want to upscale at native resolution and have the VRAM.

    Tile count estimate: (ceil(W*scale/512) * ceil(H*scale/512)) roughly.
    """

    CATEGORY = "Iris-3B"
    FUNCTION = "upscale"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "scale": ("FLOAT", {
                    "default": 4.0, "min": 1.0, "max": 8.0, "step": 0.5,
                    "tooltip": "Upscale factor. Model was trained at 4×; other values work but 4× is optimal",
                }),
                "color_fix": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Wavelet colour fix – keeps high-frequency detail from the model "
                               "but takes colour/illumination from the bicubic reference. "
                               "Disable if you want the model to freely change colours",
                }),
                "fit_budget": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Downscale input before upscaling to cap tile count and VRAM. "
                               "Disable to upscale at native resolution",
                }),
            },
            "optional": {
                "short_side_cap": ("INT", {
                    "default": 1024, "min": 256, "max": 4096, "step": 64,
                    "tooltip": "fit_budget: max short side of input before upscaling. "
                               "At 4× scale: 1024 short side → 4096 short side output",
                }),
                "long_side_cap": ("INT", {
                    "default": 2048, "min": 256, "max": 8192, "step": 64,
                    "tooltip": "fit_budget: max long side of input before upscaling. "
                               "At 4× scale: 2048 long side → 8192 long side output",
                }),
            },
        }

    def upscale(
        self,
        image: torch.Tensor,
        scale: float = 4.0,
        color_fix: bool = True,
        fit_budget: bool = False,
        short_side_cap: int = 1024,
        long_side_cap: int = 2048,
    ):
        from iris3b.downstream.restoration import Restorer, fit_budget as _fit_budget

        device = get_device()
        pil = _comfy_to_pil(image)

        if fit_budget:
            pil = _fit_budget(pil, short_side=short_side_cap, long_side=long_side_cap)

        # Tile count estimate for console info
        out_w = round(pil.width  * scale)
        out_h = round(pil.height * scale)
        tiles_w = max(1, (out_w - 1024) // 512 + 1) if out_w > 1024 else 1
        tiles_h = max(1, (out_h - 1024) // 512 + 1) if out_h > 1024 else 1
        print(f"[Iris3B Upscale] input={pil.width}×{pil.height}  "
              f"output≈{out_w}×{out_h}  "
              f"tiles≈{tiles_w * tiles_h}  "
              f"scale={scale}×  color_fix={color_fix}")

        source = get_subfolder("upscaler")
        restorer = Restorer(source, device=device)
        result = restorer(pil, scale=scale, color_fix=color_fix)

        out = _pil_to_comfy(result)
        free_vram(restorer.model)
        print(f"[Iris3B Upscale] done  {pil.width}×{pil.height} → {result.width}×{result.height} – VRAM released")
        return (out,)
