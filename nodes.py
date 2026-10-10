"""
Iris-3B ComfyUI Nodes
======================

  [Iris3BModelLoader] ──► [Iris3BLoraLoader] ──► [Iris3BLoraLoader] ──► [Iris3BGenerate]
                            (optional, chain as many as needed)

  [Iris3BGenerate] ──IMAGE──► [Iris3BDepth]
                  └─────────► [Iris3BUpscale]

ComfyUI image tensor convention: [B, H, W, C]  float32  0..1
Iris-3B internal convention:     [B, C, H, W]  float32  -1..1
"""

from __future__ import annotations

import os
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
# LoRA folder helper
# ---------------------------------------------------------------------------

def _get_lora_files() -> list[str]:
    """Scan ComfyUI/models/loras/ and return sorted list of .safetensors filenames."""
    try:
        import folder_paths
        lora_dir = Path(folder_paths.models_dir) / "loras"
    except ImportError:
        lora_dir = Path(__file__).resolve().parent.parent.parent / "models" / "loras"

    if not lora_dir.exists():
        return ["None"]

    files = sorted(
        p.name for p in lora_dir.rglob("*.safetensors")
    )
    return ["None"] + files if files else ["None"]


def _resolve_lora_path(filename: str) -> Optional[Path]:
    """Resolve a LoRA filename to its full path, or None if 'None'."""
    if not filename or filename == "None":
        return None
    try:
        import folder_paths
        lora_dir = Path(folder_paths.models_dir) / "loras"
    except ImportError:
        lora_dir = Path(__file__).resolve().parent.parent.parent / "models" / "loras"

    # Search recursively in case it's in a subfolder
    matches = list(lora_dir.rglob(filename))
    if matches:
        return matches[0]
    return None


# ---------------------------------------------------------------------------
# Node: Iris3BModelLoader
# ---------------------------------------------------------------------------

class Iris3BModelLoader:
    """
    Resolves model paths and parses config. Does NOT hold weights in VRAM.
    Outputs an Iris3BHandle passed down the wire to the Generate node.
    Chain Iris3BLoraLoader nodes after this to apply LoRAs.
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
                    "tooltip": "HF repo id or absolute path to a local model directory. "
                               "Leave as default if you used the Download Models node.",
                }),
            },
        }

    def load(self, source: str):
        from iris3b.config import inference_config
        from iris3b.sampling import load_for_inference

        resolved = resolve_source(source.strip())
        raw, _ = load_for_inference(resolved)
        cfg = inference_config(raw, [])
        patch_text_encoder_path(cfg)

        handle = Iris3BHandle(
            source=resolved,
            lora_path=None,
            lora_scale=1.0,
            raw_cfg=raw,
            sample_cfg=cfg,
        )
        print(f"[Iris3B Loader] source={resolved}")
        return (handle,)


# ---------------------------------------------------------------------------
# Node: Iris3BLoraLoader
# ---------------------------------------------------------------------------

class Iris3BLoraLoader:
    """
    Attaches a LoRA to an Iris3BHandle. Chain multiple nodes to stack LoRAs.
    LoRAs are merged into the generation model weights at inference time
    and discarded afterwards — the model on disk is never modified.

    Scans ComfyUI/models/loras/ automatically.
    Select 'None' to pass the model through unchanged.
    """

    CATEGORY = "Iris-3B"
    FUNCTION = "load_lora"
    RETURN_TYPES = ("IRIS3B_MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":    ("IRIS3B_MODEL", {}),
                "lora":     (_get_lora_files(), {"default": "None"}),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "LoRA merge strength. 1.0 = full, 0.0 = disabled",
                }),
            },
        }

    def load_lora(self, model: Iris3BHandle, lora: str, strength: float):
        import copy

        if lora == "None" or strength == 0.0:
            print(f"[Iris3B LoRA] skipped ({lora})")
            return (model,)

        lora_path = _resolve_lora_path(lora)
        if lora_path is None:
            raise FileNotFoundError(f"[Iris3B LoRA] not found: {lora}")

        # Return a new handle with this LoRA appended to the chain
        # We store multiple LoRAs as a list on the handle
        new_handle = copy.copy(model)
        existing = getattr(new_handle, 'lora_stack', [])
        new_handle.lora_stack = existing + [(lora_path, strength)]
        # Keep lora_path/lora_scale for backwards compat but prefer lora_stack
        new_handle.lora_path  = lora_path
        new_handle.lora_scale = strength

        print(f"[Iris3B LoRA] queued {lora}  strength={strength}")
        return (new_handle,)


# ---------------------------------------------------------------------------
# Node: Iris3BGenerate
# ---------------------------------------------------------------------------

class Iris3BGenerate:
    """
    Text-to-image generation. Loads IrisDiT + Qwen3-VL, applies any queued
    LoRAs, generates, then offloads everything from VRAM before returning.
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
                "size":      (list(_SIZES.keys()), {"default": "512×768"}),
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

        # Load generation model weights fresh
        _, weights = load_for_inference(model.source)
        with torch.device("meta"):
            dit = IrisDiT(cfg.model)
        dit.load_state_dict(weights, strict=True, assign=True)
        dit = dit.eval().to(device=device, dtype=torch.float32)
        del weights

        # Apply LoRA stack (supports multiple chained LoRA loaders)
        lora_stack = getattr(model, 'lora_stack', [])
        if lora_stack:
            for lora_path, lora_scale in lora_stack:
                apply_lora(dit, lora_path, lora_scale)
        elif model.lora_path:
            # Backwards compat: single LoRA on handle directly
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
    Monocular depth estimation. Accepts any image.
    Returns a colourised depth IMAGE and a single-channel MASK (0=near, 1=far).
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

        colourised  = colorize(depth_array)
        depth_comfy = _pil_to_comfy(colourised)

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
    Accepts ANY image — not just Iris-3B generated outputs.

    Fine-tuned on 1024×1024 crops (4× of 256×256 inputs).
    Processes via overlapping 1024×1024 tiles with Gaussian blend.
    Compute scales with output area.
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
                    "tooltip": "Upscale factor. 4× is optimal (trained resolution)",
                }),
                "color_fix": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Wavelet colour fix — keeps model detail but takes "
                               "colour/illumination from the bicubic reference. "
                               "Disable to let the model freely recolour",
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
                    "tooltip": "fit_budget: max short side before upscaling. "
                               "At 4×: 1024 short → 4096 short output",
                }),
                "long_side_cap": ("INT", {
                    "default": 2048, "min": 256, "max": 8192, "step": 64,
                    "tooltip": "fit_budget: max long side before upscaling. "
                               "At 4×: 2048 long → 8192 long output",
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
