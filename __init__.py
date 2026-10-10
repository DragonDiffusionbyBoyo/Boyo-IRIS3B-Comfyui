"""
Iris-3B ComfyUI Node Pack
==========================
Nodes
-----
  Iris3BDownloadModels – downloads weights into ComfyUI/models/ (run first)
  Iris3BModelLoader    – resolves model path, parses config
  Iris3BLoraLoader     – attaches a LoRA from ComfyUI/models/loras/ (chainable)
  Iris3BGenerate       – text-to-image generation
  Iris3BDepth          – monocular depth estimation
  Iris3BUpscale        – 4× restoration / upscaling (any image)
"""

from .downloader import Iris3BDownloadModels
from .nodes import (
    Iris3BModelLoader,
    Iris3BLoraLoader,
    Iris3BGenerate,
    Iris3BDepth,
    Iris3BUpscale,
)

NODE_CLASS_MAPPINGS = {
    "Iris3BDownloadModels": Iris3BDownloadModels,
    "Iris3BModelLoader":    Iris3BModelLoader,
    "Iris3BLoraLoader":     Iris3BLoraLoader,
    "Iris3BGenerate":       Iris3BGenerate,
    "Iris3BDepth":          Iris3BDepth,
    "Iris3BUpscale":        Iris3BUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Iris3BDownloadModels": "Iris-3B Download Models",
    "Iris3BModelLoader":    "Iris-3B Model Loader",
    "Iris3BLoraLoader":     "Iris-3B LoRA Loader",
    "Iris3BGenerate":       "Iris-3B Generate",
    "Iris3BDepth":          "Iris-3B Depth",
    "Iris3BUpscale":        "Iris-3B Upscale",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
