"""
Iris-3B ComfyUI Node Pack
==========================
Wraps the speridlabs/iris-3b pipeline as ComfyUI custom nodes.

Nodes
-----
  Iris3BDownloadModels – downloads weights into ComfyUI/models/iris3b/ (run first)
  Iris3BModelLoader    – resolves model path, parses config, optional LoRA
  Iris3BGenerate       – text-to-image generation
  Iris3BDepth          – monocular depth estimation
  Iris3BUpscale        – 4× restoration / upscaling

VRAM strategy
-------------
Each node loads its model, runs, then offloads to CPU and calls
torch.cuda.empty_cache() before returning. Only one 12 GB model
is in VRAM at any time.
"""

from .downloader import Iris3BDownloadModels
from .nodes import (
    Iris3BModelLoader,
    Iris3BGenerate,
    Iris3BDepth,
    Iris3BUpscale,
)

NODE_CLASS_MAPPINGS = {
    "Iris3BDownloadModels": Iris3BDownloadModels,
    "Iris3BModelLoader":    Iris3BModelLoader,
    "Iris3BGenerate":       Iris3BGenerate,
    "Iris3BDepth":          Iris3BDepth,
    "Iris3BUpscale":        Iris3BUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Iris3BDownloadModels": "Iris-3B Download Models",
    "Iris3BModelLoader":    "Iris-3B Model Loader",
    "Iris3BGenerate":       "Iris-3B Generate",
    "Iris3BDepth":          "Iris-3B Depth",
    "Iris3BUpscale":        "Iris-3B Upscale",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
