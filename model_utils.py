"""
Shared utilities: model path resolution, LoRA merging, VRAM cleanup,
and the Iris3BHandle that travels between nodes on ComfyUI wires.

Model storage priority (checked in order):
  1. ComfyUI/models/iris3b/          ← preferred, managed by downloader node
  2. IRIS3B_SOURCE env var            ← power-user override
  3. iris-3b/  sibling clone          ← developer workflow
  4. HF Hub snapshot_download         ← last resort fallback
"""

from __future__ import annotations

import gc
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Make upstream importable – looks for iris-3b as a sibling of this pack,
# or falls back to whatever is already on sys.path (pip installed package).
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
for _candidate in (
    _HERE / "iris-3b" / "src",              # submodule inside the pack
    _HERE.parent / "iris-3b" / "src",       # sibling of custom_nodes/Boyo-IRIS3B
    _HERE.parent.parent / "iris-3b" / "src",# sibling of custom_nodes/
):
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break


# ---------------------------------------------------------------------------
# ComfyUI models directories
# ---------------------------------------------------------------------------

def get_iris3b_models_dir() -> Path:
    """Return ComfyUI/models/iris3b/, creating it if needed."""
    try:
        import folder_paths
        base = Path(folder_paths.models_dir)
    except ImportError:
        base = _HERE.parent.parent / "models"
    d = base / "iris3b"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_qwen_dir() -> Path:
    """Return ComfyUI/models/text_encoders/Qwen3-VL-4B-Instruct/, creating if needed."""
    try:
        import folder_paths
        base = Path(folder_paths.models_dir)
    except ImportError:
        base = _HERE.parent.parent / "models"
    d = base / "text_encoders" / "Qwen3-VL-4B-Instruct"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_local_qwen_path() -> Optional[Path]:
    """Return the local Qwen path if weights are present, else None."""
    p = get_qwen_dir()
    if (p / "config.json").exists() and (p / "model-00001-of-00002.safetensors").exists():
        return p
    return None


# ---------------------------------------------------------------------------
# VRAM helpers
# ---------------------------------------------------------------------------

def free_vram(*things) -> None:
    for t in things:
        if t is None:
            continue
        try:
            t.to("cpu")
        except Exception:
            pass
    del things
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model path resolution
# ---------------------------------------------------------------------------

def get_local_model_path() -> Optional[Path]:
    """
    Return the local iris3b model directory if it exists and has a manifest,
    otherwise None.
    """
    # 1. ComfyUI models folder
    p = get_iris3b_models_dir()
    if (p / "manifest.json").exists() and (p / "model.safetensors").exists():
        return p

    # 2. IRIS3B_SOURCE env override
    env = os.environ.get("IRIS3B_SOURCE", "").strip()
    if env:
        p2 = Path(env)
        if p2.is_dir() and (p2 / "model.safetensors").exists():
            return p2

    # 3. iris-3b sibling clone (developer workflow, exported directory)
    for candidate in (
        _HERE / "iris-3b",
        _HERE.parent / "iris-3b",
        _HERE.parent.parent / "iris-3b",
    ):
        if candidate.is_dir() and (candidate / "model.safetensors").exists():
            return candidate

    return None


def resolve_source(source: str) -> Path:
    """
    Resolve model source to a local Path.
    If source looks like a local path and exists, use it directly.
    Otherwise check our managed location first, then fall back to HF download.
    """
    # Explicit local path
    p = Path(source)
    if p.is_dir() and (p / "model.safetensors").exists():
        return p

    # Our managed ComfyUI models location
    local = get_local_model_path()
    if local:
        return local

    # HF Hub download as last resort
    import os as _os
    from huggingface_hub import snapshot_download
    print(f"[Iris3B] Downloading from HF Hub: {source}")
    print(f"[Iris3B] Tip: use the 'Iris-3B Download Models' node to download to ComfyUI/models/iris3b/ first")
    downloaded = snapshot_download(source, token=_os.environ.get("HF_TOKEN"))
    return Path(downloaded)


def get_subfolder(name: str) -> str:
    """
    Resolve a sub-model path (depth or upscaler).
    Checks managed ComfyUI location first, then sibling clone, then HF Hub string.
    """
    # 1. ComfyUI models folder
    p = get_iris3b_models_dir() / name
    if p.is_dir() and (p / "model.safetensors").exists():
        return str(p)

    # 2. IRIS3B_SOURCE env override
    env = os.environ.get("IRIS3B_SOURCE", "").strip()
    if env:
        p2 = Path(env) / name
        if p2.is_dir() and (p2 / "model.safetensors").exists():
            return str(p2)

    # 3. Sibling iris-3b clone
    for candidate in (
        _HERE / "iris-3b" / name,
        _HERE.parent / "iris-3b" / name,
        _HERE.parent.parent / "iris-3b" / name,
    ):
        if candidate.is_dir() and (candidate / "model.safetensors").exists():
            return str(candidate)

    # 4. HF Hub string fallback
    return f"speridlabs/iris-3b/{name}"


# ---------------------------------------------------------------------------
# LoRA merger (AI-Toolkit / Ostris safetensors format)
# ---------------------------------------------------------------------------

def apply_lora(model: torch.nn.Module, lora_path: str | Path, scale: float = 1.0) -> None:
    """
    Merge a LoRA into model weights in-place.
    W_merged = W_base + scale * (B @ A)
    Supports lora_A/lora_B and lora_down/lora_up naming conventions.
    """
    from safetensors.torch import load_file

    adapter = load_file(str(lora_path))

    # Normalise to lora_A / lora_B
    normalised: dict[str, torch.Tensor] = {}
    for k, v in adapter.items():
        k2 = (k.replace(".lora_down.weight", ".lora_A.weight")
               .replace(".lora_up.weight",   ".lora_B.weight"))
        normalised[k2] = v

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in normalised.items():
        if ".lora_A.weight" in k:
            base = k.replace(".lora_A.weight", ".weight")
            pairs.setdefault(base, {})["A"] = v
        elif ".lora_B.weight" in k:
            base = k.replace(".lora_B.weight", ".weight")
            pairs.setdefault(base, {})["B"] = v

    state = model.state_dict()
    applied = 0
    skipped = 0
    for base_key, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            continue
        if base_key not in state:
            skipped += 1
            continue
        W = state[base_key].float()
        A = ab["A"].float()
        B = ab["B"].float()
        delta = scale * (B @ A)
        if delta.shape != W.shape:
            delta = delta.T
        state[base_key] = (W + delta).to(state[base_key].dtype)
        applied += 1

    model.load_state_dict(state, strict=True)
    print(f"[Iris3B LoRA] applied {applied} pairs, skipped {skipped}  ({lora_path})")


# ---------------------------------------------------------------------------
# Handle
# ---------------------------------------------------------------------------

def patch_text_encoder_path(cfg: object) -> None:
    local = get_local_qwen_path()
    if local:
        try:
            # Keep pretrained as the HF slug - upstream uses it to build
            # the null embed cache filename, backslashes in a Windows path
            # corrupt it. We store the real path separately and monkey-patch
            # from_pretrained in the text encoder module instead.
            cfg.text_encoder._local_path = str(local)
            null_dir = get_iris3b_models_dir() / "null_embeds"
            null_dir.mkdir(parents=True, exist_ok=True)
            cfg.text_encoder.null_embed_dir = str(null_dir)
            print(f"[Iris3B] text encoder local path → {local}")
            print(f"[Iris3B] null embed cache → {null_dir}")
        except Exception as e:
            print(f"[Iris3B] WARNING: could not patch text encoder path: {e}")
    else:
        print(f"[Iris3B] WARNING: Qwen3-VL not found locally at {get_qwen_dir()}")
        print(f"[Iris3B] Use the 'Iris-3B Download Models' node to download it first")      
        
@dataclass
class Iris3BHandle:
    """
    Lightweight config token passed between Loader → Generate on a wire.
    No weights stored – loaded fresh per execution and freed after.
    """
    source: Path
    lora_path: Optional[Path]
    lora_scale: float
    raw_cfg: dict = field(default_factory=dict)
    sample_cfg: object = None
