"""
Iris-3B Model Downloader Node
==============================
Downloads Iris-3B weights and the Qwen3-VL-4B-Instruct text encoder
into ComfyUI managed folders. Writes manifest.json files so subsequent
runs never touch the network.

Final layout:
    ComfyUI/models/iris3b/
        manifest.json
        config.yaml
        model.safetensors              ← generation model (~12 GB)
        depth/
            config.yaml
            model.safetensors          ← depth fine-tune (~12 GB)
            empty_prompt.safetensors
        upscaler/
            config.yaml
            model.safetensors          ← upscaler fine-tune (~12 GB)
            empty_prompt.safetensors

    ComfyUI/models/text_encoders/Qwen3-VL-4B-Instruct/
        manifest.json
        config.json
        generation_config.json
        tokenizer.json
        tokenizer_config.json
        vocab.json
        merges.txt
        preprocessor_config.json
        chat_template.jinja
        model.safetensors.index.json
        model-00001-of-00002.safetensors   ← ~4 GB each
        model-00002-of-00002.safetensors
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .model_utils import get_iris3b_models_dir

HF_REPO_IRIS   = "speridlabs/iris-3b"
HF_REPO_QWEN   = "Qwen/Qwen3-VL-4B-Instruct"

# ---------------------------------------------------------------------------
# File lists
# ---------------------------------------------------------------------------

_IRIS3B_FILES = [
    "config.yaml",
    "model.safetensors",
    "depth/config.yaml",
    "depth/model.safetensors",
    "depth/empty_prompt.safetensors",
    "upscaler/config.yaml",
    "upscaler/model.safetensors",
    "upscaler/empty_prompt.safetensors",
]

# Everything needed to run the text encoder locally
_QWEN_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_iris3b_models_dir() -> Path:
    try:
        import folder_paths
        base = Path(folder_paths.models_dir)
    except ImportError:
        base = Path(__file__).resolve().parent.parent.parent / "models"
    d = base / "iris3b"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_qwen_dir() -> Path:
    try:
        import folder_paths
        base = Path(folder_paths.models_dir)
    except ImportError:
        base = Path(__file__).resolve().parent.parent.parent / "models"
    d = base / "text_encoders" / "Qwen3-VL-4B-Instruct"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _check_files(base: Path, files: list[str]) -> list[str]:
    """Return list of missing file paths."""
    return [f for f in files if not (base / f).exists()]


def _write_manifest(base: Path, repo: str, files: list[str]) -> None:
    manifest = {"repo": repo, "files": {}}
    for f in files:
        p = base / f
        if p.exists():
            manifest["files"][f] = {"size": p.stat().st_size}
    (base / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[Iris3B Downloader] manifest written → {base / 'manifest.json'}")


def check_iris3b() -> tuple[bool, list[str]]:
    base = get_iris3b_models_dir()
    missing = _check_files(base, _IRIS3B_FILES)
    complete = len(missing) == 0 and (base / "manifest.json").exists()
    return complete, missing


def check_qwen() -> tuple[bool, list[str]]:
    base = get_qwen_dir()
    missing = _check_files(base, _QWEN_FILES)
    complete = len(missing) == 0 and (base / "manifest.json").exists()
    return complete, missing


def check_all() -> tuple[bool, list[str]]:
    _, m1 = check_iris3b()
    _, m2 = check_qwen()
    missing = (
        [f"iris3b/{f}" for f in m1] +
        [f"text_encoders/Qwen3-VL-4B-Instruct/{f}" for f in m2]
    )
    return len(missing) == 0, missing


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_file(repo_id: str, rel_path: str, dest: Path, token) -> None:
    from huggingface_hub import hf_hub_download
    parts = rel_path.split("/")
    filename = parts[-1]
    subfolder = "/".join(parts[:-1]) if len(parts) > 1 else None
    cached = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
        token=token,
        local_dir=None,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, dest)


def download_iris3b(token=None) -> list[str]:
    base = get_iris3b_models_dir()
    missing = _check_files(base, _IRIS3B_FILES)
    errors = []
    for f in missing:
        try:
            print(f"[Iris3B Downloader] iris3b: fetching {f} ...")
            _download_file(HF_REPO_IRIS, f, base / f, token)
            print(f"[Iris3B Downloader] ✓ {f}")
        except Exception as e:
            errors.append(f"iris3b/{f}: {e}")
            print(f"[Iris3B Downloader] ✗ {f}: {e}")
    if not errors:
        _write_manifest(base, HF_REPO_IRIS, _IRIS3B_FILES)
    return errors


def _download_qwen(token=None) -> list[str]:
    base = get_qwen_dir()
    missing = _check_files(base, _QWEN_FILES)
    errors = []
    for f in missing:
        try:
            print(f"[Iris3B Downloader] qwen: fetching {f} ...")
            _download_file(HF_REPO_QWEN, f, base / f, token)
            print(f"[Iris3B Downloader] ✓ {f}")
        except Exception as e:
            errors.append(f"Qwen3-VL/{f}: {e}")
            print(f"[Iris3B Downloader] ✗ {f}: {e}")
    if not errors:
        _write_manifest(base, HF_REPO_QWEN, _QWEN_FILES)
    return errors


# ---------------------------------------------------------------------------
# ComfyUI Node
# ---------------------------------------------------------------------------

class Iris3BDownloadModels:
    """
    Downloads Iris-3B weights and Qwen3-VL-4B-Instruct text encoder
    into ComfyUI/models/. Run this once before using any other Iris-3B node.
    Skips any files already present. Writes manifest.json when complete.
    """

    CATEGORY = "Iris-3B"
    FUNCTION = "download"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "download_iris3b": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Download the Iris-3B generation, depth and upscaler models (~36 GB)",
                }),
                "download_qwen": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Download Qwen3-VL-4B-Instruct text encoder (~8 GB)",
                }),
                "check_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Only report what is missing, do not download",
                }),
            },
            "optional": {
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "HuggingFace token – only needed for private/gated repos",
                }),
            },
        }

    def download(
        self,
        download_iris3b: bool = True,
        download_qwen: bool = True,
        check_only: bool = False,
        hf_token: str = "",
    ):
        token = hf_token.strip() or os.environ.get("HF_TOKEN") or None
        lines = []

        # ── Status report ──
        iris_ok, iris_missing = check_iris3b()
        qwen_ok,  qwen_missing = check_qwen()

        lines.append(f"Iris-3B models:  {'✓ complete' if iris_ok else f'{len(iris_missing)} files missing'}")
        lines.append(f"Qwen3-VL-4B:     {'✓ complete' if qwen_ok  else f'{len(qwen_missing)} files missing'}")

        if check_only or (iris_ok and qwen_ok):
            if iris_missing:
                lines += [f"  missing: {f}" for f in iris_missing]
            if qwen_missing:
                lines += [f"  missing: {f}" for f in qwen_missing]
            return ("\n".join(lines),)

        # ── Download ──
        errors = []
        if download_iris3b and not iris_ok:
            lines.append(f"\nDownloading Iris-3B → {get_iris3b_models_dir()}")
            errors += download_iris3b_fn(token)

        if download_qwen and not qwen_ok:
            lines.append(f"\nDownloading Qwen3-VL-4B-Instruct → {get_qwen_dir()}")
            errors += _download_qwen(token)

        if errors:
            lines.append("\nErrors:")
            lines += errors
        else:
            lines.append("\n✓ All downloads complete")

        return ("\n".join(lines),)

