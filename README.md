I have a pug dog to maintain and he does not eat or drink or play with fresh air!

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/E1E01W3X6E)

# Boyo-IRIS3B — ComfyUI Node Pack

ComfyUI custom nodes wrapping [speridlabs/iris-3b](https://github.com/speridlabs/iris-3b).
Sequential VRAM management — only one 12 GB model loaded at a time.
The iris-3b library is vendored inside this repo — no submodules, no separate clones.

---

## Nodes

| Node | Purpose |
|------|---------|
| **Iris-3B Download Models** | Downloads weights into `ComfyUI/models/` — run this first |
| **Iris-3B Model Loader** | Resolves model path, parses config, optional LoRA |
| **Iris-3B Generate** | Text-to-image generation |
| **Iris-3B Depth** | Monocular depth estimation — any image |
| **Iris-3B Upscale** | One-step 4× restoration and upscaling — any image |

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/BoyoDragon/Boyo-IRIS3B-Comfyui
cd Boyo-IRIS3B-Comfyui
pip install -r requirements.txt
pip install -e iris-3b/
```

Restart ComfyUI.

---

## First run — download models

Add an **Iris-3B Download Models** node to your workflow and queue it once.
It downloads everything into ComfyUI's models folder:

```
ComfyUI/models/
    iris3b/
        manifest.json
        config.yaml
        model.safetensors              (~12 GB — generation model)
        depth/
            model.safetensors          (~12 GB — depth fine-tune)
        upscaler/
            model.safetensors          (~12 GB — upscaler fine-tune)
    text_encoders/
        Qwen3-VL-4B-Instruct/
            model-00001-of-00002.safetensors   (~4 GB each)
            model-00002-of-00002.safetensors
            ... (tokeniser and config files)
```

Total: ~40 GB. Subsequent runs skip the download entirely (manifest check).

---

## Typical workflow

```
[Iris-3B Model Loader]
        │ IRIS3B_MODEL
        ▼
[Iris-3B Generate] ──── IMAGE ────┬──► [Preview Image]
                                  ├──► [Iris-3B Depth]   ──► [Preview Image]
                                  └──► [Iris-3B Upscale] ──► [Save Image]
```

The upscaler accepts **any** image — not just Iris-3B outputs.
Plug any external image straight into it.

---

## Generation — working resolutions

Square resolutions below 1024×1024 produce artefacts due to the pixel decoder.
Use these:

| Resolution | Notes |
|---|---|
| 512×768 | Portrait — good quality |
| 768×512 | Landscape — good quality |
| 768×1344 | Tall portrait — excellent, ~32 GB VRAM |
| 832×1280 | Portrait — excellent |
| 896×1152 | Portrait |
| 1024×1024 | Square — fine at this size |
| 1152×896 | Landscape |
| 1280×832 | Landscape — excellent |
| 1344×768 | Wide landscape |

---

## Upscaler

One-step 4× restoration fine-tune. Works on any input image.

- **scale** — upscale factor, 4× is optimal (trained resolution)
- **color_fix** — wavelet colour fix keeps detail but takes colour from bicubic reference
- **fit_budget** — downscales input before upscaling to cap tile count. Off by default
- **short_side_cap / long_side_cap** — budget limits when fit_budget is on

Tile count scales with output area. At 4× scale:
- 512×768 input → 2048×3072 output → ~18 tiles
- 832×1280 input → 3328×5120 output → ~50 tiles

---

## VRAM usage

| Stage | Peak VRAM |
|-------|-----------|
| Generate (512×768) | ~14 GB |
| Generate (832×1280) | ~32 GB |
| Depth | ~12 GB |
| Upscale (per tile) | ~12 GB |

Between stages: ~0 GB (CPU offloaded, cache cleared).

---

## LoRA support

Point the **Iris-3B Model Loader** at any LoRA produced by
[AI-Toolkit](https://github.com/ostris/ai-toolkit) as a `.safetensors` file.
Merged in weight-space before inference, discarded afterwards.

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | HuggingFace auth token for private repos |
| `IRIS3B_SOURCE` | Override model root directory |
| `HF_HUB_OFFLINE=1` | Force offline mode once models are downloaded |
