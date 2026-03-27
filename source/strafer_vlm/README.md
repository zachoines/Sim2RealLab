# strafer_vlm

Workstation-first VLM tooling for Qwen2.5-VL-3B grounding before ROS integration.

## Install

```powershell
# Core package
python -m pip install -e source/strafer_vlm

# With Qwen inference + live grounding (requires GPU)
python -m pip install -e 'source/strafer_vlm[qwen,live]'

# With HTTP grounding service
python -m pip install -e 'source/strafer_vlm[service]'

# With dev/test tooling
python -m pip install -e 'source/strafer_vlm[dev]'
```

## Grounding Service

The HTTP grounding service wraps Qwen2.5-VL inference behind a FastAPI endpoint.
It runs on the workstation GPU and is called by `HttpGroundingClient` on the Jetson.

### Launch

```powershell
uvicorn strafer_vlm.service.app:create_app --factory --host 0.0.0.0 --port 8100
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROUNDING_MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct` | HuggingFace model name or local path |
| `GROUNDING_DEVICE_MAP` | `auto` | PyTorch device map |
| `GROUNDING_TORCH_DTYPE` | `auto` | Torch dtype (`auto`, `float16`, `bfloat16`) |
| `GROUNDING_LOAD_4BIT` | `0` | Set `1` to enable 4-bit quantisation |
| `GROUNDING_MAX_TOKENS` | `128` | Max new tokens per inference |
| `GROUNDING_MAX_IMAGE_MP` | `20` | Max decoded image megapixels (rejects larger) |
| `GROUNDING_HOST` | `0.0.0.0` | Bind host |
| `GROUNDING_PORT` | `8100` | Bind port |

### API

**`GET /health`** — readiness check

```json
{"status": "ok", "model_loaded": true, "model_name": "Qwen/Qwen2.5-VL-3B-Instruct"}
```

**`POST /ground`** — run grounding inference

Request:
```json
{
  "request_id": "req-001",
  "prompt": "Locate: red chair",
  "image_jpeg_b64": "<base64-encoded JPEG>",
  "max_image_side": 1024
}
```

Response:
```json
{
  "request_id": "req-001",
  "found": true,
  "bbox_2d": [120, 80, 500, 400],
  "label": "chair",
  "confidence": 0.92,
  "raw_output": "{\"found\": true, ...}",
  "latency_s": 2.15
}
```

Bounding boxes use the **normalized [0, 1000] coordinate convention**, not pixel coordinates.

### curl Example

```bash
IMAGE_B64=$(base64 -w0 photo.jpg)
curl -s http://localhost:8100/ground \
  -H "Content-Type: application/json" \
  -d "{\"request_id\": \"test\", \"prompt\": \"Locate: door\", \"image_jpeg_b64\": \"$IMAGE_B64\"}"
```

### PowerShell Example

```powershell
$bytes = [System.IO.File]::ReadAllBytes("photo.jpg")
$b64   = [Convert]::ToBase64String($bytes)
$body  = @{ request_id="test"; prompt="Locate: door"; image_jpeg_b64=$b64 } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8100/ground -Method Post -Body $body -ContentType "application/json"
```

Interactive API docs are available at `http://localhost:8100/docs` (Swagger UI).

## CLI Entry Points

```powershell
python -m strafer_vlm.test_qwen25vl_grounding --help
python -m strafer_vlm.train_qwen25vl_lora --help
python -m strafer_vlm.eval_qwen25vl_grounding --help
python -m strafer_vlm.live_qwen25vl_grounding --help
```

## Live Evaluation

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.\.venv_vlm\Scripts\python -m strafer_vlm.live_qwen25vl_grounding --source 0 --prompt "Locate: TV on dresser."
```

Type a new query in the console and press Enter. The next Qwen inference uses the updated text without restarting the script. Use `status`, `clear`, or `quit` in the console while the live window is open.

## Tests

```powershell
python -m pytest source/strafer_vlm/tests/ -v
```

All unit tests run without a GPU. Tests marked `@pytest.mark.gpu` require a CUDA device.

## Dataset Building & Fine-Tuning

### Dataset Format

`dataset_io.py` supports two JSONL formats. Image paths are resolved relative to the JSONL file's directory.

**Flat format** (recommended for simplicity):

```jsonl
{"image": "images/frame_0001.jpg", "prompt": "the chair next to the couch", "target": {"found": true, "bbox_2d": [120, 340, 580, 890], "label": "chair"}}
{"image": "images/frame_0002.jpg", "prompt": "the red fire extinguisher", "target": {"found": true, "bbox_2d": [50, 100, 200, 400], "label": "fire extinguisher"}}
{"image": "images/frame_0003.jpg", "prompt": "a dog", "target": {"found": false}}
```

**Chat format** (matches HuggingFace chat template convention):

```jsonl
{"messages": [{"role": "user", "content": [{"type": "image", "image": "images/frame_0001.jpg"}, {"type": "text", "text": "Locate: the chair"}]}, {"role": "assistant", "content": "{\"found\":true,\"bbox_2d\":[120,340,580,890],\"label\":\"chair\"}"}]}
```

### Bounding Box Coordinate System

All `bbox_2d` values use **normalized [0, 1000]** coordinates, not raw pixels. This is the coordinate space Qwen2.5-VL was trained with. Convert from pixel coordinates:

```
x_norm = round(x_px / image_width  * 1000)
y_norm = round(y_px / image_height * 1000)
```

### Annotation Tools

Use any standard object detection labeling tool to draw bounding boxes, then convert the export to the flat JSONL format above.

| Tool | Notes |
|------|-------|
| [Label Studio](https://labelstud.io/) | Free, self-hosted, has JSONL export |
| [CVAT](https://www.cvat.ai/) | Free, strong bbox annotation tooling |
| [Roboflow](https://roboflow.com/) | Free tier, exports COCO JSON |
| LabelImg | Lightweight desktop app, exports VOC XML |

### Converting COCO Annotations to Grounding JSONL

Most tools can export COCO JSON. Convert it with a script like:

```python
import json
from pathlib import Path

def coco_to_grounding_jsonl(coco_json: str, output: str):
    coco = json.loads(Path(coco_json).read_text())
    img_map = {img["id"]: img for img in coco["images"]}
    cat_map = {cat["id"]: cat["name"] for cat in coco["categories"]}

    with open(output, "w") as f:
        for ann in coco["annotations"]:
            img = img_map[ann["image_id"]]
            x, y, w, h = ann["bbox"]  # COCO uses [x, y, w, h] in pixels
            x1 = round(x / img["width"] * 1000)
            y1 = round(y / img["height"] * 1000)
            x2 = round((x + w) / img["width"] * 1000)
            y2 = round((y + h) / img["height"] * 1000)
            record = {
                "image": img["file_name"],
                "prompt": cat_map[ann["category_id"]],
                "target": {
                    "found": True,
                    "bbox_2d": [x1, y1, x2, y2],
                    "label": cat_map[ann["category_id"]],
                },
            }
            f.write(json.dumps(record) + "\n")
```

### Building a Good Dataset

1. **Collect images** — capture frames from the Strafer's camera in representative environments (different rooms, lighting, angles).
2. **Annotate** — draw bounding boxes using a tool above. A few hundred examples is a reasonable starting point for LoRA fine-tuning.
3. **Include negatives** — add `{"found": false}` samples where the prompted object is not in the image, so the model learns to say "not found."
4. **Split train/val** — use ~90/10. Place them alongside the images:

```
data/vlm/
  images/
    frame_0001.jpg
    frame_0002.jpg
    ...
  train.jsonl
  val.jsonl
```

### Fine-Tuning with LoRA

```powershell
python -m strafer_vlm.training.train_qwen25vl_lora `
  --train-dataset data/vlm/train.jsonl `
  --eval-dataset data/vlm/val.jsonl `
  --output-dir outputs/qwen25vl_lora_run1
```

Key flags (see `--help` for all):

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-VL-3B-Instruct` | Base model checkpoint |
| `--train-dataset` | *(required)* | Path to training JSONL |
| `--eval-dataset` | `None` | Path to validation JSONL (auto-split if omitted) |
| `--eval-split` | `0.1` | Fraction to split for eval when no eval dataset given |
| `--output-dir` | *(required)* | Where to save adapter weights |
| `--load-4bit` | `false` | QLoRA 4-bit quantised training |

### Evaluating a Fine-Tuned Model

```powershell
python -m strafer_vlm.training.eval_qwen25vl_grounding `
  --dataset data/vlm/val.jsonl `
  --model Qwen/Qwen2.5-VL-3B-Instruct `
  --adapter outputs/qwen25vl_lora_run1 `
  --output-dir outputs/qwen25vl_eval_run1
```

This produces per-example predictions with IoU scores and an `evaluation_summary.json`.

### Notes

- The dataset loader validates that all referenced image files exist at load time.
- Prompts are auto-normalized to `"Locate: ..."` format if not already prefixed.
- For larger datasets (10k+), consider using the HuggingFace `datasets` library with Arrow-backed streaming instead of `load_grounding_dataset`, which reads everything into memory.
