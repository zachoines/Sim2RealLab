# strafer_vlm

Workstation-first VLM tooling for Qwen2.5-VL-3B grounding before ROS integration.

## Install

```powershell
python -m pip install -e source/strafer_vlm
python -m pip install -e 'source/strafer_vlm[qwen,live]'
```

## Entry Points

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

## Dataset

Both chat-style and flat JSON/JSONL are supported. Bounding boxes are normalized to `[0, 1000]`.
