# strafer_vlm

Workstation-first VLM tooling for Qwen2.5-VL-3B grounding before ROS integration.

## Install

```powershell
python -m pip install -e source/strafer_vlm
```

## Entry Points

```powershell
python -m strafer_vlm.test_qwen25vl_grounding --help
python -m strafer_vlm.train_qwen25vl_lora --help
python -m strafer_vlm.eval_qwen25vl_grounding --help
```

## Dataset

Both chat-style and flat JSON/JSONL are supported. Bounding boxes are normalized to `[0, 1000]`.
