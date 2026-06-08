# DGX Spark setup — moved

This standalone runbook was retired. DGX-side install + run knowledge now
lives with the packages that own it, refreshed against a live DGX install:

- **Isaac Lab / `strafer_lab` on the DGX** — the `env_isaaclab3` build
  recipe (Isaac Sim 6 + Isaac Lab develop + the CUDA-torch reinstall),
  DGX Spark prereqs, the `LD_PRELOAD` requirement, the unified-memory
  `nvidia-smi [N/A]` note, and the SkillGen / OpenXR / JAX-GPU / Livestream
  limitations: [`source/strafer_lab/README.md` → Install → Linux (DGX Spark)](../source/strafer_lab/README.md#install).
- **VLM + LLM planner services on the DGX** — the `.venv_vlm` bootstrap and
  the Blackwell `sm_121` NVRTC fix: [`Readme.md` → Install → DGX Spark](../Readme.md#dgx-spark-grace--blackwell-aarch64-ubuntu)
  and [`source/strafer_vlm/README.md` → Install](../source/strafer_vlm/README.md#install).
- **The env set, why each is separate, and recreate pointers**:
  [`docs/tasks/context/repo-topology.md` → Python environments (DGX)](tasks/context/repo-topology.md#python-environments-dgx).

This stub is kept for one PR cycle so existing links resolve; a follow-up
cleanup deletes it and repoints the remaining references.
