#!/usr/bin/env python3
"""Benchmark inference latency on an exported policy artifact.

Loads a ``.pt`` (TorchScript) or ``.onnx`` (ONNX Runtime) file produced by
``Scripts/export_policy.py`` and prints median, p95, and p99 inference
latency on a synthetic observation vector. The Jetson side runs this after
``rsync``-ing the artifact to verify deployment-time latency budgets; the
DGX side uses it as a regression check on the export toolchain.

The execution-provider preference is configurable so the same script can
report TensorRT-EP, CUDA-EP, and CPU-EP latencies on the Jetson::

    python Scripts/benchmark_policy.py \\
        --model models/strafer_depth_v0.onnx \\
        --variant DEPTH \\
        --providers TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider \\
        --iters 1000

For a TorchScript artifact the providers flag is ignored.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def _format_table(rows: list[tuple[str, str]]) -> str:
    """Render aligned label / value pairs."""
    width = max(len(label) for label, _ in rows)
    return "\n".join(f"{label:<{width}}  {value}" for label, value in rows)


def main() -> None:
    """CLI entry: load the artifact, time inference, print latency stats."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark inference latency for an exported strafer policy "
            "artifact (.pt or .onnx). Reports median / p95 / p99 over a "
            "synthetic observation."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the .pt or .onnx artifact produced by export_policy.py.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help=(
            "PolicyVariant name (e.g. NOCAM, DEPTH). Defaults to the value "
            "recorded in the artifact's sidecar JSON when present."
        ),
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=1000,
        help="Inference iterations to time (default: 1000).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup iterations before timing starts (default: 20).",
    )
    parser.add_argument(
        "--providers",
        type=str,
        default=None,
        help=(
            "Comma-separated ONNX Runtime execution-provider preference, e.g. "
            "'TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider'. "
            "Ignored for .pt artifacts. Defaults to ORT's auto-selection."
        ),
    )
    args = parser.parse_args()

    from strafer_shared.policy_interface import PolicyVariant

    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"Model artifact not found: {model_path}")

    # Resolve variant: explicit flag wins; otherwise read the sidecar.
    if args.variant is not None:
        variant = PolicyVariant[args.variant]
    else:
        sidecar_path = model_path.with_suffix(".json")
        if not sidecar_path.is_file():
            raise SystemExit(
                f"--variant not provided and no sidecar at {sidecar_path}. "
                f"Pass --variant NOCAM|DEPTH or rerun export_policy.py."
            )
        import json

        sidecar = json.loads(sidecar_path.read_text())
        variant = PolicyVariant[sidecar["policy_variant"]]

    # Load the policy. TorchScript is loaded directly via load_policy(); the
    # ONNX path is also loaded via load_policy() unless --providers asks for
    # a non-default provider preference, in which case we build the session
    # directly so the TRT EP can be exercised on the Jetson.
    if model_path.suffix == ".onnx" and args.providers is not None:
        import onnxruntime as ort

        provider_list = [p.strip() for p in args.providers.split(",") if p.strip()]
        sess = ort.InferenceSession(str(model_path), providers=provider_list)
        active = sess.get_providers()
        input_name = sess.get_inputs()[0].name
        obs_dim = variant.obs_dim

        def policy(obs: np.ndarray) -> np.ndarray:
            obs_f32 = obs.astype(np.float32).reshape(1, obs_dim)
            return sess.run(None, {input_name: obs_f32})[0].squeeze(0)

        provider_label = ", ".join(active) if active else "<unknown>"
    else:
        from strafer_shared.policy_interface import load_policy

        policy = load_policy(model_path, variant)
        provider_label = "torch.jit" if model_path.suffix == ".pt" else "onnxruntime (default providers)"

    obs = np.zeros(variant.obs_dim, dtype=np.float32)

    # Warmup -- exclude from timing to skip first-call setup cost (lazy
    # CUDA-context init, TRT engine build, etc.).
    for _ in range(args.warmup):
        policy(obs)

    times_ms: list[float] = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        policy(obs)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times_ms)
    rows = [
        ("model", str(model_path)),
        ("variant", variant.name),
        ("obs_dim", str(variant.obs_dim)),
        ("provider", provider_label),
        ("iterations", str(args.iters)),
        ("warmup", str(args.warmup)),
        ("median (ms)", f"{float(np.median(arr)):.3f}"),
        ("p95 (ms)", f"{float(np.percentile(arr, 95)):.3f}"),
        ("p99 (ms)", f"{float(np.percentile(arr, 99)):.3f}"),
        ("mean (ms)", f"{float(arr.mean()):.3f}"),
        ("min (ms)", f"{float(arr.min()):.3f}"),
        ("max (ms)", f"{float(arr.max()):.3f}"),
    ]
    print(_format_table(rows))


if __name__ == "__main__":
    main()
