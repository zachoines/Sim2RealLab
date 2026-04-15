"""OpenCLIP image-text contrastive fine-tuning on the strafer dataset.

Trains OpenCLIP ViT-B/32 on (image, description) pairs produced by
:mod:`strafer_lab.tools.dataset_export`. Both towers are trained jointly
with a symmetric InfoNCE loss so the model adapts its image-text alignment to
indoor robot perspectives at 25 cm camera height without drifting
away from the text tower's embedding space.

After training, both towers are exported to ONNX separately
(``clip_visual.onnx`` + ``clip_text.onnx``) because the Jetson's
``clip_encoder.py`` uses ``encode_image()`` for place recognition AND
``encode_text()`` for text queries.

Heavy imports (``torch``, ``open_clip``, ``mlflow``) are lazy so this
module stays importable for unit tests that only exercise the
dataset-plumbing helpers.

Usage:

    python scripts/finetune_clip.py \\
        --data data/clip_descriptions/clip_descriptions.csv \\
        --image-root data/perception \\
        --epochs 10 \\
        --output models/clip_finetuned/ \\
        --mlflow-experiment /Shared/strafer
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

logger = logging.getLogger("finetune_clip")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


@dataclass
class CLIPSample:
    image_path: Path
    description: str


def iter_csv_samples(
    csv_path: Path, *, image_root: Path,
) -> Iterator[CLIPSample]:
    """Yield :class:`CLIPSample` instances from ``clip_descriptions.csv``.

    Rows whose image file is missing on disk are skipped with a warning.
    """
    csv_path = Path(csv_path)
    image_root = Path(image_root)
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = row.get("image_path", "").strip()
            text = row.get("description", "").strip()
            if not rel or not text:
                continue
            full = image_root / rel
            if not full.exists():
                logger.warning("Skipping missing image: %s", full)
                continue
            yield CLIPSample(image_path=full, description=text)


def load_samples(csv_path: Path, *, image_root: Path) -> list[CLIPSample]:
    return list(iter_csv_samples(csv_path, image_root=image_root))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    data_csv: Path
    image_root: Path
    output_dir: Path
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-5
    weight_decay: float = 0.01
    num_workers: int = 4
    seed: int = 0
    mlflow_experiment: str | None = None
    mlflow_run_name: str | None = None
    export_onnx: bool = True


def parse_args(argv: Iterable[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, dest="data_csv")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, dest="output_dir")
    parser.add_argument("--model", default="ViT-B-32", dest="model_name")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mlflow-experiment", default=None)
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument("--no-export-onnx", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return TrainConfig(
        data_csv=args.data_csv,
        image_root=args.image_root,
        output_dir=args.output_dir,
        model_name=args.model_name,
        pretrained=args.pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_run_name=args.mlflow_run_name,
        export_onnx=not args.no_export_onnx,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(config: TrainConfig) -> Path:
    """Run Phase-1 image-text contrastive fine-tuning.

    Returns the output directory containing the final checkpoint.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    import open_clip
    from PIL import Image

    logger.info("Loading %s / %s", config.model_name, config.pretrained)
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.model_name, pretrained=config.pretrained,
    )
    tokenizer = open_clip.get_tokenizer(config.model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    samples = load_samples(config.data_csv, image_root=config.image_root)
    if not samples:
        raise RuntimeError(f"No samples loaded from {config.data_csv}")
    logger.info("Loaded %d samples", len(samples))

    class _StraferCLIPDataset(Dataset):
        def __init__(self, items: list[CLIPSample]) -> None:
            self._items = items

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, idx: int):
            sample = self._items[idx]
            image = Image.open(sample.image_path).convert("RGB")
            return preprocess(image), sample.description

    def _collate(batch):
        images = torch.stack([b[0] for b in batch])
        tokens = tokenizer([b[1] for b in batch])
        return images, tokens

    loader = DataLoader(
        _StraferCLIPDataset(samples),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=_collate,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    mlflow_run = _start_mlflow(config)

    step = 0
    best_loss = float("inf")
    for epoch in range(config.epochs):
        epoch_losses: list[float] = []
        for images, tokens in loader:
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            image_emb = model.encode_image(images)
            text_emb = model.encode_text(tokens)

            image_emb = F.normalize(image_emb, dim=-1)
            text_emb = F.normalize(text_emb, dim=-1)
            logit_scale = model.logit_scale.exp()
            logits_i2t = logit_scale * image_emb @ text_emb.T
            logits_t2i = logits_i2t.T
            labels = torch.arange(images.size(0), device=device)
            loss = (
                F.cross_entropy(logits_i2t, labels)
                + F.cross_entropy(logits_t2i, labels)
            ) / 2.0

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            step += 1
            if mlflow_run is not None:
                _mlflow_log_step(loss.item(), step)

        mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        logger.info("epoch %d mean_loss=%.4f", epoch, mean_loss)
        if mean_loss < best_loss:
            best_loss = mean_loss

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "clip_finetuned.pt"
    torch.save(model.state_dict(), checkpoint_path)
    _write_config_dump(config.output_dir, config)
    logger.info("Saved checkpoint to %s", checkpoint_path)

    if config.export_onnx:
        try:
            export_towers_to_onnx(
                model=model,
                tokenizer=tokenizer,
                preprocess=preprocess,
                output_dir=config.output_dir,
            )
        except Exception:
            logger.exception("ONNX export failed; checkpoint still saved.")

    if mlflow_run is not None:
        _mlflow_end(mlflow_run, checkpoint_path=checkpoint_path, config=config)

    return config.output_dir


def export_towers_to_onnx(
    *,
    model: Any,
    tokenizer: Any,
    preprocess: Any,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Export the visual and text towers to ONNX files.

    The Jetson ``clip_encoder.py`` expects both towers as separate
    ``clip_visual.onnx`` + ``clip_text.onnx`` files, so this function
    traces each tower in isolation.
    """
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device

    dummy_image = torch.zeros((1, 3, 224, 224), device=device)
    visual_path = output_dir / "clip_visual.onnx"
    torch.onnx.export(
        model.visual,
        dummy_image,
        str(visual_path),
        input_names=["images"],
        output_names=["image_embeddings"],
        dynamic_axes={"images": {0: "batch"}, "image_embeddings": {0: "batch"}},
        opset_version=17,
    )

    dummy_tokens = tokenizer(["a photo of a room"]).to(device)
    text_path = output_dir / "clip_text.onnx"

    class _TextTowerWrapper(torch.nn.Module):
        """Expose ``encode_text`` as a stand-alone ``forward``."""

        def __init__(self, parent):
            super().__init__()
            self._parent = parent

        def forward(self, tokens):
            return self._parent.encode_text(tokens)

    torch.onnx.export(
        _TextTowerWrapper(model),
        dummy_tokens,
        str(text_path),
        input_names=["tokens"],
        output_names=["text_embeddings"],
        dynamic_axes={"tokens": {0: "batch"}, "text_embeddings": {0: "batch"}},
        opset_version=17,
    )
    logger.info("Exported ONNX towers to %s and %s", visual_path, text_path)
    return visual_path, text_path


# ---------------------------------------------------------------------------
# MLflow bookkeeping
# ---------------------------------------------------------------------------


def _start_mlflow(config: TrainConfig):
    if not config.mlflow_experiment:
        return None
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not installed; skipping experiment tracking")
        return None
    mlflow.set_experiment(config.mlflow_experiment)
    run = mlflow.start_run(run_name=config.mlflow_run_name or "clip_finetune")
    mlflow.log_params({k: str(v) for k, v in asdict(config).items()})
    return run


def _mlflow_log_step(loss: float, step: int) -> None:
    try:
        import mlflow

        mlflow.log_metric("loss", loss, step=step)
    except Exception:  # pragma: no cover
        pass


def _mlflow_end(run, *, checkpoint_path: Path, config: TrainConfig) -> None:
    try:
        import mlflow

        mlflow.log_artifact(str(checkpoint_path))
        mlflow.end_run()
    except Exception:  # pragma: no cover
        pass


def _write_config_dump(output_dir: Path, config: TrainConfig) -> None:
    payload = {k: str(v) for k, v in asdict(config).items()}
    (output_dir / "training_config.json").write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = parse_args(argv)
    train(config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
