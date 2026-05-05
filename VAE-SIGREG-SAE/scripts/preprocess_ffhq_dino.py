#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from time import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
from transformers import AutoImageProcessor, Dinov2WithRegistersModel


class CenterCropTransform:
    def __init__(self, image_size: int):
        self.image_size = image_size

    def __call__(self, pil_image: Image.Image) -> Image.Image:
        s = self.image_size
        while min(*pil_image.size) >= 2 * s:
            pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
        scale = s / min(*pil_image.size)
        pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
        arr = np.array(pil_image)
        cy = (arr.shape[0] - s) // 2
        cx = (arr.shape[1] - s) // 2
        return Image.fromarray(arr[cy : cy + s, cx : cx + s])


class FFHQParquetDataset(IterableDataset):
    def __init__(
        self,
        parquet_items: list[tuple[Path, int]],
        image_size: int,
        start_index: int = 0,
        max_samples: int | None = None,
    ):
        self.parquet_items = parquet_items
        self.start_index = start_index
        self.end_index = None if max_samples is None else start_index + max_samples
        self.max_samples = max_samples
        self.transform = transforms.Compose([CenterCropTransform(image_size), transforms.ToTensor()])

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        items = self.parquet_items
        if worker is not None:
            items = items[worker.id :: worker.num_workers]

        for parquet_path, offset in items:
            table = pq.read_table(parquet_path, columns=["image"])
            image_col = table.column("image")
            for row_idx in range(table.num_rows):
                global_idx = offset + row_idx
                if global_idx < self.start_index:
                    continue
                if self.end_index is not None and global_idx >= self.end_index:
                    return

                item = image_col[row_idx].as_py()
                image_bytes = item["bytes"]
                image_path = item.get("path") or f"{parquet_path.name}:{row_idx}"

                with Image.open(io.BytesIO(image_bytes)) as img:
                    img = img.convert("RGB")
                yield self.transform(img), image_path, global_idx


def collate_batch(batch):
    images, paths, indices = zip(*batch)
    return torch.stack(images, dim=0), list(paths), torch.tensor(indices, dtype=torch.long)


def build_parquet_items(parquet_files: list[Path]) -> tuple[list[tuple[Path, int]], int]:
    items = []
    offset = 0
    for path in parquet_files:
        items.append((path, offset))
        offset += pq.ParquetFile(path).metadata.num_rows
    return items, offset


def write_split(
    *,
    name: str,
    parquet_items: list[tuple[Path, int]],
    output_dir: Path,
    model: Dinov2WithRegistersModel,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    encoder_input_size: int,
    image_size: int,
    batch_size: int,
    num_workers: int,
    start_index: int,
    num_samples: int,
    precision: str,
):
    feature_path = output_dir / f"{name}_features.bin"
    shape_path = output_dir / f"{name}_features_shape.npy"
    paths_path = output_dir / f"{name}_paths.txt"

    use_amp = precision in {"bf16", "fp16"} and device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    dataset = FFHQParquetDataset(
        parquet_items=parquet_items,
        image_size=image_size,
        start_index=start_index,
        max_samples=num_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
    )

    written = 0
    t0 = time()
    with feature_path.open("wb") as features_f, paths_path.open("w") as paths_f:
        for step, (images, paths, indices) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            if images.shape[-2:] != (encoder_input_size, encoder_input_size):
                images = F.interpolate(
                    images,
                    size=(encoder_input_size, encoder_input_size),
                    mode="bicubic",
                    align_corners=False,
                )

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                cls = model((images - mean) / std).last_hidden_state[:, 0, :]

            features = cls.float().cpu().numpy().astype(np.float32, copy=False)
            features.tofile(features_f)
            for path, index in zip(paths, indices.tolist()):
                paths_f.write(f"{index}\t{path}\n")

            written += features.shape[0]
            if step == 1:
                np.save(shape_path, np.array([num_samples, features.shape[1]], dtype=np.int64))

            if step % 50 == 0 or written >= num_samples:
                rate = written / max(time() - t0, 1e-6)
                eta_min = (num_samples - written) / max(rate, 1e-6) / 60.0
                print(
                    f"[{name}] written={written}/{num_samples} "
                    f"({100.0 * written / num_samples:.1f}%) rate={rate:.1f}/s eta={eta_min:.1f}m",
                    flush=True,
                )

    if written != num_samples:
        np.save(shape_path, np.array([written, features.shape[1]], dtype=np.int64))
        raise RuntimeError(f"{name}: expected {num_samples} samples, wrote {written}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess FFHQ parquet shards into DINOv2 CLS feature bins.")
    parser.add_argument("--parquet-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="facebook/dinov2-with-registers-base")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--encoder-input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-count", type=int, default=63001)
    parser.add_argument("--eval-count", type=int, default=None, help="Defaults to all remaining images.")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquet_dir = Path(args.parquet_dir)
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {parquet_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = [
        output_dir / "train_features.bin",
        output_dir / "train_features_shape.npy",
        output_dir / "eval_features.bin",
        output_dir / "eval_features_shape.npy",
    ]
    if not args.overwrite and any(path.exists() for path in expected):
        raise FileExistsError(f"Output files already exist in {output_dir}. Pass --overwrite to replace them.")

    parquet_items, total_rows = build_parquet_items(parquet_files)
    eval_count = args.eval_count if args.eval_count is not None else total_rows - args.train_count
    if args.train_count <= 0 or eval_count <= 0 or args.train_count + eval_count > total_rows:
        raise ValueError(
            f"Invalid split: train={args.train_count}, eval={eval_count}, total={total_rows}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} parquet_files={len(parquet_files)} total_rows={total_rows}", flush=True)
    print(f"split train={args.train_count} eval={eval_count}", flush=True)

    proc = AutoImageProcessor.from_pretrained(args.model)
    mean = torch.tensor(proc.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(proc.image_std, device=device).view(1, 3, 1, 1)
    model = Dinov2WithRegistersModel.from_pretrained(args.model).to(device)
    model.eval().requires_grad_(False)

    meta = {
        "format": "ffhq_dinov2_cls_bin_v1",
        "parquet_dir": str(parquet_dir.resolve()),
        "model": args.model,
        "image_size": args.image_size,
        "encoder_input_size": args.encoder_input_size,
        "precision": args.precision,
        "total_rows": total_rows,
        "train_count": args.train_count,
        "eval_count": eval_count,
        "dtype": "float32",
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    common = dict(
        parquet_items=parquet_items,
        output_dir=output_dir,
        model=model,
        mean=mean,
        std=std,
        device=device,
        encoder_input_size=args.encoder_input_size,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        precision=args.precision,
    )
    write_split(name="train", start_index=0, num_samples=args.train_count, **common)
    write_split(name="eval", start_index=args.train_count, num_samples=eval_count, **common)
    print(f"done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
