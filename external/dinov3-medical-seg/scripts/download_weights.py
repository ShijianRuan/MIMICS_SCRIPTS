"""(Optional) Download DINOv3 weights from HuggingFace Hub."""

import argparse
from pathlib import Path

from transformers import AutoImageProcessor, DINOv3ViTBackbone


MODELS = {
    "vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vitb16", choices=list(MODELS.keys()))
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Local HuggingFace-format destination (defaults to models/dinov3-<model>)",
    )
    args = parser.parse_args()

    model_id = MODELS[args.model]
    print(f"Downloading {model_id}...")
    model = DINOv3ViTBackbone.from_pretrained(
        model_id,
        cache_dir=args.cache_dir,
    )
    processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=args.cache_dir)
    output_dir = Path(args.output_dir or Path(__file__).resolve().parent.parent / "models" / ("dinov3-" + args.model))
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print("Download complete: {}".format(output_dir))


if __name__ == "__main__":
    main()
