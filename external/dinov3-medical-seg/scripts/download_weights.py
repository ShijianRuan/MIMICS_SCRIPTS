"""(Optional) Download DINOv3 weights from HuggingFace Hub."""

import argparse
from transformers import DINOv3ViTBackbone


MODELS = {
    "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vitb16", choices=list(MODELS.keys()))
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    model_id = MODELS[args.model]
    print(f"Downloading {model_id}...")
    model = DINOv3ViTBackbone.from_pretrained(
        model_id,
        cache_dir=args.cache_dir,
    )
    print("Download complete.")


if __name__ == "__main__":
    main()
