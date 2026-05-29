#!/usr/bin/env python3
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="HF model id or local base model path")
    parser.add_argument("--lora_path", required=True, help="LoRA adapter checkpoint path")
    parser.add_argument("--out_dir", required=True, help="Output dir for merged model")
    args = parser.parse_args()

    try:
        from transformers import AutoModel
        from peft import PeftModel
    except Exception as e:
        raise RuntimeError(
            "Missing dependencies. Install: pip install transformers peft"
        ) from e

    print(f"Loading base model: {args.base_model}")
    base = AutoModel.from_pretrained(args.base_model)
    print(f"Loading LoRA adapter: {args.lora_path}")
    lora_model = PeftModel.from_pretrained(base, args.lora_path)
    print("Merging LoRA weights into base model...")
    merged = lora_model.merge_and_unload()

    os.makedirs(args.out_dir, exist_ok=True)
    merged.save_pretrained(args.out_dir)
    print(f"Merged model saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
