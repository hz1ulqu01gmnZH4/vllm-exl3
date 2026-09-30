"""Prepare a serving overlay from an existing native EXL3 checkpoint.

Weights and tokenizer assets are symlinked. Only the new destination's config
and index are written; this does not quantize or modify the source checkpoint.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    destination = args.destination.resolve()
    if not (source / "config.json").is_file() or not any(source.glob("*.safetensors")):
        parser.error("source must contain config.json and safetensors weights")
    if source == destination or source in destination.parents:
        parser.error("destination must be outside the source checkpoint")
    destination.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.is_file() and path.name not in {
            "quantization_config.json",
            "model.safetensors.index.json",
            "config.json.native",
            "pack_scan.json",
        }:
            target = destination / path.name
            if path.name == "config.json":
                shutil.copyfile(path, target)
            else:
                target.symlink_to(path)
    scripts = Path(__file__).resolve().parent
    for script in (
        "qwen_pack_scan.py",
        "qwen_pack_config.py",
        "regenerate_safetensors_index.py",
    ):
        subprocess.run(
            [sys.executable, str(scripts / script), str(destination)], check=True
        )
    print(f"Prepared overlay: {destination}")


if __name__ == "__main__":
    main()
