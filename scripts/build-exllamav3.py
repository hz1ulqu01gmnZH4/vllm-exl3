"""Build an isolated ExLlamaV3 runtime against this Python's PyTorch.

Pass an ExLlamaV3 Python package directory containing ext.py and native sources,
then a new output directory. The source package is symlinked; its native sources
are not edited.
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--arch", default="12.0", help="PyTorch CUDA arch; RTX 5090 is 12.0"
    )
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    package = args.package.resolve(strict=True)
    output = args.output.resolve()
    if not (package / "ext.py").is_file() or not (package / "exllamav3_ext").is_dir():
        parser.error("package must contain ext.py and exllamav3_ext native sources")
    if args.jobs < 1 or package == output or package in output.parents:
        parser.error("use positive jobs and an output outside the source package")
    if importlib.util.find_spec("exllamav3_ext") is not None:
        parser.error("use an environment without a preinstalled exllamav3_ext binary")
    output.mkdir(parents=True, exist_ok=False)
    (output / "exllamav3").symlink_to(package, target_is_directory=True)
    sys.path.insert(0, str(output))
    os.environ["TORCH_CUDA_ARCH_LIST"] = args.arch
    os.environ["MAX_JOBS"] = str(args.jobs)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(output / "build")
    os.environ["EXL3_EXPANDABLE_SEGMENTS"] = "0"

    from exllamav3.ext import exllamav3_ext

    for name in ("exl3_moe", "pinned_cuda_view", "ngram_dequant"):
        if not callable(getattr(exllamav3_ext, name, None)):
            raise TypeError(f"Required extension function missing: {name}")
    binary = Path(exllamav3_ext.__file__).resolve()
    (output / binary.name).symlink_to(binary)
    print(f"Built {binary}")
    print(f"Add this directory to PYTHONPATH: {output}")


if __name__ == "__main__":
    main()
