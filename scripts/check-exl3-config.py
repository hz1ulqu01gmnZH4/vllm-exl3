"""Resolve the launcher's vLLM configuration without loading model weights."""

import json
import sys

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.launchers.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser

parser = make_arg_parser(FlexibleArgumentParser())
args = parser.parse_args(sys.argv[1:])
if args.model_tag is not None:
    args.model = args.model_tag
validate_parsed_serve_args(args)
config = AsyncEngineArgs.from_cli_args(args).create_engine_config()
print(
    json.dumps(
        {
            "validation": "configuration only; no model weights loaded",
            "model": config.model_config.model,
            "quantization": config.model_config.quantization,
            "max_model_len": config.model_config.max_model_len,
            "max_num_seqs": config.scheduler_config.max_num_seqs,
            "kv_cache_memory_bytes": config.cache_config.kv_cache_memory_bytes,
            "enforce_eager": config.model_config.enforce_eager,
        },
        indent=2,
    )
)
