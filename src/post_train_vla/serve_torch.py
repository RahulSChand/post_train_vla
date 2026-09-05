"""Serve a standalone Torch pi0/pi0.5 checkpoint without importing OpenPI or JAX."""

import argparse
import logging
import pathlib

from post_train_vla.policy_server import PolicyServer
from post_train_vla.torch_policy import TorchPolicy


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--tokenizer", type=pathlib.Path, required=True)
    parser.add_argument("--model", choices=("pi0", "pi05"), default="pi0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--batch-wait-ms", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    policy = TorchPolicy(
        args.checkpoint,
        args.tokenizer,
        device=args.device,
        pi05=args.model == "pi05",
        compile_model=args.compile_model,
    )
    PolicyServer(
        policy,
        host=args.host,
        port=args.port,
        max_batch_size=args.max_batch_size,
        batch_wait_ms=args.batch_wait_ms,
    ).serve_forever()


if __name__ == "__main__":
    main()
