"""Small policy interface and OpenPI-compatible WebSocket client."""

from __future__ import annotations

import logging
import time
from typing import Protocol, runtime_checkable

import numpy as np
import websockets.sync.client

from post_train_vla import serialization

LOGGER = logging.getLogger(__name__)


@runtime_checkable
class Policy(Protocol):
    def infer(self, observation: dict) -> dict:
        """Return an action chunk for one unbatched observation."""

    def reset(self) -> None:
        """Reset any per-episode policy state."""


class WebsocketPolicy:
    def __init__(
        self,
        url: str,
        *,
        api_key: str | None = None,
        connect_timeout: float = 120.0,
        retry_interval: float = 2.0,
    ) -> None:
        self.url = url if url.startswith(("ws://", "wss://")) else f"ws://{url}"
        self.api_key = api_key
        self.connect_timeout = connect_timeout
        self.retry_interval = retry_interval
        self._packer = serialization.Packer()
        self._connection = None
        self.metadata = {}
        self._connect()

    def _connect(self) -> None:
        deadline = time.monotonic() + self.connect_timeout
        headers = {"Authorization": f"Api-Key {self.api_key}"} if self.api_key else None
        while True:
            try:
                self._connection = websockets.sync.client.connect(
                    self.url,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=min(10.0, self.connect_timeout),
                )
                self.metadata = serialization.unpackb(self._connection.recv())
                LOGGER.info("Connected to %s; policy metadata=%s", self.url, self.metadata)
                return
            except (ConnectionRefusedError, TimeoutError, OSError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Policy server was not reachable at {self.url}") from None
                LOGGER.info("Waiting for policy server at %s", self.url)
                time.sleep(self.retry_interval)

    def infer(self, observation: dict) -> dict:
        if self._connection is None:
            raise RuntimeError("Policy connection is closed")
        self._connection.send(self._packer.pack(observation))
        response = self._connection.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Policy server error:\n{response}")  # noqa: TRY004
        result = serialization.unpackb(response)
        actions = np.asarray(result.get("actions"))
        if actions.ndim != 2 or actions.shape[1] < 7:
            raise ValueError(f"Expected policy actions with shape [horizon, >=7], got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("Policy returned NaN or infinite actions")
        return result

    def reset(self) -> None:
        # OpenPI's current protocol has no stateful reset message.
        return None

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
