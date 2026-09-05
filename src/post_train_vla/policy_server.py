"""Small OpenPI-wire-compatible WebSocket policy server."""

from __future__ import annotations

import asyncio
import logging
import time
import traceback

import websockets.asyncio.server
import websockets.exceptions
import websockets.frames

from post_train_vla import serialization

LOGGER = logging.getLogger(__name__)


class PolicyServer:
    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 8000,
        *,
        max_batch_size: int = 1,
        batch_wait_ms: float = 5.0,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self.policy = policy
        self.host = host
        self.port = port
        self.max_batch_size = max_batch_size
        self.batch_wait_ms = batch_wait_ms
        self._batch_queue = None

    async def _infer(self, observation):
        if self.max_batch_size == 1:
            return self.policy.infer(observation)
        future = asyncio.get_running_loop().create_future()
        await self._batch_queue.put((observation, future))
        return await future

    async def _batch_worker(self) -> None:
        while True:
            requests = [await self._batch_queue.get()]
            deadline = time.monotonic() + self.batch_wait_ms / 1000.0
            while len(requests) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    requests.append(await asyncio.wait_for(self._batch_queue.get(), timeout=remaining))
                except asyncio.TimeoutError:
                    break
            try:
                results = self.policy.infer_batch([observation for observation, _ in requests])
                if len(results) != len(requests):
                    raise RuntimeError(f"Batched policy returned {len(results)} results for {len(requests)} requests")
                for result, (_, future) in zip(results, requests, strict=True):
                    if not future.cancelled():
                        future.set_result(result)
            except Exception as exc:
                LOGGER.exception("Batched policy inference failed")
                for _, future in requests:
                    if not future.cancelled():
                        future.set_exception(exc)

    async def _handler(self, websocket) -> None:
        packer = serialization.Packer()
        await websocket.send(packer.pack(self.policy.metadata))
        while True:
            try:
                observation = serialization.unpackb(await websocket.recv())
                if isinstance(observation, dict) and observation.get("__command__") == "load_checkpoint":
                    self.policy.load_checkpoint(observation["checkpoint"])
                    await websocket.send(packer.pack({"ok": True, "metadata": self.policy.metadata}))
                    continue
                await websocket.send(packer.pack(await self._infer(observation)))
            except websockets.exceptions.ConnectionClosed:
                return
            except Exception:
                LOGGER.exception("Policy inference failed")
                await websocket.send(traceback.format_exc())
                await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason="Policy error")
                return

    async def run(self) -> None:
        batch_task = None
        if self.max_batch_size > 1:
            if not hasattr(self.policy, "infer_batch"):
                raise TypeError("Policy must provide infer_batch when max_batch_size is greater than one")
            self._batch_queue = asyncio.Queue()
            batch_task = asyncio.create_task(self._batch_worker())
        try:
            async with websockets.asyncio.server.serve(
                self._handler, self.host, self.port, compression=None, max_size=None
            ) as server:
                await server.serve_forever()
        finally:
            if batch_task is not None:
                batch_task.cancel()

    def serve_forever(self) -> None:
        asyncio.run(self.run())
