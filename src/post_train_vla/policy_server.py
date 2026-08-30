"""Small OpenPI-wire-compatible WebSocket policy server."""

from __future__ import annotations

import asyncio
import logging
import traceback

import websockets.asyncio.server
import websockets.exceptions
import websockets.frames

from post_train_vla import serialization

LOGGER = logging.getLogger(__name__)


class PolicyServer:
    def __init__(self, policy, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.policy = policy
        self.host = host
        self.port = port

    async def _handler(self, websocket) -> None:
        packer = serialization.Packer()
        await websocket.send(packer.pack(self.policy.metadata))
        while True:
            try:
                observation = serialization.unpackb(await websocket.recv())
                await websocket.send(packer.pack(self.policy.infer(observation)))
            except websockets.exceptions.ConnectionClosed:
                return
            except Exception:
                LOGGER.exception("Policy inference failed")
                await websocket.send(traceback.format_exc())
                await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason="Policy error")
                return

    async def run(self) -> None:
        async with websockets.asyncio.server.serve(
            self._handler, self.host, self.port, compression=None, max_size=None
        ) as server:
            await server.serve_forever()

    def serve_forever(self) -> None:
        asyncio.run(self.run())
