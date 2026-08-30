import numpy as np
import pytest

from post_train_vla import serialization
from post_train_vla.evaluator import ActionChunker


def test_numpy_messagepack_round_trip():
    value = {"image": np.arange(24, dtype=np.uint8).reshape(2, 4, 3), "state": np.float32(2.5)}
    restored = serialization.unpackb(serialization.packb(value))
    np.testing.assert_array_equal(restored["image"], value["image"])
    assert restored["state"] == value["state"]


def test_action_chunker_replans_after_requested_actions():
    chunker = ActionChunker(replan_steps=2)
    chunker.add(np.arange(21, dtype=np.float32).reshape(3, 7))
    np.testing.assert_array_equal(chunker.pop(), np.arange(7, dtype=np.float32))
    assert not chunker.empty()
    np.testing.assert_array_equal(chunker.pop(), np.arange(7, 14, dtype=np.float32))
    assert chunker.empty()


def test_action_chunker_rejects_short_horizon():
    with pytest.raises(ValueError, match="shorter"):
        ActionChunker(replan_steps=2).add(np.zeros((1, 7), dtype=np.float32))
