import numpy as np
import pytest

from post_train_vla import serialization
from post_train_vla.evaluator import ActionChunker
from post_train_vla.eval_libero import _episode_chunks


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


def test_episode_chunks_cover_each_episode_once():
    assert _episode_chunks(20, 6) == [(0, 4), (4, 4), (8, 3), (11, 3), (14, 3), (17, 3)]
    assert _episode_chunks(3, 20) == [(0, 1), (1, 1), (2, 1)]
