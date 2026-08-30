import numpy as np

from post_train_vla.libero_observation import make_policy_observation, quaternion_to_axis_angle, resize_with_pad


def test_identity_quaternion_is_zero_without_mutating_input():
    quaternion = np.asarray([0.0, 0.0, 0.0, 1.0])
    original = quaternion.copy()
    np.testing.assert_array_equal(quaternion_to_axis_angle(quaternion), np.zeros(3))
    np.testing.assert_array_equal(quaternion, original)


def test_resize_with_pad_preserves_shape_and_dtype():
    image = np.full((100, 200, 3), 127, dtype=np.uint8)
    resized = resize_with_pad(image, 224, 224)
    assert resized.shape == (224, 224, 3)
    assert resized.dtype == np.uint8
    np.testing.assert_array_equal(resized[0], np.zeros((224, 3), dtype=np.uint8))


def test_make_policy_observation_contract():
    observation = {
        "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.ones((256, 256, 3), dtype=np.uint8),
        "robot0_eef_pos": np.asarray([1.0, 2.0, 3.0]),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.asarray([0.1, 0.2]),
    }
    result = make_policy_observation(observation, "pick up the object")
    assert result["observation/image"].shape == (224, 224, 3)
    assert result["observation/wrist_image"].shape == (224, 224, 3)
    assert result["observation/state"].shape == (8,)
    assert result["observation/state"].dtype == np.float32
    assert result["prompt"] == "pick up the object"
