from post_train_vla.finetune import ExactBalancedReplaySampler


def test_exact_balanced_replay_alternates_sources_deterministically():
    sampler = ExactBalancedReplaySampler(primary_length=5, replay_length=11, num_samples=10, seed=42)
    indices = list(sampler)

    assert len(indices) == 10
    assert all(index < 5 for index in indices[::2])
    assert all(index >= 5 for index in indices[1::2])
    assert indices == list(
        ExactBalancedReplaySampler(primary_length=5, replay_length=11, num_samples=10, seed=42)
    )
