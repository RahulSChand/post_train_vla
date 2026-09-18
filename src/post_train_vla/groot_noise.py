"""Deterministic action noise shared by GR00T evaluators."""
def seeded_noise_mode(seeds):
    """Generate each row independently, so request arrival order cannot alter noise."""
    import torch
    from torch.overrides import TorchFunctionMode

    class SeededNoise(TorchFunctionMode):
        draws = 0

        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func is torch.randn:
                shape = tuple(kwargs.get('size', args[0] if len(args) == 1 else args))
                if len(shape) != 3 or shape[0] != len(seeds) or self.draws:
                    raise ValueError(f'Unexpected inference noise draw: {shape}, draw={self.draws}')
                self.draws += 1
                options = {k: v for k, v in kwargs.items() if k != 'size'}
                if 'generator' in options:
                    raise ValueError('Checkpoint supplies its own generator')
                rows = []
                for seed in seeds:
                    generator = torch.Generator(device=options.get('device', 'cpu')).manual_seed(seed)
                    rows.append(torch.randn((1, *shape[1:]), generator=generator, **options))
                return torch.cat(rows, dim=0)
            if func in (torch.rand, torch.randn_like, torch.rand_like, torch.bernoulli):
                raise ValueError(f'Unexpected stochastic inference operation: {func}')
            return func(*args, **kwargs)

    return SeededNoise()
