#!/usr/bin/env python3
"""Full GR00T fine-tuning with finite epochs and local checkpoints at every epoch."""
import argparse
import importlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


class FiniteModernDataset(Dataset):
    """Every frame once per epoch, with action chunks padded within its episode."""
    def __init__(self, loader, modalities, processor, tag):
        self.episodes = [loader[i] for i in range(len(loader))]
        self.indices = [(ep, frame) for ep, data in enumerate(self.episodes) for frame in range(len(data))]
        self.modalities, self.processor, self.tag = modalities, processor, tag

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
        from gr00t.data.types import MessageType
        ep, frame = self.indices[index]
        step = extract_step_data(self.episodes[ep], frame, self.modalities, self.tag, allow_padding=True)
        return self.processor([{'type': MessageType.EPISODE_STEP.value, 'content': step}])


def modern(args):
    version = 'gr00t_n1d' + args.version[-1]
    module = importlib.import_module(f'gr00t.model.{version}.{version}')
    proc_module = importlib.import_module(f'gr00t.model.{version}.processing_{version}')
    model_class = getattr(module, 'Gr00tN1d' + args.version[-1])
    processor_class = getattr(proc_module, 'Gr00tN1d' + args.version[-1] + 'Processor')
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    tag = EmbodimentTag.LIBERO_PANDA
    modalities = MODALITY_CONFIGS[tag.value]
    model, loading = model_class.from_pretrained(
        args.checkpoint, tune_llm=True, tune_visual=True, tune_projector=True,
        tune_diffusion_model=True, tune_vlln=True, backbone_trainable_params_fp32=True,
        load_bf16=args.version == 'n1d6', use_flash_attention=args.version == 'n1d6', output_loading_info=True)
    for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
        if loading.get(key):
            raise RuntimeError(f'Checkpoint mismatch: {key}: {loading[key]}')
    processor = processor_class.from_pretrained(
        args.checkpoint, modality_configs={tag.value: modalities}, use_relative_action=False,
        max_action_horizon=model.config.action_horizon, state_dropout_prob=0.2)
    loader = LeRobotEpisodeLoader(args.dataset, modalities)
    processor.set_statistics({tag.value: loader.get_dataset_statistics()}, override=True)
    processor.train()
    processor.save_pretrained(args.output / 'processor')
    dataset = FiniteModernDataset(loader, modalities, processor, tag)
    return model, dataset, processor.collator, processor


def legacy(args):
    from gr00t.data.dataset import CachedLeRobotSingleDataset, ModalityConfig
    from gr00t.data.schema import EmbodimentTag
    from gr00t.data.transform.base import ComposedModalityTransform
    from gr00t.data.transform.concat import ConcatTransform
    from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
    from gr00t.data.transform.video import VideoToTensor, VideoCrop, VideoResize, VideoColorJitter, VideoToNumpy
    from gr00t.model.transforms import GR00TTransform
    transforms_module = importlib.import_module('gr00t.model.transforms')
    collator_class = getattr(transforms_module, 'DefaultDataCollatorGR00T' if args.version == 'n1' else 'DefaultDataCollator')
    module = importlib.import_module('gr00t.model.gr00t_n1')
    model_class = getattr(module, 'GR00T_N1' if args.version == 'n1' else 'GR00T_N1_5')
    # The upstream legacy wrapper discards loading diagnostics. Load through its
    # parent, then apply the same tuning flags, so missing weights cannot go unnoticed.
    model, loading = super(model_class, model_class).from_pretrained(
        args.checkpoint, local_model_path=args.checkpoint, output_loading_info=True)
    allowed_unused = {'action_head.decode_layer.weight', 'action_head.decode_layer.bias'} if args.version == 'n1' else set()
    for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
        problems = loading.get(key, [])
        if key == 'unexpected_keys':
            problems = sorted(set(problems) - allowed_unused)
        if problems:
            raise RuntimeError(f'Checkpoint mismatch: {key}: {problems}')
    write_json(args.output / 'checkpoint_loading.json', loading)
    model.backbone.set_trainable_parameters(tune_llm=True, tune_visual=True)
    model.action_head.set_trainable_parameters(tune_projector=True, tune_diffusion_model=True)
    model.config.backbone_cfg.update(tune_llm=True, tune_visual=True)
    model.config.action_head_cfg.update(tune_projector=True, tune_diffusion_model=True)
    videos = ['video.image', 'video.wrist_image']
    keys = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    states, actions = [[f'{kind}.{k}' for k in keys] for kind in ('state', 'action')]
    horizon = model.action_horizon
    modalities = {name: ModalityConfig(delta_indices=indices, modality_keys=keys)
                  for name, indices, keys in [('video', [0], videos), ('state', [0], states),
                    ('action', list(range(horizon)), actions),
                    ('language', [0], ['annotation.human.action.task_description'])]}
    transforms = ComposedModalityTransform(transforms=[
        VideoToTensor(apply_to=videos), VideoCrop(apply_to=videos, scale=.95),
        VideoResize(apply_to=videos, height=224, width=224, interpolation='linear'),
        VideoColorJitter(apply_to=videos, brightness=.3, contrast=.4, saturation=.5, hue=.08),
        VideoToNumpy(apply_to=videos), StateActionToTensor(apply_to=states + actions),
        StateActionTransform(apply_to=states + actions,
                             normalization_modes={k: 'min_max' for k in states + actions}),
        ConcatTransform(video_concat_order=videos, state_concat_order=states, action_concat_order=actions),
        GR00TTransform(state_horizon=1, action_horizon=horizon,
                       max_state_dim=model.action_head.config.max_state_dim,
                       max_action_dim=model.action_dim),
    ])
    dataset = CachedLeRobotSingleDataset(dataset_path=str(args.dataset), modality_configs=modalities,
                                  transforms=transforms, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                                  video_backend='decord')
    write_json(args.output / 'experiment_cfg/metadata.json',
               {dataset.tag: dataset.metadata.model_dump(mode='json')})
    write_json(args.output / 'experiment_cfg/modalities.json',
               {k: v.model_dump(mode='json') for k, v in modalities.items()})
    (args.output / 'experiment_cfg/transforms.json').write_text(transforms.model_dump_json(indent=2))
    model.compute_dtype = 'bfloat16'
    model.config.compute_dtype = 'bfloat16'
    collator = collator_class(processor=transforms.transforms[-1].vlm_processor) if args.version == 'n1' else collator_class()
    return model, dataset, collator, None


def save_epoch(args, model, processor, dataset, epoch, step):
    cp=args.output/f'epoch-{epoch:03d}'
    cp.mkdir(exist_ok=False)
    write_json(args.output/'status.json',dict(status='saving',epoch=epoch,step=step))
    model.save_pretrained(cp,state_dict={k:v.detach().cpu().to(torch.bfloat16) if v.is_floating_point()
                                       else v.detach().cpu() for k,v in model.state_dict().items()})
    if processor is not None: processor.save_pretrained(cp)
    if (args.output/'experiment_cfg').exists():
        shutil.copytree(args.output/'experiment_cfg',cp/'experiment_cfg')
        write_json(cp/'experiment_cfg/transforms_typed.json',[
            {'class':type(t).__module__+'.'+type(t).__name__,
             'kwargs':t.model_dump(mode='json',exclude={'vlm_processor','eagle_processor'})}
            for t in dataset.transforms.transforms])
    shutil.copytree(args.dataset/'meta',cp/'dataset_metadata')
    for name in ('trajectory_manifest.json','run_config.json','runtime.json','gradient_check.json','metrics.jsonl'):
        shutil.copy2(args.output/name,cp/name)
    write_json(cp/'epoch.json',dict(epoch=epoch,optimizer_step=step,epoch_cap=args.epochs,version=args.version))
    # Save exact source used for loading, training, normalization, and evaluation.
    runtime=Path(json.loads((args.output/'runtime.json').read_text())['root'])
    shutil.copytree(runtime/'gr00t',cp/'runtime/gr00t',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('LICENSE','NOTICE','ATTRIBUTIONS.md'):
        if (runtime/name).is_file(): shutil.copy2(runtime/name,cp/name)
    code=cp/'campaign_code'; code.mkdir()
    shutil.copy2(Path(__file__), code / Path(__file__).name)
    shutil.copytree(Path(__file__).resolve().parents[1]/'src/post_train_vla',code/'post_train_vla',ignore=shutil.ignore_patterns('__pycache__'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=['n1', 'n1d5', 'n1d6', 'n1d7'], required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--accumulation', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke', action='store_true', help='One real optimizer update, no saved weights.')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.accumulation < 1:
        parser.error('Epochs, batch size and accumulation must be positive')
    if (args.output / 'status.json').exists():
        raise FileExistsError(f'Use a fresh output directory: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.dataset / 'trajectory_manifest.json').read_text())
    if manifest['suite'] != 'libero_spatial' or manifest['trajectory_count'] < 1:
        raise ValueError('Expected a nonempty LIBERO Spatial trajectory manifest')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    model, dataset, collator, processor = (legacy if args.version in ('n1', 'n1d5') else modern)(args)
    assert len(dataset) == manifest['total_frames'], (len(dataset), manifest['total_frames'])
    # Explicitly include the entire VLM and all state/action projections and DiT parameters.
    model.requires_grad_(True)
    model.float().cuda().train()
    parameter_counts = {key: sum(p.numel() for p in sub.parameters() if p.requires_grad)
                        for key, sub in [('vlm', model.backbone), ('action_head', model.action_head)]}
    assert all(parameter_counts.values()) and all(p.requires_grad for p in model.parameters())
    write_json(args.output / 'run_config.json', dict(version=args.version, checkpoint=args.checkpoint,
               suite='libero_spatial', trajectories=manifest['trajectory_count'], max_epochs=args.epochs, batch_size=args.batch_size,
               accumulation=args.accumulation, learning_rate=args.learning_rate, trainable_parameters=parameter_counts,
               seed=args.seed, tune_llm=True, tune_visual=True, tune_projector=True, tune_diffusion_model=True,
               epoch_definition='Every selected frame once, shuffled without replacement; pad chunks within episodes'))
    import gr00t
    runtime = Path(gr00t.__file__).resolve().parent.parent
    write_json(args.output / 'runtime.json', dict(root=str(runtime),
               commit=subprocess.check_output(['git', '-C', str(runtime), 'rev-parse', 'HEAD'], text=True).strip(),
               torch=torch.__version__, python=os.sys.version))
    shutil.copy2(args.dataset / 'trajectory_manifest.json', args.output)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator,
                        num_workers=0, drop_last=False, generator=torch.Generator().manual_seed(args.seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    updates_per_epoch = math.ceil(len(loader) / args.accumulation)
    total_updates = updates_per_epoch * args.epochs
    warmup = max(1, int(total_updates * .05))
    step = 0
    started = time.time()
    write_json(args.output / 'status.json', dict(status='running', epoch=0, step=0))
    try:
        for epoch in range(1, args.epochs + 1):
            losses = []
            optimizer.zero_grad(set_to_none=True)
            for batch_index, batch in enumerate(loader):
                window_start = batch_index // args.accumulation * args.accumulation
                window_end = min(window_start + args.accumulation, len(loader))
                window_samples = min(len(dataset), window_end * args.batch_size) - window_start * args.batch_size
                batch_samples = min(args.batch_size, len(dataset) - batch_index * args.batch_size)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(batch.get('inputs', batch))['loss']
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss at epoch={epoch}, batch={batch_index}')
                (loss * batch_samples / window_samples).backward()
                losses.append(float(loss.detach()))
                if batch_index + 1 == window_end:
                    if step == 0:
                        gradients = {}
                        probes = []
                        groups = [('vlm_vision', [(n, p) for n, p in model.backbone.named_parameters()
                                                 if 'vision' in n or 'visual' in n]),
                                  ('vlm_language', [(n, p) for n, p in model.backbone.named_parameters()
                                                   if 'language_model' in n]),
                                  ('action_head', list(model.action_head.named_parameters()))]
                        for name, parameters in groups:
                            candidates = [(n, p) for n, p in parameters
                                          if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max() > 0]
                            if not candidates:
                                raise RuntimeError(f'{name} receives no finite, nonzero gradients')
                            n, p = candidates[0]
                            index = p.grad.abs().view(-1).argmax().item()
                            probes.append((name, n, p, index, p.detach().view(-1)[index].item()))
                            gradients[name] = {'parameter': n, 'max_abs_gradient': p.grad.abs().max().item()}
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    scale = min(1., (step + 1) / warmup) if step < warmup else .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_updates - warmup)))
                    for group in optimizer.param_groups:
                        group['lr'] = args.learning_rate * scale
                    optimizer.step()
                    if step == 0:
                        for name, n, p, index, before in probes:
                            after = p.detach().view(-1)[index].item()
                            if after == before:
                                raise RuntimeError(f'{name} probe did not update: {n}')
                            gradients[name].update(before=before, after=after)
                        write_json(args.output / 'gradient_check.json', gradients)
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    record = dict(epoch=epoch, step=step, loss=float(loss.detach()),
                                  epoch_frames=min((batch_index + 1) * args.batch_size, len(dataset)),
                                  learning_rate=optimizer.param_groups[0]['lr'], seconds=time.time()-started)
                    print(json.dumps(record), flush=True)
                    with (args.output / 'metrics.jsonl').open('a') as stream:
                        stream.write(json.dumps(record) + '\n')
                    if step == 1 or step % 10 == 0:
                        write_json(args.output / 'status.json', dict(status='running', **record))
                    if args.smoke:
                        write_json(args.output / 'status.json', dict(status='smoke_passed', step=step,
                                   peak_memory_gib=torch.cuda.max_memory_allocated()/2**30))
                        return
            write_json(args.output / 'status.json', dict(status='running', epoch=epoch, step=step,
                       mean_loss=sum(losses)/len(losses)))
            save_epoch(args, model, processor, dataset, epoch, step)
        write_json(args.output / 'status.json', dict(status='complete', epoch=args.epochs, step=step,
                   seconds=time.time()-started, checkpoints='epoch-*'))
    except Exception as error:
        write_json(args.output / 'status.json', dict(status='failed', step=step, error=repr(error)))
        raise


if __name__ == '__main__':
    main()
