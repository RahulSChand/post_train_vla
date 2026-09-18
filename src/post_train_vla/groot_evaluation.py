"""Evaluate saved checkpoints with fixed LIBERO episodes and version-native runtimes."""
import argparse
import asyncio
import concurrent.futures
import contextlib
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

KEYS = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']


def legacy_language_batch(version, observations):
    """N1.5's tree mapper must see a string-array leaf, not nested lists."""
    import numpy as np
    prompts=[str(o['prompt']) for o in observations]
    return [[p] for p in prompts] if version=='n1' else np.asarray(prompts)


def legacy_transform(model):
    from gr00t.data.transform.base import ComposedModalityTransform
    from gr00t.data.transform.concat import ConcatTransform
    from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
    from gr00t.data.transform.video import VideoToTensor, VideoCrop, VideoResize, VideoColorJitter, VideoToNumpy
    from gr00t.model.transforms import GR00TTransform
    videos = ['video.image', 'video.wrist_image']
    states, actions = [[f'{kind}.{k}' for k in KEYS] for kind in ('state', 'action')]
    return ComposedModalityTransform(transforms=[
        VideoToTensor(apply_to=videos), VideoCrop(apply_to=videos, scale=.95),
        VideoResize(apply_to=videos, height=224, width=224, interpolation='linear'),
        VideoColorJitter(apply_to=videos, brightness=.3, contrast=.4, saturation=.5, hue=.08),
        VideoToNumpy(apply_to=videos), StateActionToTensor(apply_to=states+actions),
        StateActionTransform(apply_to=states+actions, normalization_modes={k:'min_max' for k in states+actions}),
        ConcatTransform(video_concat_order=videos, state_concat_order=states, action_concat_order=actions),
        GR00TTransform(state_horizon=1, action_horizon=model.action_horizon,
                       max_state_dim=model.action_head.config.max_state_dim, max_action_dim=model.action_dim)])


class SavedPolicy:
    def __init__(self, version, checkpoint):
        import torch
        self.version = version
        self.metadata = {'version': version, 'checkpoint': str(checkpoint)}
        torch.set_num_threads(4)
        if version in ('n1','n1d5'):
            from gr00t.data.schema import DatasetMetadata
            module = importlib.import_module('gr00t.model.gr00t_n1')
            cls = getattr(module, 'GR00T_N1' if version=='n1' else 'GR00T_N1_5')
            self.model, info = super(cls, cls).from_pretrained(str(checkpoint), local_model_path=str(checkpoint), output_loading_info=True)
            spec = checkpoint/'experiment_cfg/transforms_typed.json'
            if spec.exists():
                from gr00t.data.transform.base import ComposedModalityTransform
                transforms=[]
                for entry in json.loads(spec.read_text()):
                    mod, name = entry['class'].rsplit('.',1)
                    transforms.append(getattr(importlib.import_module(mod),name)(**entry['kwargs']))
                self.processor = ComposedModalityTransform(transforms=transforms)
            else:
                self.processor = legacy_transform(self.model)
            metadata = json.loads((checkpoint/'experiment_cfg/metadata.json').read_text())
            self.processor.set_metadata(DatasetMetadata.model_validate(metadata['new_embodiment']))
        else:
            module_name='gr00t_n1d'+version[-1]
            module=importlib.import_module(f'gr00t.model.{module_name}.{module_name}')
            proc_module=importlib.import_module(f'gr00t.model.{module_name}.processing_{module_name}')
            cls=getattr(module,'Gr00tN1d'+version[-1])
            proc_cls=getattr(proc_module,'Gr00tN1d'+version[-1]+'Processor')
            self.model, info = cls.from_pretrained(str(checkpoint), output_loading_info=True)
            self.processor = proc_cls.from_pretrained(str(checkpoint))
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError(f'Checkpoint load mismatch: {info}')
        self.processor.eval()
        self.model.eval().requires_grad_(False).cuda()
        self.metadata['strict_weight_load'] = True

    def infer(self, obs):
        return self.infer_batch([obs])[0]

    def infer_batch(self, observations):
        import numpy as np
        import torch
        from post_train_vla.groot_noise import seeded_noise_mode
        state = np.stack([o['observation/state'] for o in observations])[:,None]
        states = {k:state[:,:,i:(8 if i==6 else i+1)] for i,k in enumerate(KEYS)}
        videos = {k:np.stack([o['observation/'+k] for o in observations])[:,None] for k in ('image','wrist_image')}
        seeds = [o.get('_evaluation_noise_seed',7+i) for i,o in enumerate(observations)]
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16), seeded_noise_mode(seeds):
            if self.version in ('n1','n1d5'):
                inputs = {**{'state.'+k:v for k,v in states.items()}, **{'video.'+k:v for k,v in videos.items()},
                          'annotation.human.action.task_description': legacy_language_batch(self.version,observations)}
                output = self.model.get_action(self.processor(inputs))['action_pred'].float().cpu()
                decoded = self.processor.unapply({'action':output})
                actions = np.concatenate([np.asarray(decoded['action.'+k]) for k in KEYS],axis=-1)
            else:
                from gr00t.data.types import VLAStepData, MessageType
                from gr00t.data.embodiment_tags import EmbodimentTag
                tag = EmbodimentTag.LIBERO_PANDA
                samples=[]
                for i,o in enumerate(observations):
                    step = VLAStepData(images={k:v[i] for k,v in videos.items()}, states={k:v[i] for k,v in states.items()},
                                       actions={}, text=o['prompt'], embodiment=tag)
                    samples.append(self.processor([{'type':MessageType.EPISODE_STEP.value,'content':step}]))
                batch = self.processor.collator(samples)
                output = self.model.get_action(**batch)['action_pred'].float().cpu().numpy()
                decoded = self.processor.decode_action(output,tag,states)
                actions = np.concatenate([decoded[k] for k in KEYS],axis=-1)
        assert actions.ndim==3 and actions.shape[-1]==7 and np.isfinite(actions).all()
        return [{'actions':a.astype(np.float32)} for a in actions]


def worker(task, count, destination, port):
    import numpy as np
    from libero.libero import benchmark
    from post_train_vla.evaluator import evaluate, EvalConfig
    from post_train_vla.policy import WebsocketPolicy
    suite = benchmark.get_benchmark_dict()['libero_spatial']()
    states = suite.get_task_init_states(task)
    identities = [hashlib.sha256(np.asarray(s,dtype='<f8').tobytes()).hexdigest() for s in states[:count]]
    class ContextPolicy:
        def __init__(self):
            self.client = WebsocketPolicy(f'ws://127.0.0.1:{port}')
            self.episode = -1
        def reset(self):
            self.episode += 1
            self.request=0
        def infer(self, observation):
            seed=7+task*1000000+self.episode*1000+self.request
            self.request+=1
            return self.client.infer({**observation,'_evaluation_noise_seed':seed})
    policy=ContextPolicy()
    try:
        evaluate(policy, EvalConfig(task_ids=(task,),episodes_per_task=count,resize_size=256,
                 replan_steps=5,output_dir=destination), policy.client.metadata)
    finally:
        policy.client.close()
    (destination/'initial_state_hashes.json').write_text(json.dumps(identities,indent=2)+'\n')


def simulate(args):
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(worker,t,args.episodes,args.output/f'task_{t:02d}',args.port) for t in range(args.tasks)]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    rows=[]
    for p in sorted(args.output.glob('task_*/episodes.jsonl')):
        rows.extend(json.loads(line) for line in p.read_text().splitlines())
    if len(rows)!=args.tasks*args.episodes or any(r.get('error') for r in rows):
        raise RuntimeError('Incomplete evaluation or runtime errors; success rate will not be published')
    if len({(r['task_id'],r['episode_index']) for r in rows})!=len(rows):
        raise RuntimeError('Duplicate episodes')
    summary={'episodes':len(rows),'successes':sum(r['success'] for r in rows),'epoch':args.epoch,
             'version':args.version,'status':'complete','suite':'libero_spatial',
             'episodes_per_task':args.episodes,'seed':7,'max_policy_steps':220,'replan_steps':5,
             'per_task':[{'task_id':t,'successes':sum(r['success'] for r in rows if r['task_id']==t),
                          'episodes':args.episodes} for t in range(args.tasks)]}
    summary['success_rate']=summary['successes']/len(rows)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


async def serve_and_eval(args):
    from post_train_vla.policy_server import PolicyServer
    policy=SavedPolicy(args.version,args.checkpoint)
    server=asyncio.create_task(PolicyServer(policy,host='127.0.0.1',port=args.port,
                 max_batch_size=args.workers,batch_wait_ms=10).run())
    command=[str(args.sim_python),'-m','post_train_vla.groot_evaluation','sim','--output',str(args.output),
             '--version',args.version,'--epoch',str(args.epoch),'--episodes',str(args.episodes),
             '--tasks',str(args.tasks),'--workers',str(args.workers),'--port',str(args.port)]
    env=os.environ.copy()
    env.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    process=None
    try:
        await asyncio.sleep(.1)
        if server.done(): await server
        with (args.output/'rollout.log').open('w') as log:
            process=await asyncio.create_subprocess_exec(*command,env=env,stdout=log,stderr=subprocess.STDOUT)
            waiter=asyncio.create_task(process.wait())
            done,_=await asyncio.wait([server,waiter],return_when=asyncio.FIRST_COMPLETED)
            if server in done: await server
            code=await waiter
            if code: raise RuntimeError(f'Rollout failed ({code}): {args.output}/rollout.log')
    finally:
        if process is not None and process.returncode is None:
            process.terminate(); await process.wait()
        server.cancel()
        with contextlib.suppress(asyncio.CancelledError): await server


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['model','sim'])
    p.add_argument('--version',required=True,choices=['n1','n1d5','n1d6','n1d7'])
    p.add_argument('--sim-python',type=Path,help='Interpreter with LIBERO and post_train_vla installed')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epoch',type=int,default=8)
    p.add_argument('--episodes',type=int,default=50)
    p.add_argument('--tasks',type=int,default=10)
    p.add_argument('--workers',type=int,default=10)
    p.add_argument('--port',type=int,default=8975)
    args=p.parse_args(argv)
    if min(args.episodes,args.tasks,args.workers)<1 or args.tasks>10 or args.episodes>50:
        p.error("Spatial evaluation requires 1–10 tasks, 1–50 initial states, and positive workers")
    if args.mode=="model" and (args.checkpoint is None or args.sim_python is None):
        p.error("model mode requires --checkpoint and --sim-python")
    if any(args.output.glob("task_*/episodes.jsonl")):
        p.error("Use a fresh output directory to avoid combining evaluations")
    args.output.mkdir(parents=True,exist_ok=True)
    if args.mode=='sim': simulate(args)
    else: asyncio.run(serve_and_eval(args))


if __name__=='__main__': main()
