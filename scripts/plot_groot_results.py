#!/usr/bin/env python3
"""Export results.json and JSON on stdout by default, plus CSVs and plots.

Incomplete runs are explicitly marked partial and never plotted as zero.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS = {'n1': 'GR00T N1', 'n15': 'GR00T N1.5', 'n16': 'GR00T N1.6', 'n17': 'GR00T N1.7'}
BUDGETS = [5, 10, 15, 25, 50]


def write_results_json(output, suite, records, task_rows):
    episode_counts = {r['episodes_per_task'] for r in records}
    if len(episode_counts) > 1:
        raise ValueError('Cannot combine checkpoints with different episode counts')
    episodes_per_task = next(iter(episode_counts), None)
    expected = {(version, budget) for version in LABELS for budget in BUDGETS}
    present = {(r['version'], r['trajectory_count']) for r in records}
    if len(present) != len(records) or not present.issubset(expected):
        raise ValueError('Duplicate or unexpected checkpoint results')
    report = {
        'status': 'complete' if present == expected else 'partial',
        'suite': suite,
        'checkpoints': len(records),
        'expected_checkpoints': len(expected),
        'tasks': 10,
        'episodes_per_task': episodes_per_task,
        'episodes_per_checkpoint': episodes_per_task * 10 if episodes_per_task is not None else None,
        'total_episodes': sum(r['episodes'] for r in records),
        'runtime_errors': 0,
        'results_provisional': True,
        'caveat': 'Normalization was reconstructed from LIBERO Long. Original checkpoint preprocessing '
                  'and training dataset provenance remain unverified.',
        'success_rate_units': 'fraction',
        'results': {
            label: [
                {'trajectories': r['trajectory_count'], 'successes': r['successes'],
                 'success_rate': r['success_rate']}
                for r in records if r['version'] == version
            ]
            for version, label in LABELS.items()
        },
    }
    for filename, value in [('results.json', report), ('per_task.json', task_rows)]:
        path = output/filename
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
        temporary.replace(path)
    return report


def wilson(successes, n):
    z = 1.959963984540054
    p = successes / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return max(0.0, center-half), min(1.0, center+half)


def export(root, suite):
    records = [json.loads(p.read_text()) for p in (root/'results'/suite).glob('*/trajectories-*/complete.json')]
    records.sort(key=lambda r: (list(LABELS).index(r['version']), r['trajectory_count']))
    output = root/'plots'/suite
    output.mkdir(parents=True, exist_ok=True)
    rows, task_rows = [], []
    for r in records:
        low, high = wilson(r['successes'], r['episodes'])
        rows.append(dict(model=LABELS[r['version']], trajectories=r['trajectory_count'],
                         successes=r['successes'], episodes=r['episodes'], success_rate=r['success_rate'],
                         wilson_95_low=low, wilson_95_high=high, seconds=r['elapsed_seconds'],
                         gpu=r['gpu'], revision=r['revision'], metadata_reconstructed=True))
        for task, value in r['per_task'].items():
            task_rows.append(dict(model=LABELS[r['version']], trajectories=r['trajectory_count'],
                                  task_id=task, **value, success_rate=value['successes']/value['episodes']))
    report = write_results_json(output, suite, records, task_rows)
    if not rows:
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
        return report
    episodes_per_task = report['episodes_per_task']
    for filename, data in [('success_rates.csv', rows), ('per_task.csv', task_rows)]:
        with (output/filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(9, 5.5), layout='constrained')
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    for (version, label), color in zip(LABELS.items(), colors):
        selected = [r for r in rows if r['model'] == label]
        if not selected:
            continue
        x = [r['trajectories'] for r in selected]
        y = np.array([100*r['success_rate'] for r in selected])
        low = np.array([100*r['wilson_95_low'] for r in selected])
        high = np.array([100*r['wilson_95_high'] for r in selected])
        ax.errorbar(x, y, yerr=np.maximum(0, [y-low, high-y]), label=label, marker='o', capsize=4,
                    color=color, linewidth=2, markersize=6)
    title = 'LIBERO Long' if suite == 'libero_10' else 'LIBERO Spatial'
    suffix = '' if len(records) == 20 else f' — partial ({len(records)}/20 checkpoints)'
    ax.set(title=title+' trajectory efficiency'+suffix, xlabel='Training trajectories (total)',
           ylabel='Task success (%)', xticks=BUDGETS, xlim=(2, 53), ylim=(-2, 102))
    ax.grid(axis='y', alpha=.2)
    ax.legend(loc='best')
    fig.get_layout_engine().set(rect=(0, .12, 1, .86))
    fig.text(.5, .025, f'{episodes_per_task} initial states × 10 tasks per checkpoint • seed 7 • 95% Wilson intervals\n'
             'Provisional: normalization reconstructed from LIBERO Long; original checkpoint metadata unavailable.',
             ha='center', fontsize=9, color='#555555')
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(output/f'success_vs_trajectories.{extension}', dpi=180)
    plt.close(fig)
    # Separate panels keep identical/overlapping curves visible.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True, layout='constrained')
    for ax, (version, label), color in zip(axes.flat, LABELS.items(), colors):
        selected = [r for r in rows if r['model'] == label]
        if selected:
            x = [r['trajectories'] for r in selected]
            y = np.array([100*r['success_rate'] for r in selected])
            low = np.array([100*r['wilson_95_low'] for r in selected])
            high = np.array([100*r['wilson_95_high'] for r in selected])
            ax.errorbar(x, y, yerr=np.maximum(0, [y-low, high-y]), marker='o', capsize=4, color=color)
            for xvalue, yvalue, row in zip(x, y, selected):
                ax.annotate(f'{row["successes"]}/{row["episodes"]}', (xvalue, yvalue),
                            xytext=(0, 9), textcoords='offset points', ha='center', fontsize=8)
        ax.set(title=label, xticks=BUDGETS, xlim=(2, 53), ylim=(-3, 108))
        ax.grid(axis='y', alpha=.2)
    fig.supxlabel('Training trajectories (total)')
    fig.supylabel('Task success (%)')
    fig.suptitle(title+' trajectory efficiency'+suffix)
    fig.savefig(output/'success_by_model.png', dpi=180)
    fig.savefig(output/'success_by_model.pdf')
    plt.close(fig)
    matrix = np.full((20, 10), np.nan)
    for r in records:
        row = list(LABELS).index(r['version'])*5 + BUDGETS.index(r['trajectory_count'])
        for task, value in r['per_task'].items():
            matrix[row, int(task)] = 100*value['successes']/value['episodes']
    fig, ax = plt.subplots(figsize=(10, 10), layout='constrained')
    cmap = plt.get_cmap('YlGnBu').copy()
    cmap.set_bad('#eeeeee')
    im = ax.imshow(matrix, vmin=0, vmax=100, cmap=cmap, aspect='auto')
    ax.set(xticks=range(10), xlabel='Task ID', yticks=range(20),
           yticklabels=[f'{label} / {budget}' for label in LABELS.values() for budget in BUDGETS],
           title=title+' success by task'+suffix)
    for i in range(20):
        for j in range(10):
            value = matrix[i,j]
            ax.text(j, i, '—' if np.isnan(value) else f'{value:.0f}', ha='center', va='center',
                    fontsize=8, color='white' if value > 60 else '#222222')
    fig.colorbar(im, ax=ax, label='Success (%)', shrink=.65)
    fig.savefig(output/'per_task_heatmap.png', dpi=180)
    fig.savefig(output/'per_task_heatmap.pdf')
    plt.close(fig)
    lines = [f'# {title} checkpoint evaluation', '', f'Completed: {len(records)}/20 checkpoints.', '',
             '| Model | 5 | 10 | 15 | 25 | 50 |', '|---|---:|---:|---:|---:|---:|']
    for version, label in LABELS.items():
        rates = {r['trajectory_count']: f'{100*r["success_rate"]:.1f}%' for r in records if r['version'] == version}
        lines.append('| '+label+' | '+' | '.join(rates.get(b, 'pending') for b in BUDGETS)+' |')
    lines += ['', '## Protocol', '',
              f'- {episodes_per_task} initial states (0–{episodes_per_task-1}) per task, ten tasks, seed 7; no videos.',
              '- Five-step replanning, ten settling steps, 256px source cameras; native model preprocessing and diffusion steps.',
              f'- Full benchmark horizon: {520 if suite == "libero_10" else 220} policy steps.',
              '- One checkpoint per GPU. GPU assignment, pinned checkpoint revision, and runtime saved in CSV and JSON.',
              '- The actual training dataset could not be independently verified: uploaded repositories contain weights only.',
              '- `/root/libero_long_post` was initially identified as the training dataset in the session, so its statistics were used.',
              '- Configuration/processor files were reconstructed from the official base configs and local training recipe;',
              '  statistics were recomputed from all 379 episodes / 101469 frames, matching the code’s shared normalization.',
              '- These results are provisional: original preprocessing was unavailable for equality checks. If the checkpoints',
              '  used Spatial normalization during training, these Long-normalized evaluations do not establish their performance.',
              '- Wilson intervals summarize rollout sampling only; they do not measure variation across training seeds or tasks.',
              '- Failed or incomplete evaluations are omitted, never recorded as zero success.', '']
    (output/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1]/'outputs/groot_eval')
    p.add_argument('--suite', default='libero_spatial')
    a = p.parse_args()
    import fcntl
    with (a.root/'plot.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        export(a.root, a.suite)
