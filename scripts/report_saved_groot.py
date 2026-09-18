#!/usr/bin/env python3
"""Validate completed saved-checkpoint evaluations and export JSON, CSV, and plots."""
import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

VERSIONS = ['1', '1.5', '1.6', '1.7']
BUDGETS = [5, 10, 15, 25, 50]
COLORS = ['#2563eb', '#d97706', '#059669', '#9333ea']
TRAJECTORY_COLORS = ['#2563eb', '#f97316', '#16a34a', '#dc2626', '#7c3aed']


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temporary.replace(path)


def csv_file(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_epoch_curves(checkpoints, plots, group_by='version', versions=None):
    """Group epoch curves by model or trajectory count; mark measured maxima."""
    versions = VERSIONS if versions is None else versions
    if group_by == 'version':
        budgets = sorted({r['trajectory_count'] for r in checkpoints})
        groups, series_key, series_values = versions, 'trajectory_count', budgets
        colors = {budget: TRAJECTORY_COLORS[i % len(TRAJECTORY_COLORS)] for i, budget in enumerate(budgets)}
        series_labels = {budget: str(budget) for budget in budgets}
        legend_title = 'Training trajectories'
        subtitle = 'Each line is a different number of training trajectories · stars mark the best checkpoint'
        peaks_filename = 'success_by_epoch_peaks.json'
    elif group_by == 'trajectory_count':
        groups, series_key, series_values = sorted({r['trajectory_count'] for r in checkpoints}), 'version', versions
        colors = dict(zip(VERSIONS, COLORS))
        series_labels = {version: 'GR00T N'+version for version in versions}
        legend_title = 'Model'
        subtitle = 'Each line is a different model · stars mark the best checkpoint'
        peaks_filename = 'success_by_epoch_trajectory_peaks.json'
    else:
        raise ValueError(f'Unsupported epoch grouping: {group_by}')
    identities = [(r['version'], r['trajectory_count'], r['epoch']) for r in checkpoints]
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate model/trajectory/epoch results')
    if any(r['version'] not in VERSIONS or r['epoch'] < 1 for r in checkpoints):
        raise ValueError('Unsupported model version or invalid epoch')
    plots.mkdir(parents=True, exist_ok=True)
    peaks = []
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        for group in groups:
            rows = [r for r in checkpoints if r[group_by] == group and r['version'] in versions]
            if not rows:
                continue
            if any(r['episodes'] <= 0 or not 0 <= r['successes'] <= r['episodes'] or not np.isclose(r['success_rate'], r['successes']/r['episodes']) for r in rows):
                raise ValueError('Epoch charts require valid measured success counts and episode denominators')
            maximum_epoch = max(r['epoch'] for r in rows)
            fig, ax = plt.subplots(figsize=(12.8, 7.75), dpi=160)
            fig.subplots_adjust(left=.082, right=.985, bottom=.15, top=.79)
            fig.patch.set_facecolor('white')
            ax.set_facecolor('#f8fafc')
            ax.set_axisbelow(True)
            ax.grid(axis='y', color='#d7e0ea', linewidth=1)
            ax.tick_params(colors='#526178', length=5)
            for spine in ('left', 'bottom'):
                ax.spines[spine].set_color('#94a3b8')
            ax.set(xlim=(.75, maximum_epoch+.25), ylim=(0, 100),
                   xticks=range(1, maximum_epoch+1), yticks=range(0, 101, 10))
            ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
            ax.set_xlabel('Training epoch', fontsize=14, labelpad=14)
            ax.set_ylabel('Success rate', fontsize=14, labelpad=12)
            labels = {}
            for index, series in enumerate(series_values):
                color = colors[series]
                points = sorted((r for r in rows if r[series_key] == series), key=lambda r: r['epoch'])
                if not points:
                    continue
                # Different dashes and open markers keep coincident model curves visible.
                style = ['-', '--', '-.', ':'][index] if group_by == 'trajectory_count' else '-'
                marker = ['o', 's', '^', 'D'][index] if group_by == 'trajectory_count' else 'o'
                ax.plot([r['epoch'] for r in points], [100*r['success_rate'] for r in points],
                        color=color, linewidth=2.8, linestyle=style, marker=marker, markersize=7,
                        markerfacecolor='none' if group_by == 'trajectory_count' else color,
                        markeredgecolor=color if group_by == 'trajectory_count' else 'white',
                        markeredgewidth=1.4, label=series_labels[series], clip_on=False)
                peak = max(points, key=lambda r: (r['success_rate'], -r['epoch']))
                rate = 100*peak['success_rate']
                labels.setdefault((peak['epoch'], rate), []).append((series, color))
                peaks.append({k: peak[k] for k in ('version', 'trajectory_count', 'epoch', 'successes', 'episodes', 'success_rate')})
            ax.legend(title=legend_title, ncol=len(series_values), loc='upper left',
                      frameon=False, fontsize=12, title_fontsize=13,
                      columnspacing=1.5, handlelength=2.2, borderaxespad=.8)
            title = (f'GR00T N{group} LIBERO-Spatial: Success Rate by Training Epoch'
                     if group_by == 'version' else f'LIBERO-Spatial: {group} Training Trajectories')
            fig.text(.082, .963, title,
                     fontsize=20, fontweight='bold', color='#0f172a', va='top')
            fig.text(.082, .895, subtitle,
                     fontsize=12.5, color='#64748b', va='top')
            fig.text(.082, .045, 'Stars: highest measured success; earliest epoch on ties.', fontsize=9, color='#64748b')
            fig.text(.985, .045, 'Evaluation episodes/checkpoint: '+', '.join(str(n) for n in sorted({r['episodes'] for r in rows})), ha='right', fontsize=10.5, color='#64748b')
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            occupied = []
            for (epoch, rate), series in sorted(labels.items(), key=lambda item: item[0][1]):
                color = series[0][1] if len(series) == 1 else '#475569'
                label = f'{rate:.2f}'.rstrip('0').rstrip('.')+'%'
                if len(series) > 1:
                    names = (', '.join(str(value) for value, _ in series)+' trajectories'
                             if group_by == 'version' else ', '.join('N'+value for value, _ in series))
                    label += ' ('+names+')'
                ax.scatter([epoch], [rate], color=color, marker='*', s=210,
                           edgecolor='white', linewidth=1.1, zorder=5, clip_on=False)
                offset = 43 if len(series) > 1 else 13
                alignment = ('left' if epoch <= (maximum_epoch+1)/2 else 'right') if len(series) > 1 else 'center'
                annotation = ax.annotate(label, (epoch, rate), xytext=(0, offset), textcoords='offset points',
                                         ha=alignment, va='bottom',
                                         color=color, fontsize=11, fontweight='bold', zorder=6,
                                         bbox={'facecolor': ax.get_facecolor(), 'edgecolor': 'none', 'pad': .2})
                bounds = annotation.get_window_extent(renderer).expanded(1.08, 1.2)
                while any(bounds.overlaps(previous) for previous in occupied):
                    offset += 15
                    annotation.set_position((0, offset))
                    bounds = annotation.get_window_extent(renderer).expanded(1.08, 1.2)
                if offset > 13:
                    ax.annotate('', (epoch, rate), xytext=(0, offset-2), textcoords='offset points',
                                arrowprops={'arrowstyle': '-', 'color': color, 'linewidth': .8})
                occupied.append(bounds)
            stem = ('success_by_epoch_n'+group.replace('.', '_') if group_by == 'version'
                    else f'success_by_epoch_trajectories_{group:03d}')
            for extension in ('png', 'pdf', 'svg'):
                fig.savefig(plots / f'{stem}.{extension}', facecolor='white')
            plt.close(fig)
    write_json(plots / peaks_filename, {
        'selection': 'Highest measured success rate; earliest epoch on ties.',
        'group_by': group_by, 'versions': versions,
        'peaks': peaks})


def export(out):
    with (out / 'report.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _export(out)


def _export(out):
    manifest = read_json(out / 'run_manifest.json')
    initial = read_json(out / 'initial_states.json')
    availability_path = out / 'artifact_availability.json'
    keys = {c['key'] for c in manifest['checkpoints']}
    unavailable = [r for r in read_json(availability_path) if r['checkpoint'] in keys and not r['all_weight_objects_available']] if availability_path.exists() else []
    aggregate, task_rows, records, errors, processes, pending = [], [], [], [], [], []
    grid = {(t, e) for t in range(10) for e in range(40)}
    for checkpoint in manifest['checkpoints']:
        destination = out / 'results' / checkpoint['key']
        error_path = destination / 'process_errors.jsonl'
        if error_path.exists():
            processes.extend({**json.loads(line), 'checkpoint': checkpoint['key']} for line in error_path.read_text().splitlines())
        error_path = destination / 'runtime_errors.jsonl'
        if error_path.exists():
            errors.extend({**json.loads(line), 'checkpoint': checkpoint['key']} for line in error_path.read_text().splitlines())
        summary_path = destination / 'summary.json'
        if not summary_path.exists() or read_json(summary_path)['status'] != 'complete':
            pending.append(checkpoint['key'])
            continue
        summary = read_json(summary_path)
        rows = [json.loads(line) for line in (destination / 'episodes.jsonl').read_text().splitlines()]
        if len(rows) != 400 or {(r['task_id'], r['episode_index']) for r in rows} != grid:
            raise ValueError(f'Incomplete or duplicate episode identities: {checkpoint["key"]}')
        if any(r.get('error') or r['seed'] != 7 or r['checkpoint'] != checkpoint['key'] or
               r['initial_state_sha256'] != initial['tasks'][str(r['task_id'])]['states'][str(r['episode_index'])]
               for r in rows):
            raise ValueError(f'Episode integrity/protocol mismatch: {checkpoint["key"]}')
        if summary['successes'] != sum(r['success'] for r in rows):
            raise ValueError('Summary does not match episode records')
        loading = read_json(destination / 'load_verification.json')
        artifacts = read_json(destination / 'artifact_verification.json')
        if not all(loading.get(k) is True for k in ['strict_weight_load', 'saved_statistics_match', 'saved_embodiment_matches']):
            raise ValueError('Saved model metadata verification did not pass')
        if loading['training'] or loading['trainable_parameters'] or not artifacts['runtime_inference_matches']:
            raise ValueError('Evaluation mode or saved runtime verification did not pass')
        base = dict(checkpoint=checkpoint['key'], version=checkpoint['version'], trajectory_count=checkpoint['trajectory_count'],
                    epoch=checkpoint['epoch'], repo_id=checkpoint['repo_id'], revision=checkpoint['revision'], prefix=checkpoint['prefix'],
                    is_repository_best=checkpoint.get('is_repository_best', True), is_final_epoch=checkpoint.get('is_final_epoch', False),
                    evaluated_gpu=summary['checkpoint']['gpu'])
        aggregate.append({**base, **{k: summary[k] for k in ['episodes', 'successes', 'unsuccessful', 'success_rate', 'runtime_error_attempts', 'elapsed_seconds']}})
        task_rows.extend({**base, **task} for task in summary['per_task'])
        records.extend(rows)

    total_expected = len(manifest['checkpoints']) * 400
    scope_finalized = manifest.get('scope_finalized', True)
    complete = scope_finalized and not pending and len(records) == total_expected
    blocked = {r['checkpoint'] for r in unavailable}
    status = 'complete' if complete else 'incomplete_missing_weights' if scope_finalized and set(pending) <= blocked else 'in_progress'
    summary = dict(status=status, expected_checkpoints=len(manifest['checkpoints']),
                   complete_checkpoints=len(aggregate), expected_episodes=total_expected, episodes=len(records),
                   successes=sum(r['success'] for r in records), unsuccessful=sum(not r['success'] for r in records),
                   runtime_error_attempts=len(errors), process_errors=len(processes), pending=pending, unavailable_checkpoints=unavailable,
                   checkpoints=aggregate, protocol=manifest['protocol'], selection_note=manifest['selection_note'],
                   scope_finalized=scope_finalized, scope_update=manifest.get('scope_update'))
    trajectory_rows = []
    for version in VERSIONS:
        for budget in BUDGETS:
            expected = [c for c in manifest['checkpoints'] if c['version'] == version and c['trajectory_count'] == budget]
            measured = [r for r in aggregate if r['version'] == version and r['trajectory_count'] == budget]
            rates = [r['success_rate'] for r in measured]
            trajectory_rows.append(dict(version=version, trajectory_count=budget,
                retained_epochs=len(expected), evaluated_epochs=len(measured),
                unavailable_epochs=sum(c['key'] in blocked for c in expected),
                episodes=sum(r['episodes'] for r in measured), successes=sum(r['successes'] for r in measured),
                mean_epoch_success_rate=float(np.mean(rates)) if rates else None,
                minimum_epoch_success_rate=min(rates) if rates else None,
                maximum_epoch_success_rate=max(rates) if rates else None))
    write_json(out / 'trajectory_summary.json', trajectory_rows)
    csv_file(out / 'trajectory_summary.csv', trajectory_rows)
    write_json(out / 'summary.json', summary)
    write_json(out / 'per_task.json', task_rows)
    historical_path = out / 'historical_runtime_errors.json'
    historical = read_json(historical_path) if historical_path.exists() else None
    historical_counts = (dict(file=historical_path.name,
        episode_error_attempts=historical['episode_error_attempts'],
        process_error_attempts=historical['process_error_attempts']) if historical else None)
    write_json(out / 'runtime_errors.json', dict(episode_errors=errors, process_errors=processes,
               historical_attempts=historical_counts,
               previous_interrupted_run=manifest.get('previous_interrupted_run'),
               note='Runtime errors are excluded from unsuccessful counts and success-rate denominators. Retry outcomes use the same task/state/noise seeds.'))
    for name, rows in [('aggregate.csv', aggregate), ('per_task.csv', task_rows), ('episodes.csv', records)]:
        csv_file(out / name, rows)
    (out / 'episodes.jsonl').write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in records))
    for filename in ('episodes.jsonl', 'run_manifest.json', 'initial_states.json'):
        summary[filename.replace('.', '_') + '_sha256'] = hashlib.sha256((out / filename).read_bytes()).hexdigest()
    write_json(out / 'validation.json', dict(complete=complete, validated_checkpoints=len(aggregate),
               validated_episodes=len(records), expected_episodes=total_expected, checks=[
                   'Exactly 400 unique task/initial-state identities per complete checkpoint',
                   '10 tasks x 40 episodes, seed 7, saved initial-state hashes',
                   'Exact native weight loading and saved statistics/embodiment equality',
                   'Evaluation mode, zero trainable parameters, matching saved inference runtime',
                   'Runtime errors excluded from unsuccessful episodes',
                   'Aggregates match raw episode records'],
               hashes={k: v for k, v in summary.items() if k.endswith('_sha256')}))

    plots = out / 'plots'
    plots.mkdir(exist_ok=True)
    plot_epoch_curves(aggregate, plots)
    maximum_epoch = max(c['epoch'] for c in manifest['checkpoints'])
    plt.rcParams.update({'figure.dpi': 130, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    for version, color in zip(VERSIONS, COLORS):
        points = sorted((r for r in aggregate if r['version'] == version and r['is_repository_best']), key=lambda r: r['trajectory_count'])
        ax.plot([r['trajectory_count'] for r in points], [100*r['success_rate'] for r in points],
                marker='o', lw=2, color=color, label='GR00T N'+version)
    ax.set(xlabel='Training trajectories', ylabel='Success rate (%)', xticks=BUDGETS, ylim=(-1, 101),
           title='Repository-preselected best epochs: 400 episodes per checkpoint')
    ax.grid(alpha=.2)
    ax.legend()
    if not complete:
        ax.text(.99, .02, f'Incomplete: {len(aggregate)}/{len(manifest["checkpoints"])} checkpoints',
                transform=ax.transAxes, ha='right', color='gray')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(plots / ('success_vs_trajectories.'+ext))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5), layout='constrained')
    for version, color in zip(VERSIONS, COLORS):
        points = [r for r in trajectory_rows if r['version'] == version and r['evaluated_epochs']]
        x = [r['trajectory_count'] for r in points]
        ax.plot(x, [100*r['mean_epoch_success_rate'] for r in points], marker='o', lw=2,
                color=color, label='GR00T N'+version)
        ax.fill_between(x, [100*r['minimum_epoch_success_rate'] for r in points],
                        [100*r['maximum_epoch_success_rate'] for r in points], color=color, alpha=.10)
    ax.set(xlabel='Training trajectories', ylabel='Success rate (%)', xticks=BUDGETS, ylim=(-1,101),
           title='Mean across evaluated retained epochs; shading shows epoch range')
    ax.grid(alpha=.2)
    ax.legend()
    ax.text(.99, .02, f'{len(aggregate)}/{len(manifest["checkpoints"])} epochs evaluated; 400 episodes per epoch',
            transform=ax.transAxes, ha='right', fontsize=9, color='gray')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(plots / ('mean_retained_epoch_success_vs_trajectories.'+ext))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    for version, color in zip(VERSIONS, COLORS):
        points = sorted((r for r in aggregate if r['version']==version and r['is_final_epoch']), key=lambda r: r['trajectory_count'])
        ax.plot([r['trajectory_count'] for r in points], [100*r['success_rate'] for r in points],
                marker='o', lw=2, color=color, label='GR00T N'+version)
    ax.set(xlabel='Training trajectories', ylabel='Success rate (%)', xticks=BUDGETS, ylim=(-1,101),
           title='Final retained epochs: 400 episodes per checkpoint')
    ax.grid(alpha=.2)
    ax.legend()
    for ext in ('png','pdf','svg'):
        fig.savefig(plots / ('final_epoch_success_vs_trajectories.'+ext))
    plt.close(fig)

    fig, axes = plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True,layout='constrained')
    for version, ax in zip(VERSIONS,axes.flat):
        points = [r for r in aggregate if r['version']==version]
        # Slight horizontal offsets separate retained epochs at the same budget.
        ax.scatter([r['trajectory_count']+(r['epoch']-1)*.22 for r in points], [100*r['success_rate'] for r in points],
                   c=[r['epoch'] for r in points], cmap='viridis',vmin=1,vmax=maximum_epoch,s=45,edgecolor='white')
        ax.set(title='GR00T N'+version,xticks=BUDGETS,ylim=(-3,103))
        ax.grid(alpha=.2)
    fig.supxlabel('Training trajectories (small offsets separate epochs)')
    fig.supylabel('Success rate (%)')
    fig.suptitle('Every evaluated epoch')
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(1,maximum_epoch),cmap='viridis'),
                 ax=list(axes.flat),label='Epoch number',shrink=.75)
    for ext in ('png','pdf','svg'):
        fig.savefig(plots / ('all_epochs_success_vs_trajectories.'+ext))
    plt.close(fig)

    fig, axes = plt.subplots(2,2,figsize=(13,8),sharex=True,sharey=True,layout='constrained')
    for version, ax in zip(VERSIONS,axes.flat):
        for budget, color in zip(BUDGETS,plt.cm.tab10.colors):
            points = sorted((r for r in aggregate if r['version']==version and r['trajectory_count']==budget),key=lambda r:r['epoch'])
            ax.plot([r['epoch'] for r in points],[100*r['success_rate'] for r in points],marker='o',
                    color=color,label=f'{budget} trajectories',markersize=4)
        ax.set(title='GR00T N'+version,xticks=range(1,maximum_epoch+1),ylim=(-1,101))
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.supxlabel('Retained training epoch')
    fig.supylabel('Success rate (%)')
    fig.suptitle('All epochs, grouped by training trajectory count')
    for ext in ('png','pdf'):
        fig.savefig(plots / ('success_by_epoch_panels.'+ext))
    plt.close(fig)

    styles = ['-', '--', '-.', ':', (0, (5, 1, 1, 1, 1, 1))]
    markers = ['o', 's', '^', 'D', 'X']
    fig, ax = plt.subplots(figsize=(12, 6.5), layout='constrained')
    for version, color in zip(VERSIONS, COLORS):
        for budget, style, marker in zip(BUDGETS, styles, markers):
            points = {r['epoch']: r['success_rate'] for r in aggregate
                      if r['version'] == version and r['trajectory_count'] == budget}
            if not points:
                continue
            epochs = sorted(c['epoch'] for c in manifest['checkpoints']
                            if c['version'] == version and c['trajectory_count'] == budget)
            ax.plot(epochs, [100*points[e] if e in points else np.nan for e in epochs],
                    color=color, linestyle=style, marker=marker, linewidth=1.8, markersize=5,
                    markerfacecolor='none', markeredgewidth=1.1, alpha=.85)
    model_legend = ax.legend(handles=[Line2D([], [], color=color, lw=2, label='GR00T N'+version)
                                     for version, color in zip(VERSIONS, COLORS)],
                             title='Model — color', loc='upper left', bbox_to_anchor=(1.01, 1))
    ax.add_artist(model_legend)
    ax.legend(handles=[Line2D([], [], color='#333333', linestyle=style, marker=marker,
                             markerfacecolor='none', label=f'{budget} trajectories')
                       for budget, style, marker in zip(BUDGETS, styles, markers)],
              title='Training data — line / marker', loc='upper left', bbox_to_anchor=(1.01, .58))
    ax.set(xlabel='Training epoch', ylabel='Success rate (%)', xticks=range(1, maximum_epoch+1),
           xlim=(.7, maximum_epoch+.3), ylim=(-2, 102), title='Success rate by training epoch — all models and trajectory counts')
    ax.grid(alpha=.2)
    epoch_note = (f'{len(aggregate)}/{len(manifest["checkpoints"])} checkpoints evaluated · 400 episodes per point'
                  if scope_finalized else f'{len(aggregate)} checkpoints evaluated · inventory still growing · 400 episodes per point')
    ax.text(.01, .98, epoch_note,
            transform=ax.transAxes, va='top', fontsize=9, color='#555555')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(plots / ('success_by_epoch.'+ext))
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5), sharex=True, sharey=True, layout='constrained')
    for task, ax in enumerate(axes.flat):
        for version, color in zip(VERSIONS, COLORS):
            rows = sorted((r for r in task_rows if r['version']==version and r['task_id']==task and r['is_repository_best']), key=lambda r: r['trajectory_count'])
            ax.plot([r['trajectory_count'] for r in rows], [100*r['success_rate'] for r in rows], color=color, marker='o', label='N'+version)
        ax.set(title=f'Task {task}', xticks=BUDGETS, ylim=(-1, 101))
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.supxlabel('Training trajectories')
    fig.supylabel('Success rate (%)')
    fig.suptitle('Repository-preselected epochs: per-task success (40 episodes per point)')
    for ext in ('png', 'pdf'):
        fig.savefig(plots / ('per_task_success_vs_trajectories.'+ext))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    values = np.full((4, 5), np.nan)
    for r in aggregate:
        if r['is_repository_best']:
            values[VERSIONS.index(r['version']), BUDGETS.index(r['trajectory_count'])] = 100*r['success_rate']
    im = ax.imshow(values, vmin=0, vmax=100, cmap='Blues', aspect='auto')
    for row in range(4):
        for column in range(5):
            value = values[row, column]
            key = next((c['key'] for c in manifest['checkpoints'] if c['version']==VERSIONS[row]
                       and c['trajectory_count']==BUDGETS[column] and c.get('is_repository_best',True)), None)
            label = f'{value:.2f}%' if np.isfinite(value) else 'missing weights' if key in blocked else 'pending'
            ax.text(column, row, label, ha='center', va='center', fontsize=9,
                    color='white' if value > 60 else 'black')
    ax.set(xticks=range(5), xticklabels=BUDGETS, yticks=range(4), yticklabels=['N'+v for v in VERSIONS],
           xlabel='Training trajectories', title='Repository-preselected epochs: aggregate success rate (%)')
    fig.colorbar(im, ax=ax, label='Success rate (%)')
    fig.savefig(plots / 'success_heatmap.png')
    plt.close(fig)

    lines = ['# GR00T LIBERO Spatial evaluation', '',
             f'Status: **{summary["status"]}** — {len(aggregate)}/{len(manifest["checkpoints"])} checkpoints, {len(records):,}/{total_expected:,} valid episodes.', '',
             'The table below uses repository-preselected best epochs; CSV/JSON records and the all-epochs plot contain every evaluated epoch.', '',
             '| Model | 5 trajectories | 10 | 15 | 25 | 50 |', '|---|---:|---:|---:|---:|---:|']
    if not scope_finalized:
        lines[4:4] = ['The checkpoint inventory is still growing. Totals cover currently registered checkpoints.', '']
    scope_note = (manifest.get('scope_update') or {}).get('note')
    if scope_note:
        lines[4:4] = [scope_note, '']
    for version in VERSIONS:
        cells = []
        for budget in BUDGETS:
            point = next((r for r in aggregate if r['version']==version and r['trajectory_count']==budget and r['is_repository_best']), None)
            selected = next((c for c in manifest['checkpoints'] if c['version']==version and c['trajectory_count']==budget and c.get('is_repository_best',True)), None)
            cells.append(f'{100*point["success_rate"]:.2f}% ({point["successes"]}/400)' if point else 'missing weights' if selected and selected['key'] in blocked else 'pending')
        lines.append('| GR00T N'+version+' | '+' | '.join(cells)+' |')
    lines += ['', '## Completed checkpoints', '',
              'Each number below is a training epoch with all 400 evaluation episodes complete. Ranges include both endpoints. The trajectory columns cover all five training budgets.', '',
              '| Model | 5 trajectories | 10 | 15 | 25 | 50 | Complete / retained |',
              '|---|---|---|---|---|---|---:|']
    for version in VERSIONS:
        cells = []
        for budget in BUDGETS:
            epochs = sorted(r['epoch'] for r in aggregate if r['version'] == version and r['trajectory_count'] == budget)
            ranges = []
            for epoch in epochs:
                if ranges and epoch == ranges[-1][-1] + 1:
                    ranges[-1].append(epoch)
                else:
                    ranges.append([epoch])
            cells.append(', '.join(str(r[0]) if len(r) == 1 else f'{r[0]}–{r[-1]}' for r in ranges) or 'None')
        done = sum(r['version'] == version for r in aggregate)
        retained = sum(c['version'] == version for c in manifest['checkpoints'])
        lines.append('| GR00T N'+version+' | '+' | '.join(cells)+f' | {done}/{retained} |')
    lines += ['', '## Protocol', '',
              'All 10 LIBERO Spatial tasks; initial states 0–39; seed 7; 10 settling steps; 220 policy steps; replan every 5 steps; two 256×256 camera images. Native checkpoint processors perform their own resizing. One model checkpoint per GPU, with 16 simulator workers and inference batches up to 8.', '',
              'Saved configurations, processor assets, embodiment IDs, and statistics were loaded offline and checked against their artifact hashes. No normalization statistics were reconstructed or borrowed. Saved action horizons and denoising settings were retained. Inference used bfloat16 autocast with the native parameter dtypes: mixed bfloat16/float32 for N1 and N1.5, and float32 for N1.6 and N1.7. Each loading receipt records the actual dtypes.', '',
              'Action-noise seeds are 7 + task_id × 1,000,000 + episode_index × 1,000 + inference_index. Each batch row uses an independent generator, preserving its seed when scheduling or retrying changes.', '',
              '## Checkpoint selection', '', manifest['selection_note'], '',
              'Each retained epoch has its own 400-episode estimate. The mean-retained-epoch plot averages the evaluated epochs within each model/trajectory combination; shading shows their minimum and maximum, not a confidence interval. The number of retained epochs varies by combination and is recorded in `trajectory_summary.csv`.', '',
              'Repository-best curves reuse the repository\'s checkpoint selection. Their selection data overlap the evaluation, so those curves do not estimate performance on entirely held-out selection data.', '',
              '## Runtime errors', '',
              f'This run: {len(errors)} episode runtime-error attempts and {len(processes)} process/download errors. Runtime errors are recorded separately; they are not counted as unsuccessful episodes. Missing episodes are retried with their original identities and seeds.', '',
              (f'Historical attempts include {historical["episode_error_attempts"]} episode errors and {historical["process_error_attempts"]} process/download errors. `historical_runtime_errors.json` preserves those records, their source locations, and source hashes. These historical errors are excluded from success-rate denominators and unsuccessful counts.'
               if historical else ''), '',
              '## Missing artifacts', '',
              (f'{len(unavailable)} checkpoints have missing Hub weight objects. `artifact_availability.json` records each missing shard and its expected SHA-256 hash. These checkpoints have no measured success rate and are not treated as failures.'
               if unavailable else 'All checkpoint weights and required metadata were available and verified. `artifact_availability.json` records the availability checks.'), '',
              ('See `MISSING_WEIGHTS.md` for artifact diagnostics.' if (out / 'MISSING_WEIGHTS.md').exists() else ''), '',
              '## Artifacts', '',
              '- `episodes.jsonl` and `episodes.csv`: all valid episode outcomes.',
              '- `aggregate.csv`, `per_task.csv`, `summary.json`, `per_task.json`: success counts and rates.',
              '- `trajectory_summary.csv` and `trajectory_summary.json`: evaluated/retained epoch counts and the mean/range across epochs for each trajectory count.',
              '- `runtime_errors.json`: separate errors and retry history.',
              '- `run_manifest.json`, `initial_states.json`, `validation.json`: revisions, protocol, state hashes, validation.',
              '- `results/<checkpoint>/`: raw worker records, model/simulator logs, checksums and strict loading receipts.',
              '- `plots/`: separate curves for repository-best, final, and mean retained epochs, all-epochs scatter plots (PNG/PDF/SVG), per-task best-epoch curves (PNG/PDF), and best-epoch heatmap (PNG).', '']
    lines.insert(-1, '- `plots/success_by_epoch.{png,pdf,svg}`: all models and trajectory counts on one epoch plot; `success_by_epoch_panels.{png,pdf}` keeps the separate model panels.')
    lines.insert(-1, '- `plots/success_by_epoch_n*.{png,pdf,svg}`: one epoch chart per model; stars mark each trajectory count\'s highest measured success rate.')
    if historical:
        lines.insert(-1, '- `historical_runtime_errors.json`: historical errors with their source records and hashes.')
    (out / 'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({k: summary[k] for k in ['status', 'complete_checkpoints', 'episodes', 'runtime_error_attempts', 'process_errors']}), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    epoch_plots = parser.add_mutually_exclusive_group()
    epoch_plots.add_argument('--model-epochs-only', action='store_true', help='Plot per-model epoch curves from an existing summary.json')
    epoch_plots.add_argument('--trajectory-epochs-only', action='store_true', help='Plot one model-comparison epoch chart per trajectory count from an existing summary.json')
    parser.add_argument('--versions', nargs='+', choices=VERSIONS, help='Models to include when plotting epoch curves (default: all four)')
    parser.add_argument('--trajectory-count', type=int, help='Training trajectory count for JSON records lacking it')
    epoch_plots.add_argument('--success-json', type=Path, help='Plot exported model/epoch/successes/episodes records')
    args = parser.parse_args()
    out = args.out.resolve()
    if args.success_json:
        raw = read_json(args.success_json)
        records = []
        for row in raw:
            count = row.get('trajectory_count', args.trajectory_count)
            if count is None or count < 1:
                parser.error('--trajectory-count is required when JSON records omit it')
            records.append(dict(version=row['model'].removeprefix('N'), trajectory_count=count,
                                epoch=row['epoch'], successes=row['successes'], episodes=row['episodes'],
                                success_rate=row['success_rate_percent']/100))
        plot_epoch_curves(records, out / 'plots', group_by='trajectory_count', versions=args.versions)
        raise SystemExit(0)
    if args.versions and not (args.model_epochs_only or args.trajectory_epochs_only):
        parser.error('--versions requires --model-epochs-only or --trajectory-epochs-only')
    if args.model_epochs_only or args.trajectory_epochs_only:
        plot_epoch_curves(read_json(out / 'summary.json')['checkpoints'], out / 'plots',
                          group_by='trajectory_count' if args.trajectory_epochs_only else 'version',
                          versions=list(dict.fromkeys(args.versions)) if args.versions else None)
    else:
        export(out)
