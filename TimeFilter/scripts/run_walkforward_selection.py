"""Run the frozen SP500 checkpoint-selection audit on historical rolling folds."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_financial import build_command
from utils.financial_progress import FinancialProgress, PROGRESS_PREFIX
from utils.financial_provenance import collect_provenance


def load_plan(path):
    plan = yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(plan, dict) or set(plan) != {
            'base_config', 'output_root', 'seeds', 'future_each_epoch', 'folds'}:
        raise ValueError('Invalid walk-forward plan fields')
    if type(plan['future_each_epoch']) is not bool:
        raise ValueError('future_each_epoch must be true or false')
    seeds = plan['seeds']
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('seeds must be a nonempty list of non-negative integers')
    if len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be distinct')
    folds = plan['folds']
    if not isinstance(folds, list) or not folds:
        raise ValueError('folds must be a nonempty list')
    names = set()
    future_ranges = []
    for fold in folds:
        if not isinstance(fold, dict) or set(fold) != {
                'name', 'train_end', 'valid_end', 'future_end'}:
            raise ValueError('Each fold needs name, train_end, valid_end and future_end')
        name = fold['name']
        if not isinstance(name, str) or not re.fullmatch(r'fold_[1-9][0-9]*', name) or name in names:
            raise ValueError('Fold names must be unique fold_N identifiers')
        names.add(name)
        train, valid, future = (fold[key] for key in ('train_end', 'valid_end', 'future_end'))
        if (any(type(v) is not int for v in (train, valid, future))
                or not 16 < train < valid < future <= 1259):
            raise ValueError('Require 16 < train_end < valid_end < future_end <= 1259')
        future_ranges.append((valid, future))
    if any(left[1] > right[0] for left, right in zip(future_ranges, future_ranges[1:])):
        raise ValueError('Future blocks must be ordered and non-overlapping')
    base = (PROJECT_ROOT / plan['base_config']).resolve()
    output_root = (PROJECT_ROOT / plan['output_root']).resolve()
    if not base.is_file() or not base.is_relative_to(PROJECT_ROOT):
        raise ValueError('base_config must be an existing project file')
    if not output_root.is_relative_to(PROJECT_ROOT / 'outputs'):
        raise ValueError('output_root must stay within TimeFilter/outputs')
    config = yaml.safe_load(base.read_text(encoding='utf-8-sig'))
    if (config.get('dataset') != 'SP500' or config.get('mode') != 'train'
            or config.get('forecast', {}).get('seq_len') != 16
            or config.get('forecast', {}).get('input_features') != 'eod5'
            or config.get('training', {}).get('rank_weight') != 5
            or config.get('finance_adaptation', {}).get('loss', {}).get('ic_weight') != 0
            or config.get('training', {}).get('financial_selection') != 'stockmixer_val_loss'
            or config.get('training', {}).get('itr') != 1):
        raise ValueError('base_config must be the frozen 15_49 SP500 training protocol')
    return plan, base, output_root


def make_command(base, fold, seed, output_dir, future_each_epoch):
    launcher_args = argparse.Namespace(config=str(base), dataset='SP500', mode='train',
                                       checkpoint=None, force_rerun=False,
                                       batch_size=None, train_epochs=None,
                                       learning_rate=None, moe_aux_weight=None)
    command = build_command(launcher_args)
    split = {key: fold[key] for key in ('train_end', 'valid_end', 'future_end')}
    command.extend(['--financial_walkforward', '--financial_split', json.dumps(split),
                    '--financial_seed', str(seed),
                    '--financial_test_each_epoch', str(int(future_each_epoch)),
                    '--financial_output_dir', str(output_dir),
                    '--checkpoints', str(output_dir / 'checkpoints')])
    return command


def run_fold(command, directory):
    status_path = directory / 'run_status.json'
    status = {'status': 'running', 'started_at': datetime.now().isoformat()}
    status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
    with (directory / 'terminal.log').open('w', encoding='utf-8', buffering=1) as log:
        progress = FinancialProgress(log)
        environment = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
        child = subprocess.Popen(command, cwd=directory, env=environment, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 encoding='utf-8', errors='replace')
        status['training_pid'] = child.pid
        status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
        try:
            for line in child.stdout:
                if line.startswith(PROGRESS_PREFIX):
                    progress.update(line)
                    continue
                progress.clear_line()
                log.write(re.sub(r'\x1b\[[0-9;]*m', '', line))
                print(line, end='', flush=True)
            code = child.wait()
        except KeyboardInterrupt:
            child.terminate()
            child.wait()
            code = 130
        finally:
            progress.clear_line()
        status.update(status='completed' if code == 0 else 'failed', return_code=code,
                      finished_at=datetime.now().isoformat())
        status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
    return code


def completed_fold(directory, command):
    status = directory / 'run_status.json'
    saved_command = directory / 'command.json'
    if not status.is_file() or not saved_command.is_file():
        return False
    return (json.loads(status.read_text(encoding='utf-8')).get('status') == 'completed'
            and json.loads(saved_command.read_text(encoding='utf-8')) == command)


def collect_summary(root, plan):
    rows = []
    for fold in plan['folds']:
        for seed in plan['seeds']:
            directory = root / fold['name'] / f'seed_{seed}'
            report_paths = list((directory / 'results').glob('*/financial'))
            if len(report_paths) != 1:
                raise ValueError(f'Expected one financial report in {directory}')
            report = report_paths[0]
            selection = json.loads((report / 'selection_summary.json').read_text(encoding='utf-8'))
            a = json.loads((report / 'future_metrics_A.json').read_text(encoding='utf-8'))
            b = json.loads((report / 'future_metrics_B.json').read_text(encoding='utf-8'))
            row = {'fold': fold['name'], 'seed': seed,
                   'train_end': fold['train_end'], 'valid_end': fold['valid_end'],
                   'future_end': fold['future_end'],
                   'A_epoch': selection['A']['epoch'], 'B_epoch': selection['B']['epoch']}
            for key in ('mse', 'IC', 'RIC', 'RankIC', 'RankICIR', 'prec_10',
                        'sharpe5', 'directional_accuracy'):
                row[f'A_{key}'] = a[key]
                row[f'B_{key}'] = b[key]
                row[f'delta_B_minus_A_{key}'] = b[key] - a[key]
            rows.append(row)
    with (root / 'walkforward_summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'folds': len(plan['folds']), 'seeds': plan['seeds'], 'rows': rows,
               'interpretation': 'paired historical future blocks; original test is not used for selection'}
    for key in ('mse', 'IC', 'RankIC', 'prec_10', 'sharpe5'):
        delta = np.asarray([row[f'delta_B_minus_A_{key}'] for row in rows], dtype=float)
        summary[f'paired_delta_B_minus_A_{key}'] = {
            'mean': float(delta.mean()),
            'std': float(delta.std(ddof=1)) if len(delta) > 1 else None,
            'B_wins': int(np.sum(delta < 0 if key == 'mse' else delta > 0)),
            'comparisons': len(delta),
        }
    (root / 'walkforward_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=str(PROJECT_ROOT / 'walkforward_selection.yaml'))
    parser.add_argument('--dry-run', action='store_true', help='Validate and print commands without training')
    parser.add_argument('--resume', type=Path, help='Continue an existing audit directory')
    parser.add_argument('--retry-incomplete', action='store_true',
                        help='With --resume, archive an incomplete fold and rerun it')
    args = parser.parse_args(argv)
    if args.retry_incomplete and not args.resume:
        parser.error('--retry-incomplete requires --resume')
    plan, base, output_root = load_plan(args.plan)
    root = args.resume.resolve() if args.resume else None
    if root is not None and not root.is_relative_to(output_root):
        parser.error('--resume must point inside the configured output_root')
    if root is None:
        root = output_root / datetime.now().strftime('%Y%m%d_%H%M%S')
        if root.exists():
            parser.error(f'Audit directory already exists: {root}')
    if args.dry_run:
        for fold in plan['folds']:
            for seed in plan['seeds']:
                directory = root / fold['name'] / f'seed_{seed}'
                print(json.dumps(make_command(base, fold, seed, directory,
                                              plan['future_each_epoch']), ensure_ascii=False))
        return 0
    if args.resume:
        if not root.is_dir() or (root / 'plan.yaml').read_bytes() != Path(args.plan).read_bytes():
            parser.error('Resume directory is missing or uses a different plan')
        if (root / 'base_config.sha256').read_text(encoding='ascii') != hashlib.sha256(base.read_bytes()).hexdigest():
            parser.error('Frozen base_config changed since this audit started')
    else:
        root.mkdir(parents=True)
        shutil.copy2(args.plan, root / 'plan.yaml')
        shutil.copy2(base, root / 'base_config.yaml')
        (root / 'base_config.sha256').write_text(hashlib.sha256(base.read_bytes()).hexdigest(), encoding='ascii')
    frozen_base = root / 'base_config.yaml'
    data_root = PROJECT_ROOT / 'stockmixer_dataset' / 'SP500'
    provenance = collect_provenance(PROJECT_ROOT, data_root, 'SP500')
    provenance_file = root / 'provenance.json'
    if args.resume:
        if json.loads(provenance_file.read_text(encoding='utf-8')) != provenance:
            parser.error('Code or data changed since this audit started')
    else:
        provenance_file.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(f'Walk-forward audit: {root}', flush=True)
    for fold in plan['folds']:
        for seed in plan['seeds']:
            directory = root / fold['name'] / f'seed_{seed}'
            if collect_provenance(PROJECT_ROOT, data_root, 'SP500') != provenance:
                parser.error('Code or data changed during this audit')
            command = make_command(frozen_base, fold, seed, directory, plan['future_each_epoch'])
            if completed_fold(directory, command):
                print(f'Already completed: {fold["name"]}, seed {seed}', flush=True)
                continue
            if directory.exists():
                if not args.retry_incomplete:
                    parser.error(f'Incomplete fold at {directory}; use --resume with --retry-incomplete')
                archive = directory.with_name(directory.name + '_incomplete_'
                                              + datetime.now().strftime('%Y%m%d_%H%M%S'))
                if (archive.exists() or not directory.resolve().is_relative_to(root.resolve())
                        or not archive.resolve().is_relative_to(root.resolve())):
                    parser.error('Cannot safely archive the incomplete fold')
                directory.rename(archive)
            directory.mkdir(parents=True)
            shutil.copy2(frozen_base, directory / 'source_config.yaml')
            split = {key: fold[key] for key in ('train_end', 'valid_end', 'future_end')}
            (directory / 'split.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
            (directory / 'command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
            (directory / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
            code = run_fold(command, directory)
            if code:
                print(f'Fold failed: {fold["name"]}, seed {seed}, exit {code}', flush=True)
                return code
    summary = collect_summary(root, plan)
    print(f'Completed {len(summary["rows"])} paired comparisons: {root}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
