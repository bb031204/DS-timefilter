"""Audit four validation-only checkpoint rules on historical SP500 folds."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_financial import build_command
from scripts.run_walkforward_selection import run_fold
from utils.financial_provenance import collect_provenance


def load_plan(path):
    plan = yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))
    required = {'base_config', 'output_root', 'seeds', 'future_each_epoch',
                'stability_lambda', 'selection_rules', 'decision_metrics', 'folds'}
    if not isinstance(plan, dict) or set(plan) != required:
        raise ValueError('Invalid model-selection plan fields')
    if plan['selection_rules'] != ['A', 'B', 'C', 'D']:
        raise ValueError('First-stage rules must be frozen as A/B/C/D')
    if plan['decision_metrics'] != {
            'primary': ['RankIC', 'mean_top5_excess_return'],
            'guardrails': ['IC', 'prec_10', 'sharpe5']}:
        raise ValueError('First-stage decision metrics must be frozen before Future evaluation')
    if type(plan['future_each_epoch']) is not bool:
        raise ValueError('future_each_epoch must be true or false')
    stability_lambda = plan['stability_lambda']
    if (type(stability_lambda) not in (int, float) or not np.isfinite(stability_lambda)
            or stability_lambda < 0):
        raise ValueError('stability_lambda must be finite and non-negative')
    seeds = plan['seeds']
    if (not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('seeds must be distinct non-negative integers')
    folds = plan['folds']
    if not isinstance(folds, list) or not folds:
        raise ValueError('folds must be a nonempty list')
    names, ranges = set(), []
    for fold in folds:
        if not isinstance(fold, dict) or set(fold) != {'name', 'train_end', 'valid_end', 'future_end'}:
            raise ValueError('Each fold needs name, train_end, valid_end and future_end')
        name = fold['name']
        if not isinstance(name, str) or not re.fullmatch(r'fold_[1-9][0-9]*', name) or name in names:
            raise ValueError('Fold names must be distinct fold_N identifiers')
        names.add(name)
        train, valid, future = (fold[key] for key in ('train_end', 'valid_end', 'future_end'))
        if (any(type(value) is not int for value in (train, valid, future))
                or not 16 < train < valid < future <= 1259
                or valid - train != 253 or future - valid != 126):
            raise ValueError('Each fold needs 253 validation days and 126 future days before day 1259')
        ranges.append((valid, future))
    if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
        raise ValueError('Future blocks must be ordered and non-overlapping')
    base = (PROJECT_ROOT / plan['base_config']).resolve()
    output_root = (PROJECT_ROOT / plan['output_root']).resolve()
    if not base.is_file() or not base.is_relative_to(PROJECT_ROOT):
        raise ValueError('base_config must be an existing project file')
    if not output_root.is_relative_to(PROJECT_ROOT / 'outputs'):
        raise ValueError('output_root must stay inside TimeFilter/outputs')
    config = yaml.safe_load(base.read_text(encoding='utf-8-sig'))
    if (config.get('dataset') != 'SP500' or config.get('mode') != 'train'
            or config.get('forecast', {}).get('seq_len') != 16
            or config.get('forecast', {}).get('pred_len') != 1
            or config.get('forecast', {}).get('input_features') != 'eod5'
            or config.get('model', {}).get('norm') is not False
            or config.get('training', {}).get('financial_selection') != 'stockmixer_val_loss'
            or config.get('training', {}).get('itr') != 1):
        raise ValueError('base_config must use the fixed SP500 eod5/16/1 selection protocol')
    return plan, base, output_root


def make_command(base, fold, seed, output_dir, future_each_epoch, stability_lambda):
    launcher_args = argparse.Namespace(config=str(base), dataset='SP500', mode='train',
                                       checkpoint=None, force_rerun=False,
                                       batch_size=None, train_epochs=None,
                                       learning_rate=None, moe_aux_weight=None)
    command = build_command(launcher_args)
    split = {key: fold[key] for key in ('train_end', 'valid_end', 'future_end')}
    for flag, value in (('--financial_seed', seed),
                        ('--financial_test_each_epoch', int(future_each_epoch))):
        command[command.index(flag) + 1] = str(value)
    command.extend(['--financial_model_selection', '--financial_split', json.dumps(split),
                    '--financial_stability_lambda', str(stability_lambda),
                    '--financial_output_dir', str(output_dir),
                    '--checkpoints', str(output_dir / 'checkpoints')])
    return command


def _completed_fold(directory, command):
    status = directory / 'run_status.json'
    saved_command = directory / 'command.json'
    selection = list((directory / 'results').glob('*/financial/selection_summary.json'))
    return (status.is_file() and saved_command.is_file() and len(selection) == 1
            and json.loads(status.read_text(encoding='utf-8')).get('status') == 'completed'
            and json.loads(saved_command.read_text(encoding='utf-8')) == command)


def collect_summary(root, plan):
    rows = []
    for fold in plan['folds']:
        for seed in plan['seeds']:
            directory = root / fold['name'] / f'seed_{seed}'
            reports = list((directory / 'results').glob('*/financial'))
            if len(reports) != 1:
                raise ValueError(f'Expected one financial report in {directory}')
            report = reports[0]
            selection = json.loads((report / 'selection_summary.json').read_text(encoding='utf-8'))
            for rule in plan['selection_rules']:
                chosen = selection['rules'][rule]
                future = json.loads((report / f'future_metrics_rule_{rule}.json').read_text(encoding='utf-8'))
                row = {'fold': fold['name'], 'seed': seed, 'rule': rule,
                       'train_end': fold['train_end'], 'valid_end': fold['valid_end'],
                       'future_end': fold['future_end'],
                       'selected_epoch': chosen['selected_epoch'],
                       'validation_metric': chosen['metric'],
                       'validation_score': chosen['validation_score']}
                for key in ('mse', 'mae', 'IC', 'RIC', 'RankIC', 'RankIC_std',
                            'RankICIR', 'prec_10', 'sharpe5', 'directional_accuracy',
                            'mean_top5_return', 'mean_universe_return',
                            'mean_top5_excess_return'):
                    row[f'future_{key}'] = future[key]
                rows.append(row)
    with (root / 'walkforward_model_selection_summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_rule = {}
    for rule in plan['selection_rules']:
        selected = [row for row in rows if row['rule'] == rule]
        # Seeds repeat the same historical blocks: aggregate within fold first.
        fold_values = {}
        for fold in plan['folds']:
            fold_rows = [row for row in selected if row['fold'] == fold['name']]
            fold_values[fold['name']] = {
                key: float(np.mean([row[f'future_{key}'] for row in fold_rows]))
                for key in ('IC', 'RankIC', 'prec_10', 'sharpe5',
                            'mean_top5_return', 'mean_top5_excess_return')}
        rankic = np.asarray([value['RankIC'] for value in fold_values.values()])
        ic = np.asarray([value['IC'] for value in fold_values.values()])
        excess = np.asarray([value['mean_top5_excess_return'] for value in fold_values.values()])
        by_rule[rule] = {
            'fold_macro_means': fold_values,
            'mean_future_RankIC': float(rankic.mean()),
            'worst_fold_future_RankIC': float(rankic.min()),
            'std_fold_future_RankIC': float(rankic.std()),
            'positive_fold_RankIC_ratio': float(np.mean(rankic > 0)),
            'mean_future_IC': float(ic.mean()),
            'worst_fold_future_IC': float(ic.min()),
            'std_fold_future_IC': float(ic.std()),
            'mean_future_top5_excess_return': float(excess.mean()),
            'worst_fold_future_top5_excess_return': float(excess.min()),
            'positive_fold_top5_excess_return_ratio': float(np.mean(excess > 0)),
            **{f'mean_future_{key}': float(np.mean([value[key] for value in fold_values.values()]))
               for key in ('prec_10', 'sharpe5', 'mean_top5_return')},
        }
    summary = {'folds': len(plan['folds']), 'seeds': plan['seeds'],
               'decision_metrics': plan['decision_metrics'],
               'interpretation': 'Historical future blocks are meta-validation for rule comparison, not untouched test data. Formal test has already been inspected in prior development.',
               'decision': 'descriptive only; no automatic winner from three folds',
               'rows': rows, 'by_rule': by_rule}
    (root / 'walkforward_model_selection_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=str(PROJECT_ROOT / 'walkforward_model_selection.yaml'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--retry-incomplete', action='store_true')
    args = parser.parse_args(argv)
    if args.retry_incomplete and not args.resume:
        parser.error('--retry-incomplete requires --resume')
    plan, base, output_root = load_plan(args.plan)
    root = args.resume.resolve() if args.resume else None
    if root is not None and not root.is_relative_to(output_root):
        parser.error('--resume must point inside output_root')
    if root is None:
        root = output_root / datetime.now().strftime('%Y%m%d_%H%M%S')
        if root.exists():
            parser.error(f'Audit directory already exists: {root}')
    if args.dry_run:
        for fold in plan['folds']:
            for seed in plan['seeds']:
                directory = root / fold['name'] / f'seed_{seed}'
                print(json.dumps(make_command(base, fold, seed, directory,
                                              plan['future_each_epoch'],
                                              plan['stability_lambda']), ensure_ascii=False))
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
    print(f'Model-selection audit: {root}', flush=True)
    for fold in plan['folds']:
        for seed in plan['seeds']:
            directory = root / fold['name'] / f'seed_{seed}'
            if collect_provenance(PROJECT_ROOT, data_root, 'SP500') != provenance:
                parser.error('Code or data changed during this audit')
            command = make_command(frozen_base, fold, seed, directory,
                                   plan['future_each_epoch'], plan['stability_lambda'])
            if _completed_fold(directory, command):
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
    print(f'Completed {len(summary["rows"])} rule results: {root}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
