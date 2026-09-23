"""Compare patches on validation, lock configuration, then repeat across seeds."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from data_provider.financial_registry import canonical_market
from utils.financial_runtime import allocate_run


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def run(command):
    directory = None
    import os
    with subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                          env=dict(os.environ, PYTHONIOENCODING='utf-8')) as child:
        try:
            for line in child.stdout:
                print(line, end='', flush=True)
                if line.startswith('Run directory: '):
                    directory = Path(line.strip().split('Run directory: ', 1)[1])
            code = child.wait()
        except KeyboardInterrupt:
            child.terminate()
            raise
    if code or directory is None:
        raise RuntimeError(f'Experiment failed (exit {code}); inspect its terminal.log. '
                           'Batch size is never reduced automatically; restart all comparisons with the same smaller batch size.')
    return directory


def choose(rows, metric):
    # Ties use the first patch in the predeclared order, never test metrics.
    return (min if metric == 'mse' else max)(rows, key=lambda row: row['validation_value'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(PROJECT_ROOT / 'config.yaml'))
    parser.add_argument('--dataset', default='SP500', choices=['SP500', 'S&P500', 'NASDAQ', 'NYSE'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--selection', choices=['mse', 'RankIC', 'IC'], default='mse')
    parser.add_argument('--seeds', type=int, nargs='+', default=[2021, 2022, 2023])
    parser.add_argument('--smoke', action='store_true', help='One epoch per run, workflow validation only')
    args = parser.parse_args()
    if len(set(args.seeds)) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Provide at least two distinct seeds')
    market = canonical_market(args.dataset)
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8-sig'))
    config.update(dataset=market, mode='train', checkpoint=None)
    config['forecast'] = dict(seq_len=16, label_len=0, pred_len=1)
    config['training'].update(batch_size=args.batch_size, itr=1, financial_selection=args.selection)
    if args.epochs is not None:
        config['training']['train_epochs'] = args.epochs
    if args.smoke:
        config['training']['train_epochs'] = 1
    if config['training']['train_epochs'] < 1 or args.batch_size < 1:
        parser.error('epochs and batch size must be positive')
    study = allocate_run(market + ('_patch_smoke' if args.smoke else '_patch_study'))
    plan = {'status': 'running', 'smoke_only': args.smoke, 'patches': [16, 8, 4],
            'selection_metric': args.selection, 'seeds': args.seeds, 'config': config,
            'started_at': datetime.now().isoformat()}
    write_json(study / 'study.json', plan)
    rows = []

    def train(patch, seed, stage):
        local = dict(config)
        local['model'] = dict(config['model'], patch_len=patch)
        local['training'] = dict(config['training'], financial_seed=seed)
        filename = study / f'{stage}_patch{patch}_seed{seed}.yaml'
        filename.write_text(yaml.safe_dump(local, sort_keys=False), encoding='utf-8')
        directory = run([sys.executable, '-u', str(PROJECT_ROOT / 'scripts/run_financial.py'),
                         '--config', str(filename), '--financial_validation_only'])
        selection_file = next(directory.rglob('selection.json'))
        selection = json.loads(selection_file.read_text())['best']
        row = {'stage': stage, 'patch_len': patch, 'seed': seed, 'run_directory': str(directory),
               'best_epoch': selection['epoch'], 'validation_value': selection['value'],
               'validation_metrics': selection['validation_metrics']}
        rows.append(row)
        write_json(study / 'validation_runs.json', rows)
        return row

    try:
        candidates = [train(patch, args.seeds[0], 'search') for patch in (16, 8, 4)]
        winner = choose(candidates, args.selection)
        # Persist the decision before any test evaluation or seed replication.
        write_json(study / 'locked_selection.json', winner)
        print(f"Locked patch_len={winner['patch_len']} using validation {args.selection} only")
        repetitions = [winner] + [train(winner['patch_len'], seed, 'repeat') for seed in args.seeds[1:]]
        test_rows = []
        for row in repetitions:
            source = Path(row['run_directory'])
            config_file = study / f"{row['stage']}_patch{row['patch_len']}_seed{row['seed']}.yaml"
            checkpoint = next(source.rglob('best.pth'))
            evaluation = run([sys.executable, '-u', str(PROJECT_ROOT / 'scripts/run_financial.py'),
                              '--config', str(config_file), '--mode', 'evaluate', '--checkpoint', str(checkpoint)])
            metrics = json.loads(next(evaluation.rglob('test_metrics.json')).read_text())
            test_rows.append(dict(seed=row['seed'], best_epoch=row['best_epoch'], metrics=metrics,
                                  evaluation_directory=str(evaluation)))
            write_json(study / 'test_runs.json', test_rows)
        summary = {}
        for key in test_rows[0]['metrics']:
            values = [r['metrics'].get(key) for r in test_rows]
            valid = [v for v in values if v is not None and np.isfinite(v)]
            summary[key] = {'mean': float(np.mean(valid)) if valid else None,
                            'std_ddof1': float(np.std(valid, ddof=1)) if len(valid) > 1 else None,
                            'valid_seeds': len(valid), 'total_seeds': len(values)}
        write_json(study / 'test_summary.json', summary)
        plan['status'] = 'completed'
    except BaseException:
        plan['status'] = 'failed_or_interrupted'
        raise
    finally:
        plan['finished_at'] = datetime.now().isoformat()
        write_json(study / 'study.json', plan)
        print(f'Study directory: {study}')


if __name__ == '__main__':
    main()
