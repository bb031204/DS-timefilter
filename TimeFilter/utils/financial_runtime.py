"""Isolate each financial invocation and tee child output, including failures."""
from datetime import datetime
from contextlib import contextmanager
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def allocate_run(market, root=None, now=None):
    now = now or datetime.now()
    parent = Path(root or PROJECT_ROOT / 'outputs')
    parent.mkdir(parents=True, exist_ok=True)
    run_name = market + now.strftime('_%Y_%m_%d_%H_%M')
    for index in range(10000):
        path = parent / (run_name if index == 0 else f'{run_name}_{index + 1:02d}')
        try:
            path.mkdir()
            return path.resolve()
        except FileExistsError:
            continue
    raise RuntimeError('Too many runs in the same minute')


def experiment_key(values, provenance):
    """Compare effective settings and code/data bytes, not output paths."""
    from data_provider.financial_registry import canonical_market
    settings = dict(values)
    for key in ('financial_output_dir', 'financial_checkpoint_root', 'financial_config',
                'checkpoints', 'financial_force_rerun'):
        settings.pop(key, None)
    settings['data'] = canonical_market(settings['data'])
    for key in ('root_path', 'financial_checkpoint'):
        if settings.get(key):
            settings[key] = os.path.normcase(str(Path(settings[key]).resolve()))
    # Configs saved before these options existed retain their original defaults.
    for key, default in (('moe_aux_weight', 0.05), ('rank_weight', 0.0), ('financial_norm', 1),
                         ('financial_seed', 2021),
                         ('financial_selection', 'mse'), ('financial_validation_only', False),
                         ('financial_input_features', 'returns'),
                         ('stockmixer_selection_rank_weight', 0.1)):
        settings.setdefault(key, default)
    identity = {'settings': settings,
                'code_sha256': provenance['code_sha256'],
                'data_sha256': provenance['data_sha256']}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode('utf-8')).hexdigest()


class DuplicateFinancialRunError(RuntimeError):
    pass


@contextmanager
def experiment_lock(root, key):
    """OS lock releases on process exit, including crashes; keep the lock file."""
    folder = root / '.locks'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / (key + '.lock')).open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise DuplicateFinancialRunError('Identical financial experiment is already running; duplicate launch blocked.') from error
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def launch_financial(args, argv):
    import yaml
    from utils.financial_provenance import collect_provenance
    provenance = collect_provenance(PROJECT_ROOT, args.root_path, args.data)
    # Evaluation can intentionally use an overwritten checkpoint at the same path.
    if not args.is_training:
        return _launch_financial(args, argv, provenance)
    root = PROJECT_ROOT / 'outputs'
    key = experiment_key(vars(args), provenance)
    try:
        with experiment_lock(root, key):
            if not getattr(args, 'financial_force_rerun', False):
                for directory in sorted(root.glob('*'), reverse=True):
                    status_file = directory / 'run_status.json'
                    config_file = directory / 'config.yaml'
                    provenance_file = directory / 'provenance.json'
                    if not status_file.is_file() or not config_file.is_file() or not provenance_file.is_file():
                        continue
                    try:
                        status = json.loads(status_file.read_text(encoding='utf-8'))
                        config = yaml.safe_load(config_file.read_text(encoding='utf-8-sig'))
                        previous = json.loads(provenance_file.read_text(encoding='utf-8'))
                        matches = status.get('status') == 'completed' and experiment_key(config, previous) == key
                    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
                        continue
                    if matches:
                        print(f'Run directory: {directory.resolve()}', flush=True)
                        print('Identical training already completed; reused existing results. No new training or output directory. '
                              'Use scripts/run_financial.py --force-rerun to intentionally repeat.', flush=True)
                        return 0
            return _launch_financial(args, argv, provenance)
    except DuplicateFinancialRunError as error:
        print(str(error), flush=True)
        return 2


def _launch_financial(args, argv, provenance):
    import yaml
    from data_provider.financial_registry import canonical_market
    market = canonical_market(args.data)
    directory = allocate_run(market)
    root_path = str(Path(args.root_path).resolve())
    checkpoint_root = str(Path(args.checkpoints).resolve())
    checkpoint = getattr(args, 'financial_checkpoint', None)
    command = [sys.executable, '-u', str(PROJECT_ROOT / 'run.py'), *argv,
               '--data', market, '--root_path', root_path,
               '--financial_output_dir', str(directory),
               '--checkpoints', './checkpoints/',
               '--financial_checkpoint_root', checkpoint_root]
    if checkpoint:
        command += ['--financial_checkpoint', str(Path(checkpoint).resolve())]
    snapshot = dict(vars(args), root_path=root_path, financial_output_dir=str(directory),
                    checkpoints='./checkpoints/', financial_checkpoint_root=checkpoint_root)
    if checkpoint:
        snapshot['financial_checkpoint'] = str(Path(checkpoint).resolve())
    (directory / 'config.yaml').write_text(yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False), encoding='utf-8')
    (directory / 'provenance.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    source = getattr(args, 'financial_config', None)
    if source:
        (directory / 'source_config.yaml').write_bytes(Path(source).read_bytes())
    (directory / 'command.json').write_text(json.dumps(command, ensure_ascii=False, indent=2), encoding='utf-8')
    status = {'status': 'running', 'started_at': datetime.now().isoformat(), 'dataset': market,
              'launcher_pid': os.getpid(), 'launcher_parent_pid': os.getppid(),
              'launcher_argv': sys.argv}
    status_path = directory / 'run_status.json'
    def save_status():
        status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
    save_status()
    print(f'Run directory: {directory}', flush=True)
    child = None
    code = 1
    with (directory / 'terminal.log').open('w', encoding='utf-8', buffering=1) as log:
        log.write(f'Run directory: {directory}\n')
        from utils.financial_progress import FinancialProgress, PROGRESS_PREFIX
        progress = FinancialProgress(log)
        try:
            environment = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
            child = subprocess.Popen(command, cwd=directory, env=environment,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding='utf-8', errors='replace')
            status['training_pid'] = child.pid
            save_status()
            for line in child.stdout:
                if line.startswith(PROGRESS_PREFIX):
                    progress.update(line)
                    continue
                progress.clear_line()
                log.write(re.sub(r'\x1b\[[0-9;]*m', '', line))
                print(line, end='', flush=True)
            code = child.wait()
            status['status'] = 'completed' if code == 0 else 'failed'
        except KeyboardInterrupt:
            if child is not None:
                child.terminate()
                child.wait()
            code = 130
            status['status'] = 'interrupted'
            log.write('Interrupted by user\n')
        except Exception as error:
            status['status'] = 'failed'
            log.write(f'{type(error).__name__}: {error}\n')
            raise
        finally:
            progress.clear_line()
            status.update(return_code=code, finished_at=datetime.now().isoformat())
            save_status()
    return code
