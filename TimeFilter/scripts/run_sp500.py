"""S&P500 compatibility entry point for the unified financial configuration."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_command(extra_args=()):
    """Retain the legacy command-builder API; the CLI below uses config.yaml."""
    # Model/optimizer hyperparameters deliberately inherit run.py's defaults.
    # Additional arguments are last, so explicit user overrides take priority.
    return [
        sys.executable, '-u', str(PROJECT_ROOT / 'run.py'),
        '--task_name', 'long_term_forecast',
        '--is_training', '1',
        '--model_id', 'SP500_returns_16_1',
        '--model', 'TimeFilter',
        '--data', 'SP500',
        '--root_path', str(PROJECT_ROOT / 'stockmixer_dataset' / 'SP500'),
        '--data_path', 'SP500.npy',
        '--features', 'M',
        '--freq', 'd',
        '--seq_len', '16',
        '--label_len', '0',
        '--pred_len', '1',
        '--enc_in', '474', '--dec_in', '474', '--c_out', '474',
        '--num_workers', '0',
        *extra_args,
    ]


if __name__ == '__main__':
    from run_financial import main
    raise SystemExit(main(['--dataset', 'SP500', *sys.argv[1:]]))
