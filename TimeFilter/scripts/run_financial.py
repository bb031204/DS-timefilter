"""Launch one of the three financial datasets using the shared YAML config."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from data_provider.financial_registry import MARKETS, canonical_market
from utils.finance_adaptation_config import normalize_finance_adaptation

SECTIONS = {
    'forecast': {'seq_len', 'label_len', 'pred_len', 'input_features'},
    'model': {'d_model', 'd_ff', 'n_heads', 'e_layers', 'patch_len', 'alpha', 'top_p', 'dropout', 'pos', 'norm'},
    'training': {'batch_size', 'train_epochs', 'learning_rate', 'patience', 'lradj', 'itr', 'financial_seed', 'financial_selection', 'test_each_epoch', 'stockmixer_selection_rank_weight', 'moe_aux_weight', 'rank_weight', 'gradient_diagnostic_epochs', 'gradient_diagnostic_batch_size', 'optimizer', 'weight_decay', 'grad_clip_norm'},
    'runtime': {'num_workers', 'gpu', 'cpu'},
}


def build_command(cli):
    import yaml
    config_path = Path(cli.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding='utf-8-sig'))
    allowed = {'dataset', 'data_root', 'mode', 'checkpoint', 'finance_adaptation', *SECTIONS}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError('Invalid or unknown top-level config.yaml keys')
    values = {}
    for section, keys in SECTIONS.items():
        entries = config.get(section, {})
        if not isinstance(entries, dict) or set(entries) - keys:
            raise ValueError(f'Unknown or invalid config section: {section}')
        if section == 'model' and 'norm' in entries:
            if not isinstance(entries['norm'], bool):
                raise ValueError('model.norm must be YAML true or false')
            values['financial_norm'] = int(entries['norm'])
            entries = {key: value for key, value in entries.items() if key != 'norm'}
        if section == 'forecast' and 'input_features' in entries:
            values['financial_input_features'] = entries['input_features']
            entries = {key: value for key, value in entries.items() if key != 'input_features'}
        if section == 'training':
            if 'test_each_epoch' in entries:
                if type(entries['test_each_epoch']) is not bool:
                    raise ValueError('training.test_each_epoch must be YAML true or false')
                values['financial_test_each_epoch'] = int(entries['test_each_epoch'])
            for key in ('optimizer', 'weight_decay', 'grad_clip_norm'):
                if key in entries:
                    values['financial_' + key] = entries[key]
            entries = {key: value for key, value in entries.items()
                       if key not in ('test_each_epoch', 'optimizer', 'weight_decay', 'grad_clip_norm')}
        values.update(entries)
    if 'finance_adaptation' in config:
        values['finance_adaptation'] = json.dumps(
            normalize_finance_adaptation(config['finance_adaptation']), separators=(',', ':'))
    for key in ('batch_size', 'train_epochs', 'learning_rate', 'moe_aux_weight'):
        if getattr(cli, key, None) is not None:
            values[key] = getattr(cli, key)
    market = canonical_market(cli.dataset or config.get('dataset', 'SP500'))
    if market not in MARKETS:
        raise ValueError('dataset must be SP500, NASDAQ or NYSE')
    if values.get('financial_input_features', 'returns') == 'eod5' and market != 'SP500':
        raise ValueError('Five-feature input is currently supported only for SP500')
    mode = cli.mode or config.get('mode', 'train')
    if mode not in ('train', 'evaluate'):
        raise ValueError('mode must be train or evaluate')
    checkpoint = cli.checkpoint or config.get('checkpoint')
    if mode == 'evaluate' and not checkpoint:
        raise ValueError('evaluate requires --checkpoint or checkpoint in config.yaml')
    seq_len = values.setdefault('seq_len', 16)
    values.setdefault('label_len', 0)
    values.setdefault('pred_len', 1)
    if values['pred_len'] != 1 or seq_len < 1 or seq_len % values.get('patch_len', 16):
        raise ValueError('Require pred_len=1 and positive seq_len divisible by patch_len')
    for key in ('batch_size', 'train_epochs', 'learning_rate', 'itr'):
        if key in values and values[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if values.get('financial_optimizer', 'adam') not in ('adam', 'adamw'):
        raise ValueError('training.optimizer must be adam or adamw')
    for key in ('financial_weight_decay', 'financial_grad_clip_norm'):
        if key in values and (type(values[key]) not in (int, float) or
                              not 0 <= values[key] < float('inf')):
            raise ValueError(f'training.{key.removeprefix("financial_")} must be finite and non-negative')
    if values.get('financial_optimizer', 'adam') == 'adam' and values.get('financial_weight_decay', 0):
        raise ValueError('training.weight_decay requires training.optimizer: adamw')
    epochs = values.get('gradient_diagnostic_epochs', [])
    if (not isinstance(epochs, list) or any(type(epoch) is not int or epoch < 0 for epoch in epochs)
            or epochs != sorted(set(epochs))):
        raise ValueError('gradient_diagnostic_epochs must be a sorted list of distinct non-negative integers')
    batch_size = values.get('gradient_diagnostic_batch_size', 8)
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('gradient_diagnostic_batch_size must be a positive integer')
    root = (PROJECT_ROOT / config.get('data_root', 'stockmixer_dataset') / market).resolve()
    command = [sys.executable, '-u', str(PROJECT_ROOT / 'run.py'),
               '--task_name', 'long_term_forecast', '--is_training', str(int(mode == 'train')),
               '--model', 'TimeFilter', '--model_id',
               f"{market}_{values.get('financial_input_features', 'returns')}_{seq_len}_1",
               '--data', market, '--root_path', str(root),
               '--data_path', 'SP500.npy' if market == 'SP500' else 'gt_data.pkl',
               '--features', 'M', '--freq', 'd', '--financial_config', str(config_path)]
    for name in ('enc_in', 'dec_in', 'c_out'):
        command.extend(['--' + name, str(MARKETS[market])])
    cpu = values.pop('cpu', False)
    if not isinstance(cpu, bool):
        raise ValueError('runtime.cpu must be YAML true or false')
    if cpu:
        command.append('--financial_cpu')
    for key, value in values.items():
        command.append('--' + key)
        if key == 'gradient_diagnostic_epochs':
            command.extend(map(str, value))
        else:
            command.append(str(value))
    if checkpoint:
        command.extend(['--financial_checkpoint', str((PROJECT_ROOT / checkpoint).resolve())])
    if getattr(cli, 'force_rerun', False):
        command.append('--financial_force_rerun')
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(PROJECT_ROOT / 'config.yaml'))
    parser.add_argument('--dataset', choices=[*MARKETS, 'S&P500'])
    parser.add_argument('--mode', choices=['train', 'evaluate'])
    parser.add_argument('--checkpoint')
    parser.add_argument('--force-rerun', action='store_true', help='Intentionally repeat completed training with identical settings')
    parser.add_argument('--batch-size', '--batch_size', dest='batch_size', type=int)
    parser.add_argument('--epochs', '--train_epochs', dest='train_epochs', type=int)
    parser.add_argument('--learning-rate', '--learning_rate', dest='learning_rate', type=float)
    parser.add_argument('--moe-aux-weight', '--moe_aux_weight', dest='moe_aux_weight', type=float,
                        help='MoE auxiliary loss weight; 0 disables the auxiliary objective')
    # Preserve the legacy run_sp500.py interface: remaining run.py options
    # override YAML values and are validated by run.py's own argument parser.
    args, extra = parser.parse_known_args(argv)
    return subprocess.call([*build_command(args), *extra], cwd=PROJECT_ROOT)


if __name__ == '__main__':
    raise SystemExit(main())
