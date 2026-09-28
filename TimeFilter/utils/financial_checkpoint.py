"""Resolve and validate a financial best checkpoint before standalone evaluation."""
import json
import os
from pathlib import Path

import yaml

from data_provider.financial_registry import canonical_market
from utils.financial_provenance import collect_provenance


# These describe the evaluation invocation, not the trained model or data.
EVALUATION_ONLY = {
    'is_training', 'financial_checkpoint', 'financial_checkpoint_root',
    'financial_output_dir', 'financial_config', 'financial_force_rerun',
    'financial_validation_only', 'checkpoints', 'use_gpu', 'gpu',
    'financial_cpu', 'num_workers', 'gradient_diagnostic_epochs',
    'gradient_diagnostic_batch_size',
}


def resolve_financial_checkpoint(args, setting, project_root):
    explicit = getattr(args, 'financial_checkpoint', None)
    if explicit:
        checkpoint = Path(explicit).resolve()
    else:
        root = Path(getattr(args, 'financial_checkpoint_root', None) or args.checkpoints)
        checkpoint = (root / setting / 'best.pth').resolve()
    if checkpoint.name != 'best.pth' or not checkpoint.is_file():
        raise ValueError(f'Financial evaluation requires an existing best.pth: {checkpoint}. '
                         'Pass --checkpoint with the full path to the training run\'s best.pth.')
    if checkpoint.parent.name != setting or checkpoint.parent.parent.name != 'checkpoints':
        raise ValueError('Checkpoint does not match this model setting. '
                         'Use the training run\'s best.pth and matching configuration.')

    run_root = checkpoint.parents[2]
    config_path = run_root / 'config.yaml'
    if not config_path.is_file():
        raise ValueError(f'Cannot verify checkpoint configuration: {config_path} is missing')
    saved = yaml.safe_load(config_path.read_text(encoding='utf-8-sig'))
    if not isinstance(saved, dict) or saved.get('is_training') != 1:
        raise ValueError(f'Invalid training configuration beside checkpoint: {config_path}')
    saved.setdefault('moe_aux_weight', 0.05)
    saved.setdefault('rank_weight', 0.0)
    saved.setdefault('financial_norm', 1)
    current = vars(args)
    mismatches = []
    for key, saved_value in saved.items():
        if key in EVALUATION_ONLY:
            continue
        default = {'moe_aux_weight': 0.05, 'rank_weight': 0.0, 'financial_norm': 1}
        current_value = current.get(key, default.get(key))
        if key == 'data':
            saved_value = canonical_market(saved_value)
            current_value = canonical_market(current_value)
        elif key == 'root_path':
            saved_value = os.path.normcase(str(Path(saved_value).resolve()))
            current_value = os.path.normcase(str(Path(current_value).resolve()))
        if saved_value != current_value:
            mismatches.append(key)
    if mismatches:
        raise ValueError('Checkpoint configuration differs for: ' + ', '.join(mismatches) +
                         f'. Use the matching training config with {checkpoint}.')

    provenance_path = run_root / 'provenance.json'
    if provenance_path.is_file():
        saved_provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
        current_provenance = collect_provenance(project_root, args.root_path, args.data)
        if saved_provenance.get('data_sha256') != current_provenance['data_sha256']:
            raise ValueError('Checkpoint data version differs: data_sha256. '
                             f'Restore the original data for {checkpoint}.')
        if saved_provenance.get('code_sha256') != current_provenance['code_sha256']:
            print('Checkpoint code version differs (code_sha256); configuration and data match. '
                  'Evaluation will use the current code.')
    else:
        print('Legacy checkpoint: configuration verified; original code/data hashes are unavailable.')
    return checkpoint
