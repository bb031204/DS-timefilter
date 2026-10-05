"""Read-only gradient measurements for financial training objectives."""

import csv
import math
import random

import numpy as np
import torch


GROUPS = ('all', 'patch_embed', 'backbone', 'moe_gate', 'head')
FIELDS = ('epoch', 'stage', 'batch_size', 'train_indices', 'component', 'group',
          'raw_loss', 'weight', 'weighted_loss', 'grad_norm', 'ratio_to_mse',
          'active_tensors', 'total_tensors')


def _parameter_group(name):
    name = name.removeprefix('module.')
    if name.startswith('finance_input_adapter.'):
        return 'finance_adapter'
    if name.startswith('patch_embed.'):
        return 'patch_embed'
    if name.startswith('head.'):
        return 'head'
    if '.mask_moe.gate.' in name or '.mask_moe.noise.' in name:
        return 'moe_gate'
    return 'backbone'


def _gradient_summary(parameters, gradients, groups):
    squared = dict.fromkeys(groups, 0.0)
    active = dict.fromkeys(groups, 0)
    total = dict.fromkeys(groups, 0)
    for (name, _), gradient in zip(parameters, gradients):
        group = _parameter_group(name)
        total['all'] += 1
        total[group] += 1
        if gradient is None:
            continue
        length_squared = gradient.detach().float().square().sum().item()
        squared['all'] += length_squared
        squared[group] += length_squared
        active['all'] += 1
        active[group] += 1
    return {group: (math.sqrt(squared[group]), active[group], total[group]) for group in groups}


def record_gradient_diagnostic(model, batch, train_indices, loss_fn, report_path, epoch):
    """Measure weighted losses on one fixed batch without touching optimizer grads or RNG."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    training_modes = [(module, module.training) for module in model.modules()]
    normalizer = getattr(model.module if hasattr(model, 'module') else model, 'norm', None)
    normalizer_state = ({name: getattr(normalizer, name) for name in ('mean', 'stdev', 'last')
                         if hasattr(normalizer, name)} if normalizer is not None else {})
    parameters = [(name, parameter) for name, parameter in model.named_parameters()
                  if parameter.requires_grad]
    groups = (GROUPS + ('finance_adapter',) if any(
        _parameter_group(name) == 'finance_adapter' for name, _ in parameters) else GROUPS)
    try:
        model.train()
        with torch.enable_grad():
            # loss_fn performs exactly one training-mode forward for all three terms.
            raw_losses = loss_fn(batch)
            weighted = {name: raw * weight for name, (raw, weight) in raw_losses.items()}
            differentiable = [name for name, term in weighted.items()
                              if torch.is_tensor(term) and term.requires_grad]
            summaries = {}
            for position, name in enumerate(differentiable):
                gradients = torch.autograd.grad(
                    weighted[name], [parameter for _, parameter in parameters],
                    retain_graph=position + 1 < len(differentiable), allow_unused=True)
                summaries[name] = _gradient_summary(parameters, gradients, groups)
            for name in raw_losses:
                if name not in summaries:
                    summaries[name] = _gradient_summary(parameters, (None,) * len(parameters), groups)
            values = {name: (float(raw.detach()) if torch.is_tensor(raw) else float(raw), weight)
                      for name, (raw, weight) in raw_losses.items()}
    finally:
        if normalizer is not None:
            for name in ('mean', 'stdev', 'last'):
                if name in normalizer_state:
                    setattr(normalizer, name, normalizer_state[name])
                elif hasattr(normalizer, name):
                    delattr(normalizer, name)
        for module, mode in training_modes:
            module.training = mode
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    stage = 'pretrain' if epoch == 0 else 'post_epoch'
    index_text = '|'.join(map(str, train_indices))
    rows = []
    print(f'[Gradient Diagnostic] stage={stage} epoch={epoch} '
          f'batch_size={len(train_indices)} train_indices={index_text}')
    for name, (raw, weight) in values.items():
        print(f'  {name}: raw={raw:.8g} weight={weight:.6g} weighted={raw * weight:.8g}')
        for group in groups:
            norm, active, total = summaries[name][group]
            mse_norm = summaries['MSE'][group][0]
            ratio = norm / mse_norm if mse_norm > 1e-12 else None
            rows.append({'epoch': epoch, 'stage': stage, 'batch_size': len(train_indices),
                         'train_indices': index_text, 'component': name, 'group': group,
                         'raw_loss': raw, 'weight': weight, 'weighted_loss': raw * weight,
                         'grad_norm': norm, 'ratio_to_mse': ratio,
                         'active_tensors': active, 'total_tensors': total})
            ratio_text = f'{ratio:.4g}' if ratio is not None else 'n/a'
            print(f'    {group}: grad_norm={norm:.6g} ratio_to_mse={ratio_text} '
                  f'active_tensors={active}/{total}')

    path = report_path / 'gradient_diagnostics.csv'
    new_file = not path.exists()
    with path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
    print(f'  Saved: {path.resolve()}')
