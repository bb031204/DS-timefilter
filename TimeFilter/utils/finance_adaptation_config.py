"""Validated, reproducible switches for finance-only TimeFilter variants."""

import math


DEFAULT_FINANCE_ADAPTATION = {
    'enabled': False,
    'input_adapter': {
        'enabled': False,
        'input_layernorm': False,
        'indicator_mixer': {'enabled': False, 'hidden_dim': 16},
        'temporal_mixer': {'enabled': False, 'hidden_dim': 16},
    },
    'positional_encoding': {'mode': 'original'},
    'graph': {'strict_masked_softmax': False},
    'loss': {'ic_weight': 0.0},
}


def _merge(defaults, supplied, location):
    if not isinstance(supplied, dict):
        raise ValueError(f'{location} must be a mapping')
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f'{location} has unknown keys: {sorted(unknown)}')
    return {
        key: _merge(value, supplied.get(key, {}), f'{location}.{key}')
        if isinstance(value, dict) else supplied.get(key, value)
        for key, value in defaults.items()
    }


def normalize_finance_adaptation(supplied=None):
    """Fill defaults and reject ambiguous or malformed finance settings."""
    config = _merge(DEFAULT_FINANCE_ADAPTATION, {} if supplied is None else supplied, 'finance_adaptation')
    adapter = config['input_adapter']
    boolean_fields = (
        ('enabled', config['enabled']),
        ('input_adapter.enabled', adapter['enabled']),
        ('input_adapter.input_layernorm', adapter['input_layernorm']),
        ('input_adapter.indicator_mixer.enabled', adapter['indicator_mixer']['enabled']),
        ('input_adapter.temporal_mixer.enabled', adapter['temporal_mixer']['enabled']),
        ('graph.strict_masked_softmax', config['graph']['strict_masked_softmax']),
    )
    for name, value in boolean_fields:
        if type(value) is not bool:
            raise ValueError(f'finance_adaptation.{name} must be true or false')
    for name in ('indicator_mixer', 'temporal_mixer'):
        hidden = adapter[name]['hidden_dim']
        if type(hidden) is not int or hidden <= 0:
            raise ValueError(f'finance_adaptation.input_adapter.{name}.hidden_dim must be positive')
    if config['positional_encoding']['mode'] not in ('original', 'patch_only', 'none'):
        raise ValueError('finance_adaptation.positional_encoding.mode must be original, patch_only or none')
    weight = config['loss']['ic_weight']
    if type(weight) not in (float, int) or not math.isfinite(weight) or weight < 0:
        raise ValueError('finance_adaptation.loss.ic_weight must be finite and non-negative')
    config['loss']['ic_weight'] = float(weight)
    if config['enabled'] and not adapter['enabled'] and (
            adapter['input_layernorm'] or adapter['indicator_mixer']['enabled']
            or adapter['temporal_mixer']['enabled']):
        raise ValueError('Enable finance_adaptation.input_adapter before its submodules')
    return config
