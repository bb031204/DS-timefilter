"""The only datasets that enable the financial experiment extensions."""
from pathlib import Path

MARKETS = {'SP500': 474, 'NASDAQ': 1026, 'NYSE': 1737}


def canonical_market(name):
    return 'SP500' if name == 'S&P500' else name


def is_financial_dataset(name):
    return canonical_market(name) in MARKETS


def required_files(market):
    return ('SP500.npy',) if canonical_market(market) == 'SP500' else (
        'price_data.pkl', 'gt_data.pkl', 'mask_data.pkl')


def validate_files(root, market):
    names = required_files(market)
    bad = [str(Path(root) / name) for name in names
           if not (Path(root) / name).is_file() or (Path(root) / name).stat().st_size == 0]
    if bad:
        raise ValueError('Missing or empty StockMixer data files; restore the dataset first: ' + ', '.join(bad))
