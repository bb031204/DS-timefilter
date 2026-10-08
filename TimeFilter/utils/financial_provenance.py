"""Content hashes for the code and dataset actually used by a financial run."""
import hashlib
import json
from pathlib import Path

from data_provider.financial_registry import required_files, validate_files


CODE_FOLDERS = ('data_provider', 'exp', 'layers', 'models', 'scripts', 'utils')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _collection_hash(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode('utf-8')).hexdigest()


def collect_provenance(project_root, data_root, market, input_features='returns'):
    project_root = Path(project_root)
    data_root = Path(data_root)
    validate_files(data_root, market, input_features)
    code_paths = [project_root / 'run.py']
    for folder in CODE_FOLDERS:
        code_paths.extend(sorted((project_root / folder).rglob('*.py')))
    code_files = {p.relative_to(project_root).as_posix(): file_hash(p) for p in code_paths}
    data_files = {name: file_hash(data_root / name)
                  for name in required_files(market, input_features)}
    return {'code_sha256': _collection_hash(code_files),
            'data_sha256': _collection_hash(data_files),
            'code_files': code_files, 'data_files': data_files}
