"""Warm the original experiment's pinned Hugging Face caches, without using a GPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--recipe', type=Path, required=True)
    p.add_argument('--metadata', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.runtime.resolve()))
    from experiment_cli.cli import resolve
    from gsm8k_experiment.assets import resolve_assets
    from gsm8k_experiment.common import atomic_json, read_json
    from gsm8k_experiment.data import partition_rows
    from datasets import load_dataset
    plan = resolve(str(a.recipe.resolve()), stage='prepare')
    config = plan['settings']
    if config['arms'] != ['proxy', 'judge', 'knn_static']:
        raise ValueError('The recipe must contain proxy, judge and knn_static.')
    a.metadata.mkdir(parents=True, exist_ok=True)
    config_path = a.metadata / 'preparation_config.json'
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError('Preparation configuration changed; use a new experiment ID.')
    atomic_json(config_path, config)
    resolved = resolve_assets(config, a.metadata)
    # Call even if split metadata exists: this verifies/warms the actual dataset cache.
    ds = load_dataset(config['dataset']['id'], config['dataset']['config'], revision=resolved['dataset'])
    split = partition_rows(ds['train'], ds['test'], config['dataset'], config['data_seed'])
    path = a.metadata / 'split_audit.json'
    if path.exists() and read_json(path) != split['audit']:
        raise ValueError('The pinned data partition changed.')
    atomic_json(path, split['audit'])
    print(json.dumps({'prepared_model_roles': sorted(k for k in resolved if k != 'dataset'),
                      'resolved_revisions': resolved, 'data_audit': split['audit']}, indent=2), flush=True)


if __name__ == '__main__':
    from windows_io import replacement_retries
    with replacement_retries():
        main()
