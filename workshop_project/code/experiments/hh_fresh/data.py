"""Reserve disjoint conversation groups directly from pinned Anthropic HH data."""
import gzip
import json
from pathlib import Path
import random

from chat_format import extract_hh_prompt, format_user_prompt
from knn_distillation.data import group, validate_cohorts
from hh_ridge_ppo.protocol import read, write, sha, digest, require


def partition(train, test, counts, seed):
    """Reserve all cohorts before scoring; exclude test openings from train."""
    test = sorted(test, key=lambda r: r['conversation_group'])
    test_groups = {r['conversation_group'] for r in test}
    train = sorted([r for r in train if r['conversation_group'] not in test_groups], key=lambda r: r['conversation_group'])
    require(len({r['conversation_group'] for r in train}) == len(train) and len(test_groups) == len(test), 'Deduplicate HH conversation openings before splitting')
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(test)
    data, positions = {}, {'train': 0, 'test': 0}
    for name in sorted(counts):
        count = counts[name]
        split = 'test' if name in ('final', 'refresh2_eval') else 'train'
        pool, start = (test if split == 'test' else train), positions[split]
        require(start+count <= len(pool), f'Insufficient disjoint HH {split} prompts for {name}; no cohort was silently reduced')
        data[name] = pool[start:start+count]
        positions[split] += count
    validate_cohorts(data)
    return data


def prepare(out, c, recipe, assets):
    folder = out / 'data'
    expected = {'counts': recipe['counts'], 'seed': recipe['data_seed'], 'max_prompt_tokens': c['max_prompt_tokens']}
    if (folder / 'complete.json').exists():
        done = read(folder / 'complete.json')
        require(done['options'] == expected, 'Fresh split settings changed')
        for name, value in done['sha256'].items():
            require(sha(folder / name) == value, 'Reserved HH prompts changed: '+name)
        data = {Path(n).stem: read(folder / n) for n in done['sha256']}
        validate_cohorts(data)
        return data
    from assets import resolve_files, DATASET, DATA_DIRS
    from transformers import AutoTokenizer
    names = [d+'/'+split+'.jsonl.gz' for d in DATA_DIRS for split in ('train', 'test')]
    snapshot = resolve_files(c, *DATASET, names, kind='dataset')
    tokenizer = AutoTokenizer.from_pretrained(assets['policy'], local_files_only=True)
    pools, skipped = {'train': {}, 'test': {}}, {'malformed': 0, 'long': 0, 'duplicates': 0}
    for name in names:
        split = 'test' if name.endswith('/test.jsonl.gz') else 'train'
        with gzip.open(snapshot / name, 'rt', encoding='utf-8') as stream:
            for line_number, line in enumerate(stream):
                try:
                    prompt = extract_hh_prompt(json.loads(line)['chosen'])
                    key = group(prompt)
                except (ValueError, KeyError, TypeError):
                    skipped['malformed'] += 1
                    continue
                if key in pools[split]:
                    skipped['duplicates'] += 1
                    continue
                if len(tokenizer(format_user_prompt(tokenizer, prompt))['input_ids']) > c['max_prompt_tokens']:
                    skipped['long'] += 1
                    continue
                pools[split][key] = {'prompt_id': 'fresh_'+digest(prompt)[:24], 'conversation_group': key,
                                     'prompt': prompt, 'source': name, 'dataset_line': line_number}
        print(f'HH prompts: read {name}; retained {len(pools[split])} unique {split} openings', flush=True)
    data = partition(list(pools['train'].values()), list(pools['test'].values()), recipe['counts'], recipe['data_seed'])
    for name, rows in data.items():
        write(folder / (name+'.json'), rows)
    write(folder / 'complete.json', {'options': expected, 'dataset': DATASET,
          'dataset_sha256': {n: sha(snapshot / n) for n in names}, 'skipped': skipped,
          'sha256': {n+'.json': sha(folder / (n+'.json')) for n in data},
          'no_old_experiment_inputs': True, 'all_conversation_groups_disjoint': True})
    return data
