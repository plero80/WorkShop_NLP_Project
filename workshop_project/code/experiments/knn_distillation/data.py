"""Reuse declared training prompts; reserve new refresh and final-test groups."""
from pathlib import Path
import gzip
import json
import random
import unicodedata
from knn_distillation.io import read, write, sha, digest


def group(prompt):
    from chat_format import _HH_ROLE_RE
    matches = list(_HH_ROLE_RE.finditer(prompt))
    first = prompt
    if matches and not prompt[:matches[0].start()].strip() and matches[0].group(1) == 'Human':
        first = prompt[matches[0].end():matches[1].start() if len(matches) > 1 else len(prompt)]
    first = ' '.join(unicodedata.normalize('NFKC', first).casefold().split())
    if not first:
        raise ValueError('Empty conversation opening.')
    return digest(first)


def groups(rows):
    return {group(row['prompt']) for row in rows}


def prior_data_files(project):
    files = set()
    for root in ('inputs/data', 'inputs/training'):
        files.update((project / root).glob('*.json'))
    for root in ('refresh2_outputs', 'next_studies_outputs', 'overopt_stress_outputs', 'refresh34_outputs', 'knn_distillation_outputs'):
        files.update((project / root).glob('*/data/*.json'))
    return sorted(files)


def schedule(pool, updates, batch, seed):
    """Repeated training epochs are intentional; the policy generates new answers."""
    if len(pool) < batch:
        raise ValueError('Training pool must contain at least one rollout batch.')
    rng = random.Random(seed)
    result = []
    while len(result) < updates * batch:
        epoch = list(pool)
        rng.shuffle(epoch)
        result.extend(epoch)
    return result[:updates * batch]

def partition_training(pool, o):
    pool = sorted(pool, key=lambda r: group(r['prompt']))
    if len(groups(pool)) != len(pool):
        raise ValueError('Duplicate conversation groups in source training pool.')
    random.Random(o['data_seed']).shuffle(pool)
    sizes = {'distill_train': o['train_prompts'], 'distill_validation': o['validation_prompts'],
             'distill_offline': o['offline_prompts']}
    if sum(sizes.values()) > len(pool):
        raise ValueError(f'Distillation requires {sum(sizes.values())} training-only groups; available {len(pool)}. '
                         'Declare smaller split sizes before starting; no held-out prompts are recycled.')
    result, at = {}, 0
    for name, size in sizes.items():
        result[name] = pool[at:at + size]; at += size
    return result

def validate_cohorts(data):
    seen = set()
    for name, rows in data.items():
        current = groups(rows)
        if len(current) != len(rows) or current & seen:
            raise ValueError('Overlapping/duplicate groups in ' + name)
        if len({r['prompt_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate prompt IDs in ' + name)
        seen.update(current)

def prepare(project, refresh2, out, c, o, assets):
    import pandas as pd
    from assets import resolve_files, DATASET, DATA_DIRS
    from chat_format import extract_hh_prompt, format_user_prompt
    from transformers import AutoTokenizer
    folder = out / 'data'; marker = folder / 'complete.json'
    options = {k: o[k] for k in ('train_prompts', 'validation_prompts', 'offline_prompts', 'final_prompts', 'data_seed')}
    if marker.exists():
        done = read(marker)
        if done['options'] != options:
            raise ValueError('Reserved distillation split configuration changed.')
        for name, expected in done['sha256'].items():
            if sha(folder / name) != expected:
                raise ValueError('Sealed distillation cohort changed: ' + name)
        data = {Path(name).stem: read(folder / name) for name in done['sha256']}
        validate_cohorts(data)
        return data
    pool = read(refresh2 / 'data/ppo3.json')
    bank = pd.read_csv(project / 'inputs/candidate_bank.csv', keep_default_na=False)
    memory_groups = {group(p) for p in bank.prompt}
    patterns = ['outputs/*/refresh/seed_*/added_examples.csv',
                'refresh2_outputs/*/memories/seed_*/added_examples.csv',
                'refresh34_outputs/*/memories/M*/seed_*/added_examples.csv']
    for pattern in patterns:
        for p in project.glob(pattern):
            frame = pd.read_csv(p, keep_default_na=False)
            memory_groups.update(group(x) for x in frame.prompt)
    if groups(pool) & memory_groups:
        raise ValueError('Source training prompts overlap stored teacher-memory prompt groups. Self-neighbor labels are not allowed.')
    data = partition_training(pool, o)
    data['monitor'] = read(refresh2 / 'data/refresh2_audit.json')
    forbidden = set(read(project / 'inputs/forbidden_opening_groups.json')) | memory_groups | groups(pool) | groups(data['monitor'])
    prior_evaluation = set()
    for p in prior_data_files(project):
        if p.parent == folder:
            continue
        r = read(p)
        if not isinstance(r, list):
            continue
        current = groups([row for row in r if isinstance(row, dict) and 'prompt' in row])
        forbidden.update(current)
        if any(word in p.stem.lower() for word in ('final', 'test', 'offline', 'monitor', 'audit', 'validation', 'selection')):
            prior_evaluation.update(current)
    if groups(data['distill_train']) & prior_evaluation:
        raise ValueError('Distillation training overlaps prior evaluation reservations. Keep the same fixed split seed or use a new training pool.')
    names = [d + '/test.jsonl.gz' for d in DATA_DIRS]
    snapshot = resolve_files(c, *DATASET, names, kind='dataset')
    tok = AutoTokenizer.from_pretrained(assets['policy'], local_files_only=True)
    eligible = {}
    for name in names:
        with gzip.open(snapshot / name, 'rt', encoding='utf-8') as handle:
            for index, line in enumerate(handle):
                try:
                    prompt = extract_hh_prompt(json.loads(line)['chosen']); key = group(prompt)
                except (ValueError, KeyError):
                    continue
                if key in forbidden or key in eligible:
                    continue
                if len(tok(format_user_prompt(tok, prompt))['input_ids']) > c['max_prompt_tokens']:
                    continue
                eligible[key] = {'prompt_id': 'kd_' + digest(prompt)[:24], 'conversation_group': key,
                                 'prompt': prompt, 'source': name, 'dataset_line': index}
    candidates = sorted(eligible.values(), key=lambda r: r['conversation_group'])
    random.Random(o['data_seed'] + 1).shuffle(candidates)
    if len(candidates) < o['final_prompts']:
        raise ValueError(f'Need {o["final_prompts"]} unused final HH test groups; found {len(candidates)}. No split was shrunk or reused.')
    data['final'] = candidates[:o['final_prompts']]
    validate_cohorts(data)
    for name, rows in data.items():
        write(folder / (name + '.json'), rows)
    write(marker, {'options': options, 'sha256': {name + '.json': sha(folder / (name + '.json')) for name in data},
          'counts': {name: len(rows) for name, rows in data.items()}, 'memory_prompt_overlap': False,
          'source_pool_sha256': sha(refresh2 / 'data/ppo3.json'), 'dataset': DATASET,
          'dataset_file_sha256': {name: sha(snapshot / name) for name in names}, 'eligible_final_groups': len(candidates),
          'training': 'Two new answers per training prompt, generated by base and source policy; teacher pseudo-labels.',
          'validation_and_offline': 'Held out from student fitting and NEW PPO. Source policy may have seen these prompts before.',
          'ppo': 'Repeated epochs over distill_train prompts only, with newly generated answers.',
          'final': 'Fresh HH test groups, excluded from all recognized old reservations.'})
    print('Reserved distillation prompts:', {k: len(v) for k, v in data.items()}, flush=True)
    return data

