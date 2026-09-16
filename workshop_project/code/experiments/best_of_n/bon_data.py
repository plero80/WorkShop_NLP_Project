"""Reserve fresh HH conversation groups before examining candidate rewards."""
import csv, gzip, random, unicodedata
from pathlib import Path
from bon_io import read, write, digest, sha

def group(prompt):
    from chat_format import prompt_messages
    first = next(m['content'] for m in prompt_messages(prompt) if m['role'] == 'user')
    return digest(' '.join(unicodedata.normalize('NFKC', first).casefold().split()))

def collect_forbidden(project, own_output):
    forbidden = set(read(project / 'inputs/forbidden_opening_groups.json'))
    sources = {}
    # Audit all recognized project cohort stores, including suites reserved but not run.
    roots = [project/'inputs', project/'outputs', project/'refresh2_outputs',
             project/'next_studies_outputs', project/'overopt_stress_outputs',
             project/'stress_outputs', project/'best_of_n_outputs']
    for root in roots:
        if not root.exists(): continue
        for path in sorted(root.rglob('*')):
            if not path.is_file() or own_output in path.parents: continue
            relative = path.relative_to(root)
            if any(p in ('shards','checkpoints','vendor','reports','review','evaluations','runs') for p in relative.parts): continue
            is_cohort = any(p in ('data','suite_data','training') for p in relative.parts)
            is_bank = path.name in ('candidate_bank.csv','added_examples.csv')
            if not is_cohort and not is_bank: continue
            if path.suffix not in ('.json','.csv','.jsonl'): continue
            sources[str(path.relative_to(project))] = sha(path)
            if path.suffix == '.csv':
                with open(path, newline='', encoding='utf-8') as f:
                    for row in csv.DictReader(f):
                        if row.get('prompt'): forbidden.add(group(row['prompt']))
            else:
                rows = read(path) if path.suffix == '.json' else [__import__('json').loads(l) for l in path.read_text().splitlines() if l.strip()]
                if isinstance(rows, list):
                    for row in rows:
                        if isinstance(row, dict) and row.get('prompt'): forbidden.add(group(row['prompt']))
    return forbidden, sources

def reserve(project, output, c, settings, tokenizer):
    from assets import DATASET, DATA_DIRS, resolve_files
    from chat_format import extract_hh_prompt, format_user_prompt
    folder = output / 'data'; marker = folder / 'complete.json'
    if marker.exists():
        meta = read(marker)
        for name, h in meta['sha256'].items():
            if sha(folder/name) != h: raise ValueError('Reserved prompts changed.')
        return {name: read(folder/(name+'.json')) for name in ('development','confirmation')}
    forbidden, sources = collect_forbidden(project, output)
    names = [d+'/'+settings['dataset_split']+'.jsonl.gz' for d in DATA_DIRS]
    base = resolve_files(c, *DATASET, names, kind='dataset')
    pool = {}; malformed = 0; too_long = 0
    import json
    for name in names:
        with gzip.open(base/name, 'rt', encoding='utf-8') as f:
            for line_number, line in enumerate(f):
                try:
                    prompt = extract_hh_prompt(json.loads(line)['chosen'])
                    g = group(prompt)
                except (ValueError, KeyError, StopIteration):
                    malformed += 1; continue
                if g in forbidden or g in pool: continue
                if len(tokenizer(format_user_prompt(tokenizer, prompt))['input_ids']) > c['max_prompt_tokens']:
                    too_long += 1; continue
                pool[g] = {'prompt_id':'bon_'+digest(prompt)[:24], 'conversation_group':g,
                           'prompt':prompt, 'source':name, 'dataset_line':line_number}
    rows = sorted(pool.values(), key=lambda x:x['conversation_group'])
    random.Random(settings['data_seed']).shuffle(rows)
    total = settings['development_prompts'] + settings['confirmation_prompts']
    if len(rows) < total:
        raise ValueError(f'Need {total} NEW conversation groups; only {len(rows)} eligible. No reuse or shrinking occurred. '
                         'Review data/length exclusions; choose smaller explicit cohort counts or explicitly set dataset_split=test in a NEW protocol.')
    d = settings['development_prompts']
    cohorts = {'development':rows[:d], 'confirmation':rows[d:total]}
    assert len({r['conversation_group'] for r in rows[:total]}) == total
    for name, cohort in cohorts.items(): write(folder/(name+'.json'), cohort)
    write(marker, {'sha256':{name+'.json':sha(folder/(name+'.json')) for name in cohorts},
                   'dataset':DATASET, 'dataset_sha256':{name:sha(base/name) for name in names},
                   'exclusion_sources_sha256':sources, 'forbidden_groups':len(forbidden),
                   'eligible_groups':len(rows), 'malformed':malformed, 'too_long':too_long,
                   'opening_group_disjoint':True})
    return cohorts
