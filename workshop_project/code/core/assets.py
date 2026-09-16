"""Resolve exact model/data revisions locally before downloading missing files."""
from pathlib import Path
import json
import os
import re
import time
from common import ROOT, read_json, write_json

MODELS = {
    'policy': ('Qwen/Qwen2.5-0.5B-Instruct', '7ae557604adf67be50417f59c2c2f167def9a775'),
    'proxy': ('Skywork/Skywork-Reward-V2-Qwen3-0.6B', '8c14a4e9e6321deaf572544339b16b8d6bbe8886'),
    'judge': ('Skywork/Skywork-Reward-V2-Qwen3-4B', 'fd958fef475f323f4e6b195930e3dd918485c668'),
}
DATASET = ('Anthropic/hh-rlhf', '09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa')
DATA_DIRS = ['harmless-base', 'helpful-base', 'helpful-online', 'helpful-rejection-sampled']
TOKENIZER_FILES = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
MODEL_FILES = {
    'policy': TOKENIZER_FILES + ['generation_config.json', 'model.safetensors'],
    'proxy': TOKENIZER_FILES + ['chat_template.jinja', 'model.safetensors'],
    'judge': TOKENIZER_FILES + ['chat_template.jinja',
                               'model.safetensors.index.json', 'model-00001-of-00002.safetensors',
                               'model-00002-of-00002.safetensors'],
}


def cache_roots(c):
    from huggingface_hub import constants
    choices = [c.get('extra_hf_cache'), os.environ.get('HF_HUB_CACHE'),
               str(constants.HF_HUB_CACHE), str(Path.home() / '.cache/huggingface/hub'),
               '/workspace/.cache/huggingface/hub', '/workspace/huggingface/hub',
               '/workspace/hf_cache', '/workspace/.hf_cache']
    if os.environ.get('HF_HOME'):
        choices.insert(0, str(Path(os.environ['HF_HOME']) / 'hub'))
    return list(dict.fromkeys(Path(p).expanduser().absolute() for p in choices if p))


def snapshot(root, repo, revision, kind='model'):
    return Path(root) / (('models--' if kind == 'model' else 'datasets--') + repo.replace('/', '--')) / 'snapshots' / revision


def present(path):
    return path.is_file() and path.stat().st_size > 0


def retry_seconds(headers):
    from email.utils import parsedate_to_datetime
    import math
    h = {str(k).lower(): str(v) for k, v in headers.items()}
    try:
        return max(1, math.ceil(float(h.get('retry-after', ''))))
    except (ValueError, OverflowError):
        try:
            return max(1, math.ceil(parsedate_to_datetime(h.get('retry-after', '')).timestamp() - time.time()))
        except (ValueError, TypeError, AttributeError, OverflowError):
            pass
    match = re.search(r'(?:^|[;,\s])t\s*=\s*(\d+)', h.get('ratelimit', ''))
    return max(1, int(match[1])) if match else 300


def resolve_files(c, repo, revision, names, kind='model'):
    from huggingface_hub import get_token, hf_hub_download
    roots = cache_roots(c)
    for root in roots:
        folder = snapshot(root, repo, revision, kind)
        if all(present(folder / n) for n in names):
            return folder
    counts = [sum(present(snapshot(r, repo, revision, kind) / n) for n in names) for r in roots]
    root = roots[max(range(len(roots)), key=lambda i: counts[i])]
    folder = snapshot(root, repo, revision, kind)
    if not c['allow_downloads']:
        raise FileNotFoundError('Complete cached revision not found for ' + repo + '. Set extra_hf_cache or enable downloads during preflight.')
    record = ROOT / 'cache/hub_download_status.json'
    if record.is_file() and read_json(record).get('retry_not_before', 0) > time.time():
        raise RuntimeError('Hugging Face cooldown is active. See cache/hub_download_status.json; no request sent.')
    token = (os.environ.get('HF_TOKEN') or get_token() or '').strip() or False
    for name in names:
        if present(folder / name):
            continue
        print('Downloading', repo, name, '| token available:', bool(token), flush=True)
        try:
            hf_hub_download(repo, name, revision=revision, repo_type=kind,
                            cache_dir=str(root), token=token)
        except Exception as error:
            cause, response, seen = error, None, set()
            while cause is not None and id(cause) not in seen:
                seen.add(id(cause))
                response = getattr(cause, 'response', None)
                if response is not None:
                    break
                cause = cause.__cause__ or cause.__context__
            code = getattr(response, 'status_code', None)
            info = {'repo': repo, 'revision': revision, 'file': name, 'http_status': code,
                    'error_type': type(error).__name__, 'token_supplied': bool(token)}
            if code == 429:
                delay = retry_seconds(getattr(response, 'headers', {}))
                info.update(retry_after_seconds=delay, retry_not_before=time.time()+delay)
            write_json(record, info)
            raise RuntimeError(f'Asset download failed: HTTP {code}, {type(error).__name__}. See cache/hub_download_status.json. '
                               'For 429, wait for the provider reset; adding a token does not override a rate limit.') from None
    if not all(present(folder / n) for n in names):
        raise FileNotFoundError('Snapshot remains incomplete: ' + str(folder))
    return folder


def resolve_all(c):
    paths = {role: str(resolve_files(c, *info, MODEL_FILES[role])) for role, info in MODELS.items()}
    print('All three pinned models are ready. Prompt data and adapters are bundled.', flush=True)
    return paths
