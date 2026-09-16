"""Independent LoRA reward student. Inference needs no teacher or memory."""
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from knn_distillation.io import read, write, sha, digest


def dtype_argument():
    import transformers
    return 'dtype' if int(transformers.__version__.split('.')[0]) >= 5 else 'torch_dtype'


def add_adapter(base, options):
    from peft import LoraConfig, get_peft_model
    base.requires_grad_(False)
    cfg = LoraConfig(task_type='SEQ_CLS', r=options['student_lora_rank'],
                     lora_alpha=options['student_lora_alpha'], lora_dropout=0., bias='none',
                     target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], modules_to_save=['score'])
    model = get_peft_model(base, cfg)
    for name, p in model.named_parameters():
        if p.requires_grad:
            if 'lora_' not in name and '.score.modules_to_save.' not in name:
                raise ValueError('Unexpected trainable reward parameter: ' + name)
            p.data = p.data.float()
    if not any(p.requires_grad and '.score.modules_to_save.' in name for name, p in model.named_parameters()):
        raise ValueError('The scalar reward head was not included in student training.')
    return model


class StudentScorer:
    def __init__(self, model, tokenizer, calibration, max_length=4096, batch_size=4, device='cuda', identity='untrained'):
        self.model, self.tokenizer, self.calibration = model, tokenizer, calibration
        self.max_length, self.batch_size, self.device, self.identity = max_length, batch_size, device, identity
        tokenizer.padding_side = 'right'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
        model.to(device).eval()

    @classmethod
    def create(cls, snapshot, calibration, c, o, seed, device='cuda'):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.manual_seed(seed)
        if device.startswith('cuda'):
            torch.cuda.manual_seed_all(seed)
        base = AutoModelForSequenceClassification.from_pretrained(str(snapshot), local_files_only=True,
               attn_implementation='sdpa', **{dtype_argument(): torch.bfloat16 if device.startswith('cuda') else torch.float32})
        if base.score.out_features != 1:
            raise ValueError('Expected a pretrained scalar proxy reward head.')
        model = add_adapter(base, o)
        if o['gradient_checkpointing']:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
        return cls(model, tokenizer, calibration, c['reward_max_tokens'], o['student_batch_size'], device)

    @classmethod
    def load(cls, snapshot, folder, batch_size=16, device='cuda'):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from peft import PeftModel
        folder = Path(folder); metadata = verify_bundle(folder)
        if sha(Path(snapshot) / 'config.json') != metadata['base_config_sha256']:
            raise ValueError('Student base architecture/config differs from its saved source.')
        for name, expected in metadata.get('base_weight_sha256', {}).items():
            if sha(Path(snapshot) / name) != expected:
                raise ValueError('Student base weights differ from the pinned source: ' + name)
        base = AutoModelForSequenceClassification.from_pretrained(str(snapshot), local_files_only=True,
               attn_implementation='sdpa', **{dtype_argument(): torch.bfloat16 if device.startswith('cuda') else torch.float32})
        model = PeftModel.from_pretrained(base, str(folder), is_trainable=False, local_files_only=True)
        model.requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
        return cls(model, tokenizer, metadata['calibration'], metadata['max_length'], batch_size, device, digest(metadata))

    def encode(self, prompts, answers):
        from chat_format import format_prompt_answer
        if not prompts or len(prompts) != len(answers):
            raise ValueError('Nonempty aligned prompts and answers required.')
        text = [format_prompt_answer(self.tokenizer, p, a) for p, a in zip(prompts, answers)]
        encoded = self.tokenizer(text, padding=True, truncation=False, return_tensors='pt')
        lengths = encoded['attention_mask'].sum(1)
        limit = min(self.max_length, self.model.config.max_position_embeddings)
        if int(lengths.max()) > limit:
            raise ValueError(f'Whole student input requires {int(lengths.max())} tokens; guard is {limit}. No truncation was performed.')
        return encoded.to(self.device), lengths

    def raw_forward(self, prompts, answers):
        encoded, lengths = self.encode(prompts, answers)
        autocast = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if self.device.startswith('cuda') else nullcontext()
        with autocast:
            raw = self.model(**encoded).logits.reshape(-1).float()
        if not torch.isfinite(raw).all():
            raise FloatingPointError('Nonfinite student reward.')
        return raw, lengths

    def normalized_forward(self, prompts, answers):
        raw, _ = self.raw_forward(prompts, answers)
        return (raw - self.calibration['proxy_mean']) / self.calibration['proxy_std']

    @torch.inference_mode()
    def score(self, prompts, answers):
        if len(prompts) != len(answers):
            raise ValueError('Mismatched prompt/answer counts.')
        values, lengths = [], []
        self.model.eval()
        for start in range(0, len(prompts), self.batch_size):
            raw, n = self.raw_forward(prompts[start:start + self.batch_size], answers[start:start + self.batch_size])
            values.extend(raw.cpu().tolist()); lengths.extend(n.cpu().tolist())
        return {'raw': np.asarray(values, float), 'tokens': np.asarray(lengths, int), 'truncated': np.zeros(len(values), bool)}


def save_bundle(scorer, folder, metadata):
    folder = Path(folder)
    scorer.model.save_pretrained(folder, safe_serialization=True)
    scorer.tokenizer.save_pretrained(folder)
    files = {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*'))
             if p.is_file() and p.name != 'reward_config.json'}
    write(folder / 'reward_config.json', {**metadata, 'calibration': scorer.calibration,
          'max_length': scorer.max_length, 'files': files, 'runtime_reward': '(student_raw - proxy_mean) / proxy_std',
          'knn_lookup_required': False, 'judge_required': False, 'original_proxy_forward_required': False})


def verify_bundle(folder):
    folder = Path(folder); metadata = read(folder / 'reward_config.json')
    for name, expected in metadata['files'].items():
        path = (folder / name).resolve(); path.relative_to(folder.resolve())
        if sha(path) != expected:
            raise ValueError('Student inference artifact changed: ' + name)
    if metadata['calibration']['proxy_std'] <= 0:
        raise ValueError('Invalid student normalization scale.')
    return metadata
