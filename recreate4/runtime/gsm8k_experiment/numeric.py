#!/usr/bin/env python3
"""Re-evaluate saved GSM8K answers on CPU. Python standard library only.

Run inside the existing project, or use --input PROJECT|EXPORTED_FOLDER|ZIP.
Numerical answer matching, original strict correctness, and format compliance
are reported separately. Ambiguous/incomplete answers go to a review queue.
No models, API calls, training, gold-assisted extraction, or source edits.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import random
import re
import sys
import zipfile

VERSION = 'gsm8k_numeric_recheck_v1'
ARMS = ('base', 'proxy', 'judge', 'knn_static')
PROTOCOL = {
    'version': VERSION,
    'scope': 'Final numeric-answer matching only; reasoning and unit semantics are not adjudicated.',
    'status': 'Post-hoc re-evaluation of previously inspected results; not preregistered.',
    'extraction_inputs': ['response text', 'length_capped', 'ended_with_eos'],
    'incomplete': 'Unresolved: do not guess an answer from an unfinished response.',
    'box': 'One parseable numeric box is accepted unless followed by an explicit conflicting answer declaration. Multiple boxes require an unambiguous final-paragraph box; otherwise unresolved.',
    'plain_text': 'Use only the last nonempty paragraph, at most 400 characters. Accept one numeric literal, an explicit Answer/Result declaration, or a numeric final right-hand side of an equation.',
    'ambiguity': 'Reject question/conditional/negative conclusions, unsupported arithmetic and conflicting or multiple unanchored numbers. Never search for the gold value.',
    'normalization': 'Exact rational comparison; signs, decimal/scientific notation, thousands grouping and simple fractions supported. Numeric percentage values stay in the written scale; units are not converted.',
    'metrics': 'Original strict correctness; strict-format compliance; confirmed numeric matches / all responses; resolved numeric mismatches; unresolved rate. Unresolved cases remain separate and are not dropped from the denominator.',
    'fairness': 'Identical extraction rules for all arms; identical question IDs and references required.',
}
PLAIN = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
GROUPED = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?|[+-]?\.\d+'
ATOM = rf'(?:\\(?:d?frac)\{{{PLAIN}\}}\{{{PLAIN}\}}|(?:{GROUPED})(?:\s*/\s*(?:{GROUPED}))?)'
NUMBER = re.compile(rf'(?<![\w.,]){ATOM}(?![\w,])')
UNSURE = re.compile(r"(?i)\b(?:if|cannot|can't|not|maybe|might|unsure|either|neither)\b")
UNITS = re.compile(r'^[A-Za-z%²³°][A-Za-z\s%²³°.-]*$')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def numeric(text):
    """Parse a numeric atom exactly; no eval, unit conversion, or gold access."""
    if text is None:
        return None
    t = str(text).strip().replace('−','-').replace('\\,','').strip('$').strip()
    t = re.sub(r'^\\(?:boxed|text)\{([^{}]+)\}$',r'\1',t)
    t = re.sub(r'\\(?:dfrac|tfrac)',r'\\frac',t)
    m = re.fullmatch(r'\\frac\{('+PLAIN+r')\}\{('+PLAIN+r')\}',t)
    if m:
        t = m[1]+'/'+m[2]
    if len(t)>120:
        return None
    if ',' in t:
        if not re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?',t):
            return None
        t=t.replace(',','')
    parts=t.split('/')
    if len(parts) not in (1,2) or any(not re.fullmatch(PLAIN,p.strip()) for p in parts):
        return None
    try:
        for p in parts:
            exp=re.search(r'[eE]([+-]?\d+)$',p.strip())
            if exp and abs(int(exp[1]))>100:
                return None
        result=Fraction(parts[0].strip())
        if len(parts)==2:
            result/=Fraction(parts[1].strip())
        return str(result)
    except (ValueError,ZeroDivisionError,OverflowError):
        return None


def boxes(text):
    result=[]
    for m in re.finditer(r'\\boxed\s*\{',text):
        start=m.end();level=1;end=start
        while end<len(text) and level:
            level+=(text[end]=='{')-(text[end]=='}');end+=1
        if level==0:
            result.append((m.start(),end,text[start:end-1]))
    return result


def original_check(text,reference):
    gold=numeric(reference.rsplit('####',1)[-1])
    if gold is None:
        raise ValueError('Unsupported reference numeric answer.')
    found=boxes(text)
    pred=numeric(found[0][2]) if len(found)==1 else None
    return {'correct':pred is not None and pred==gold,'format_valid':pred is not None,
            'boxed_count':len(found),'predicted_answer':pred,'gold_answer':gold}


def plain_display(text):
    t=text.replace('−','-').replace('\\,','')
    t=re.sub(r'\\(?:text|mathrm)\{([^{}]*)\}',r'\1',t)
    for item in ('\\[','\\]','\\(','\\)','\\$','$','**','`','\\!'):
        t=t.replace(item,'')
    return t.strip()


def quantity(text):
    """One number, optionally decorated with simple units/punctuation."""
    t=plain_display(text).strip().rstrip(' .!;:')
    m=NUMBER.fullmatch(t)
    if m:
        return numeric(m[0])
    m=NUMBER.match(t)
    if not m:
        return None
    suffix=t[m.end():].strip().strip('.!;:').strip()
    if not suffix or (UNITS.fullmatch(suffix) and not re.search(r'(?i)\b(?:or|and|but|instead|wrong)\b',suffix)):
        return numeric(m[0])
    return None


def conflicting_tail(value, text):
    tail=plain_display(text)
    declaration=re.search(r'(?i)\b(?:final\s+)?(?:answer|result)\s*(?:is|:|=)\s*(.+)$',tail)
    if declaration:
        other=quantity(declaration[1])
        return other is None or other!=value
    return bool(re.search(r'(?i)\b(?:actually|instead|correction)\b',tail) and NUMBER.search(tail))


def decision(value=None,method='unresolved',evidence='',reason=''):
    return {'prediction':value,'method':method,'evidence':evidence,'reason':reason}


def extract_answer(response, *, length_capped=False, ended_with_eos=True):
    """This signature deliberately excludes question, reference, grades and arm."""
    if length_capped or not ended_with_eos:
        return decision(reason='incomplete_response')
    text=response.strip()
    if not text:
        return decision(reason='empty_response')
    last=re.split(r'\n\s*\n',text)[-1].strip()
    found=boxes(text)
    if len(found)==1:
        value=quantity(found[0][2])
        if value is not None:
            # Repeated problem quantities after a box are common and are not
            # alternative answers. Only an explicit conflicting declaration is
            # ambiguous here; no comparison to the reference is involved.
            if conflicting_tail(value,text[found[0][1]:]):
                return decision(evidence=text[found[0][0]:],reason='conflicting_answer_after_box')
            return decision(value,'single_numeric_box',found[0][2])
        return decision(evidence=found[0][2],reason='unsupported_box_contents')
    if len(found)>1:
        tail_boxes=boxes(last)
        if len(tail_boxes)==1:
            value=quantity(tail_boxes[0][2])
            if value is not None:
                prefix=plain_display(last[:tail_boxes[0][0]])
                if not UNSURE.search(prefix) and '?' not in prefix and not conflicting_tail(value,last[tail_boxes[0][1]:]):
                    return decision(value,'final_paragraph_box',last)
        return decision(evidence=last,reason='multiple_boxes_without_clear_final_answer')
    if '\\boxed' in text:
        return decision(evidence=last,reason='unclosed_or_malformed_box')
    if len(last)>400:
        return decision(evidence=last,reason='long_final_paragraph')
    cleaned=plain_display(last)
    if '?' in cleaned or UNSURE.search(cleaned):
        return decision(evidence=last,reason='question_conditional_or_negative_conclusion')
    # Explicit declarations and an equation's final RHS provide an answer anchor.
    declaration=re.search(r'(?i)(?:\b(?:final\s+)?answer\s*(?:is|:|=)|\bresult\s*(?:is|:|=)|####)\s*(.+)$',cleaned)
    if declaration:
        value=quantity(declaration[1])
        if value is not None:
            return decision(value,'explicit_answer_declaration',last)
        return decision(evidence=last,reason='ambiguous_answer_declaration')
    if '=' in cleaned:
        value=quantity(cleaned.rsplit('=',1)[-1])
        if value is not None:
            return decision(value,'final_equation_rhs',last)
        return decision(evidence=last,reason='unsupported_final_equation')
    hits=list(NUMBER.finditer(cleaned))
    if len(hits)==1:
        value=numeric(hits[0][0])
        # Reject an unmatched fraction/arithmetic operator adjacent to the number.
        outside=cleaned[:hits[0].start()]+cleaned[hits[0].end():]
        if re.search(r'\\(?:d?frac)|[=<>+*/]|\d',outside):
            return decision(evidence=last,reason='unsupported_numeric_expression')
        if value is not None:
            return decision(value,'one_number_final_paragraph',last)
    return decision(evidence=last,reason='no_unique_final_number')


class InputData:
    def __init__(self,path):
        self.path=Path(path).expanduser().resolve()
        self.archive=None
        if self.path.is_file() and zipfile.is_zipfile(self.path):
            self.archive=zipfile.ZipFile(self.path)
            names=self.archive.namelist()
            if len(names)!=len(set(names)):
                raise ValueError('Archive contains duplicate filenames.')
            candidates=[n[:-len('config.json')] for n in names if n.endswith('config.json') and '/evaluations/' not in n]
            roots=[r for r in candidates if any(n.startswith(r+'evaluations/final/') and n.endswith('/responses.jsonl') for n in names)]
            if len(roots)!=1:
                raise ValueError('Cannot identify one outcomes directory in the archive.')
            self.prefix=roots[0]
            self.names={n[len(self.prefix):] for n in names if n.startswith(self.prefix)}
        elif self.path.is_dir():
            candidates=[self.path,self.path/'outcomes',self.path/'gsm8k_outputs/main']
            roots=[r for r in candidates if (r/'config.json').is_file() and (r/'evaluations/final').is_dir()]
            if len(roots)!=1:
                raise ValueError('Pass the project directory, outcomes directory, or original export ZIP.')
            self.root=roots[0]
            self.names={p.relative_to(self.root).as_posix() for p in self.root.rglob('*') if p.is_file()}
        else:
            raise ValueError(f'Input not found: {self.path}')
        self.used_hashes={}

    def read(self,name):
        if name not in self.names:
            raise ValueError(f'Required saved file is missing: {name}')
        data=self.archive.read(self.prefix+name) if self.archive else (self.root/name).read_bytes()
        self.used_hashes[name]=hashlib.sha256(data).hexdigest()
        return data.decode('utf-8')

    def json(self,name):
        return json.loads(self.read(name))

    def rows(self,name):
        return [json.loads(line) for line in io.StringIO(self.read(name)) if line.strip()]

    def close(self):
        if self.archive:self.archive.close()


def default_input():
    for root in (Path(__file__).resolve().parents[1], Path.cwd()):
        if (root/'gsm8k_outputs/main/evaluations/final').is_dir():return root
    raise ValueError('No saved final answers found. Use --input /path/to/project or --input important_outcomes_gsm8k.zip')


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['review_id'])
        writer.writeheader();writer.writerows(rows)


def recheck(input_path,output_path=None):
    data=InputData(input_path)
    try:
        config=data.json('config.json')
        protocol=dict(PROTOCOL)
        prospective=config.get('evaluation',{}).get('numeric_protocol_frozen_before_run',False)
        if prospective:
            protocol['status']='Frozen before this follow-up run; developed after inspecting the earlier experiment.'

        declared=data.json('final_protocol.json') if 'final_protocol.json' in data.names else {}
        arms=('base', *declared.get('arms', config.get('arms', ARMS[1:])))
        allowed={'base','proxy','judge','knn_static','knn_static_30b','knn_refresh','oracle'}
        if len(arms)!=len(set(arms)) or not set(arms)<=allowed or len(arms)<2:
            raise ValueError('Unknown, empty, or duplicate experiment arms.')
        rows_by_arm={}; final_paths={}; steps={}
        for arm in arms:
            matches=sorted(n for n in data.names if re.fullmatch(r'evaluations/final/'+arm+r'/step_\d+/responses\.jsonl',n))
            if len(matches)!=1:
                raise ValueError(f'Expected exactly one final evaluation for {arm}; found {len(matches)}. No checkpoint was selected automatically.')
            name=matches[0];final_paths[arm]=name;steps[arm]=int(name.split('/')[3][5:])
            rows=data.rows(name)
            ids=[r['id'] for r in rows]
            if len(ids)!=len(set(ids)) or not ids:
                raise ValueError(f'Duplicate or empty question IDs in {arm}.')
            rows_by_arm[arm]={r['id']:r for r in rows}
        common=set(rows_by_arm['base'])
        expected=config.get('dataset',{}).get('final')
        if expected is not None and len(common)!=expected:
            raise ValueError(f'Expected {expected} final questions; found {len(common)}.')
        if steps['base']!=0 or len({steps[a] for a in arms if a!='base'})!=1:
            raise ValueError('Final arm checkpoints do not share one declared training step.')
        for arm,rows in rows_by_arm.items():
            if set(rows)!=common:
                raise ValueError(f'Final question IDs differ for {arm}.')
            for key,r in rows.items():
                if any(r[field]!=rows_by_arm['base'][key][field] for field in ('question','reference')):
                    raise ValueError(f'Question/reference mismatch for {arm}, {key}.')
        metrics=[]; decisions=[]; reviews=[]; review_keys=[]; changes=[]
        for arm in arms:
            stats={'arm':arm,'questions':len(common),'update':steps[arm],'original_strict_correct':0,
                   'strict_format_valid':0,'numeric_matches':0,'numeric_mismatches':0,'unresolved':0,
                   'new_numeric_matches':0,'original_matches_now_unresolved':0}
            for key in sorted(common):
                r=rows_by_arm[arm][key]
                original=original_check(r['response'],r['reference'])
                for field in ('correct','format_valid','boxed_count','predicted_answer','gold_answer'):
                    if field in r and r[field]!=original[field]:
                        raise ValueError(f'Saved original {field} cannot be reproduced: {arm}, {key}.')
                prediction=extract_answer(r['response'],length_capped=r.get('length_capped',False),
                                          ended_with_eos=r.get('ended_with_eos',True))
                if prediction['prediction'] is None:
                    status='unresolved'
                elif prediction['prediction']==original['gold_answer']:
                    status='numeric_match'
                else:
                    status='numeric_mismatch'
                item={'arm':arm,'id':key,'original_strict_correct':original['correct'],
                      'strict_format_valid':original['format_valid'],'status':status,**prediction,
                      'gold_answer':original['gold_answer'],'length_capped':r.get('length_capped',False)}
                decisions.append(item)
                stats['original_strict_correct']+=int(original['correct'])
                stats['strict_format_valid']+=int(original['format_valid'])
                stats[{'numeric_match':'numeric_matches','numeric_mismatch':'numeric_mismatches','unresolved':'unresolved'}[status]]+=1
                if status=='numeric_match' and not original['correct']:
                    stats['new_numeric_matches']+=1
                    changes.append({**item,'question':r['question'],'response':r['response'],'reference':r['reference']})
                if original['correct'] and status=='unresolved':stats['original_matches_now_unresolved']+=1
                if status=='unresolved':
                    # Review export deliberately excludes gold, arm and all model grades.
                    rid=hashlib.sha256((VERSION+'|'+arm+'|'+key).encode()).hexdigest()[:20]
                    reviews.append({'review_id':rid,'question':r['question'],'response':r['response'],
                                    'reason':prediction['reason'],'extracted_final_answer':'','reviewer_note':''})
                    review_keys.append({'review_id':rid,'arm':arm,'id':key,'gold_answer':original['gold_answer']})
            n=len(common)
            stats.update(original_strict_accuracy=stats['original_strict_correct']/n,
                         format_compliance=stats['strict_format_valid']/n,
                         confirmed_numeric_match_rate=stats['numeric_matches']/n,
                         unresolved_rate=stats['unresolved']/n,
                         resolved_fraction=(stats['numeric_matches']+stats['numeric_mismatches'])/n)
            metrics.append(stats)
        random.Random(42).shuffle(reviews)
        source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        identity={'protocol':protocol,'evaluator_sha256':source_sha,'inputs':data.used_hashes}
        run_id=digest(identity)[:16]
        if output_path:
            output=Path(output_path).expanduser().resolve()
        else:
            parent=data.root.parent if not data.archive else data.path.parent
            output=parent/('gsm8k_answer_recheck_'+run_id)
        # Never write into the original outcomes tree, or overwrite another audit.
        if not data.archive and (output==data.root or output.is_relative_to(data.root)):
            raise ValueError('Choose an output directory outside the original outcomes directory.')
        if output.exists():
            manifest=output/'manifest.json'
            if manifest.is_file() and json.loads(manifest.read_text()).get('identity')==identity and (output/'COMPLETE.json').is_file():
                print(f'Identical re-evaluation already completed: {output / "report.md"}')
                return output
            raise ValueError(f'Output already exists or is incomplete; choose a new --output directory: {output}')
        output.mkdir(parents=True)
        manifest={'identity':identity,'created_at':datetime.now(timezone.utc).isoformat(),
                  'source_input':str(data.path),'cohort':'final','run_id':run_id}
        write_json(output/'manifest.json',manifest)
        write_json(output/'protocol.json',protocol)
        write_json(output/'summary.json',{'protocol_version':VERSION,'metrics':metrics,'review_required':len(reviews)})
        write_csv(output/'metrics.csv',metrics)
        write_csv(output/'answer_decisions.csv',decisions)
        write_csv(output/'review_blinded.csv',reviews)
        write_csv(output/'review_key.csv',review_keys)
        with (output/'new_numeric_matches.jsonl').open('w') as f:
            for r in changes:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        lines=['# GSM8K saved-answer re-evaluation','',f'Protocol: {VERSION}. Final cohort: {len(common)} questions per arm. Seed: {config.get("seed")}.',
               '', ('This is a CPU replay of the numeric protocol frozen before this run. ' if prospective else 'This is a post-hoc, CPU-only re-evaluation of the same saved answers. ') + 'The original strict results remain unchanged. '
               'Numeric matching checks the final value, not mathematical reasoning, units, or instruction compliance. '
               'Ambiguous and unfinished responses are unresolved and require review; all rows remain in the denominator.', '',
               '| Policy | Original strict accuracy | Format compliance | Confirmed numeric matches / all | Numeric mismatches | Unresolved |',
               '|---|---:|---:|---:|---:|---:|']
        for s in metrics:
            lines.append(f'| {s["arm"]} | {s["original_strict_accuracy"]:.2%} | {s["format_compliance"]:.2%} | {s["confirmed_numeric_match_rate"]:.2%} ({s["numeric_matches"]}/{s["questions"]}) | {s["numeric_mismatches"]} | {s["unresolved"]} ({s["unresolved_rate"]:.2%}) |')
        lines += ['', 'The confirmed-match column is provisional while unresolved cases remain. It is not accuracy among a selected resolved subset. '
                  'Do not treat it as a complete semantic evaluation or replace the declared original metric without reporting this protocol change.', '',
                  'Extraction uses only response text and completion flags. Exactly parseable boxes, explicit final Answer/Result declarations, '
                  'a numeric final equation right-hand side, or a last paragraph with one number are supported. '
                  'There is no arbitrary last-number search through the solution and no gold-assisted answer selection. '
                  'Decimal/currency decoration, conventional commas, simple fractions, and LaTeX numeric wrappers are normalized exactly. '
                  'Percentage numbers retain their written scale. Cases needing unit conversion or semantic interpretation may still require manual review.', '',
                  'review_blinded.csv contains unresolved responses with randomized order and no arm, gold answer or model grade. '
                  'Fill extracted_final_answer only when the response states a clear answer; use reviewer_note for ambiguity or incompleteness. '
                  'review_key.csv is the separate mapping for later scoring. Review should extract the stated answer, not solve the problem and substitute an answer.', '',
                  'answer_decisions.csv records every decision, evidence and reason. new_numeric_matches.jsonl contains all newly matched answers. '
                  'manifest.json records source hashes and the exact evaluation protocol. No original outputs were edited.', '']
        (output/'report.md').write_text('\n'.join(lines))
        (output/'RECHECK_GSM8K_ANSWERS.py').write_bytes(Path(__file__).read_bytes())
        # Check source data again before declaring a complete audit.
        initial_hashes=dict(data.used_hashes)
        for name,expected_hash in initial_hashes.items():
            data.read(name)
            if data.used_hashes[name]!=expected_hash:
                raise ValueError('Source responses changed during re-evaluation; discard this incomplete audit.')
        write_json(output/'COMPLETE.json',{'run_id':run_id,'questions_per_arm':len(common),'arms':list(arms)})
        archive=output.with_suffix('.zip')
        if archive.exists():raise ValueError(f'Archive path already exists: {archive}')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(output.iterdir()):z.write(p,p.name)
        print('\n'.join(lines[:12]))
        print(f'\nReport: {output / "report.md"}\nUploadable results: {archive}')
        return output
    finally:
        data.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,help='Existing project, exported outcomes folder, or important_outcomes_gsm8k.zip')
    parser.add_argument('--output',type=Path,help='New directory outside the original outcomes; default is identified by input/protocol hashes')
    args=parser.parse_args()
    recheck(args.input or default_input(),args.output)


if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError,zipfile.BadZipFile) as exc:
        raise SystemExit(f'Re-evaluation stopped: {exc}')
