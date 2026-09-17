"""Ridge learns actual signed gaps on exactly the static kNN memory rows."""
from collections import Counter

import numpy as np
from sklearn.linear_model import Ridge

from gsm8k_experiment.common import atomic_json, digest, read_json, read_jsonl
from gsm8k_experiment.validation import fit_thresholds, options
from .inputs import require, selection, sha


class RidgeGap:
    def __init__(self, coef, intercept, memory):
        self.coef, self.intercept = np.asarray(coef,float),float(intercept)
        self.gaps, self.group_ids = memory.gaps,memory.group_ids
        self.encoder_identity = memory.encoder_identity
        self.label_definition = None

    def predict(self, embeddings, query_ids=None, encoder_identity=None):
        require(encoder_identity in (None,self.encoder_identity), 'Ridge proxy encoder differs')
        x = np.asarray(embeddings,np.float32)
        require(x.ndim == 2 and x.shape[1] == len(self.coef) and np.isfinite(x).all() and
                np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-4), 'Invalid ridge features')
        if query_ids is not None:
            require(not set(query_ids) & set(self.group_ids), 'Ridge evaluation/training queries overlap fitting memory')
        return x @ self.coef+self.intercept,np.full(len(x),np.nan),[[] for _ in x]


def fit(plan, folder, alphas):
    folder.mkdir(parents=True,exist_ok=True)
    marker = folder/'selection.json'
    dependency = digest({'inputs':plan['files'],'alphas':alphas})
    if marker.exists():
        chosen = read_json(marker)
        require(chosen['dependency'] == dependency and sha(folder/'ridge.npz') == chosen['model_sha256'], 'Ridge model or fitting inputs changed')
        with np.load(folder/'ridge.npz',allow_pickle=False) as z:
            model = RidgeGap(z['coef'],z['intercept'],plan['memory'])
        model.label_definition = chosen['common_label_definition']
        return model
    rows,x,y,coverage = selection(plan)
    counts = Counter(r['id'] for r in rows)
    weights = np.array([1/counts[r['id']] for r in rows])
    candidates,best = [],None
    memory = plan['memory']
    for alpha in alphas:
        reg = Ridge(alpha=alpha,fit_intercept=True,solver='cholesky').fit(np.asarray(memory.embeddings,float),memory.gaps)
        mse = float(np.average((reg.predict(x)-y)**2,weights=weights))
        candidates.append({'alpha':alpha,'validation_question_weighted_mse':mse})
        if best is None or (mse,-alpha) < best:
            best,model,selected = (mse,-alpha),RidgeGap(reg.coef_,reg.intercept_,memory),alpha
    path = folder/'ridge.npz'
    with path.with_suffix('.tmp').open('wb') as handle:
        np.savez_compressed(handle,coef=model.coef,intercept=model.intercept)
    path.with_suffix('.tmp').replace(path)
    # Reproduce the existing kNN validation target, then apply it to BOTH models.
    # Ridge's alpha never uses AUROC, and ridge does not select its own target.
    cal = [r for r in read_jsonl(plan['source']/'prepared/calibration_raw.jsonl')
           if all(r.get(k) is not None and np.isfinite(r[k]) for k in ('proxy_score','judge_score'))]
    cal_gaps = plan['norm'].gap([r['proxy_score'] for r in cal],[r['judge_score'] for r in cal])
    knn = memory.predict(x,[r['id'] for r in rows],memory.encoder_identity)[0]
    locked = fit_thresholds(cal_gaps,[{**r,'gap':float(g),'predicted_gap':float(h)} for r,g,h in zip(rows,y,knn)],options(plan['config']))
    model.label_definition = locked['label_definition']
    atomic_json(marker,{'dependency':dependency,'model_sha256':sha(path),'alpha':selected,'candidates':candidates,
                'memory_answers':len(memory.gaps),'memory_questions':len(set(memory.group_ids)),
                'validation':coverage,'test_used':False,'refit_on_validation':False,'new_judge_calls':0,
                'common_label_definition':model.label_definition,'label_selection':locked,
                'label_selection_method':'existing kNN validation rule; same target for ridge; no ridge-specific AUROC tuning',
                'target':'normalized proxy minus normalized 4B judge','weighting':'equal question weight on validation'})
    return model
