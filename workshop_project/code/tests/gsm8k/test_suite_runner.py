from gsm8k_experiment.common import DEFAULT_CONFIG, OUTPUT_ROOT
import copy
import json
import zipfile
from pathlib import Path
import pytest
from gsm8k_experiment import suite
from gsm8k_experiment.common import ROOT, atomic_json, load_config, read_json
from gsm8k_experiment.export import export_results


def test_sequential_seeds_share_revisions_and_partition_but_keep_training_seed(tmp_path, monkeypatch):
    c=load_config(DEFAULT_CONFIG);configpath=tmp_path/'input.json';atomic_json(configpath,c)
    output=tmp_path/'seeds';seen=[]
    resolved={**{k:'a'*40 for k in c['models']},'dataset':c['revisions']['dataset']}
    def child(command,**kwargs):
        cfg=read_json(Path(command[command.index('--config')+1]));dest=Path(command[command.index('--output')+1])
        seen.append(cfg['seed']);assert cfg['data_seed']==42
        if cfg['seed']==42: atomic_json(dest/'resolved_assets.json',resolved)
        else: assert read_json(dest/'resolved_assets.json')==resolved
        atomic_json(dest/'final_protocol.json',{'arms':cfg['arms'],'updates':400})
        metrics=[{'arm':a,'cohort':'final','accuracy':.3,'numeric_accuracy':.4,'numeric_unresolved_rate':.1,
                  'format_valid_rate':.8,'length_cap_rate':.02} for a in ['base',*cfg['arms']]]
        atomic_json(dest/'summary.json',{'metrics':metrics})
        return 0
    monkeypatch.setattr(suite.subprocess,'call',child)
    suite.main(['--config',str(configpath),'--output',str(output),'--seeds','42','43','44'])
    assert seen==[42,43,44]
    data=read_json(output/'suite_summary.json')
    assert data['paired_teacher_differences']['numeric_difference']['values']==[0.,0.,0.]
    assert len(data['arms'])==5
    assert read_json(output/'suite_status.json')['stage']=='complete'
    with pytest.raises(ValueError,match='seeds or configuration changed'):
        suite.main(['--config',str(configpath),'--output',str(output),'--seeds','42','43'])


def test_suite_stops_after_failed_seed(tmp_path,monkeypatch):
    calls=[]
    def failed(command,**kw): calls.append(command);return 7
    monkeypatch.setattr(suite.subprocess,'call',failed)
    with pytest.raises(SystemExit,match='Seed 42 stopped'):
        suite.main(['--output',str(tmp_path/'seeds'),'--seeds','42','43'])
    assert len(calls)==1
    assert read_json(tmp_path/'seeds'/'suite_status.json')['stage']=='failed'


def test_export_contains_new_memories_and_excludes_weights(tmp_path):
    output=tmp_path/'seeds';atomic_json(output/'suite_protocol.json',{'seeds':[42]})
    root=output/'seed_42';atomic_json(root/'config.json',load_config(DEFAULT_CONFIG))
    (root/'prepared_30b').mkdir();(root/'prepared_30b'/'memory_initial.npz').write_bytes(b'array fixture')
    (root/'checkpoint.pt').write_bytes(b'not for export')
    archive=export_results(output,tmp_path/'outcomes.zip')
    with zipfile.ZipFile(archive) as z:
        names=set(z.namelist())
        assert 'outcomes/seed_42/prepared_30b/memory_initial.npz' in names
        assert 'code/gsm8k_experiment/teacher_memory.py' in names
        assert not any(x.endswith('.pt') for x in names)
    with pytest.raises(ValueError,match='already exists'):export_results(output,archive)
