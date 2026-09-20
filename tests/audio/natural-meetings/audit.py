"""Preserve application disagreements separately from complete human word errors."""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.metadata
import math
import re
import wave
import numpy as np
from common import NAMES,load,pin,read,write


def original_functions(path,names,scope):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope);return scope


def compare(actual,expected,family):
    failures=[];confidence=[]
    def visit(a,e,path):
        if isinstance(e,dict):
            if not isinstance(a,dict) or set(a)!=set(e):failures.append(dict(path=path,reason='keys'));return
            for key in e:visit(a[key],e[key],path+'/'+key)
        elif isinstance(e,list):
            if not isinstance(a,list) or len(a)!=len(e):failures.append(dict(path=path,reason='length',actual=len(a) if isinstance(a,list) else None,expected=len(e)));return
            for i,(x,y) in enumerate(zip(a,e)):visit(x,y,path+'/'+str(i))
        elif family=='whisper' and path.rsplit('/',1)[-1] in ['no_speech_probability','average_log_probability']:
            if a is None or e is None:
                if a is not e:failures.append(dict(path=path,reason='missing confidence',actual=a,expected=e))
            else:
                assert math.isfinite(a) and math.isfinite(e);confidence.append(dict(path=path,absolute_difference=abs(a-e)))
        elif a!=e or isinstance(a,bool)!=isinstance(e,bool):failures.append(dict(path=path,reason='value',actual=a,expected=e))
    visit(actual,expected,'')
    return dict(passed=not failures,mismatch_count=len(failures),mismatches=failures,
        maximum_observed_confidence_difference=max((r['absolute_difference'] for r in confidence),default=0.),confidence_differences=confidence)


def worker(value,manifest,base,engine,family):
    assert value['schema']==1 and value['engine']==engine and value['family']==family
    assert value['affinity']==4 and value['held_outputs_unchanged'] is True
    assert value['manifest_sha256']==pin(base/'manifest.json')['sha256']
    assert [r['name'] for r in value['records']]==NAMES
    for row,case in zip(value['records'],manifest['cases'],strict=True):
        assert row['ownership'] is True and row['input_sha256']==case['pcm_sha256']
        assert all(type(row[k]) is int for k in ['start_ticks','end_ticks','frequency'])
        assert row['frequency']>0 and row['end_ticks']>row['start_ticks']
        assert math.isfinite(row['seconds']) and row['seconds']==(row['end_ticks']-row['start_ticks'])/row['frequency']
    if engine=='managed':
        assert value['core_sha256']==manifest['core_sha256'] and value['data_sha256']==manifest['data_sha256']
        assert value['runner_sha256']==pin(base/'bin/NaturalAsr.dll')['sha256']
        assert value['runtime']=='.NET 10.0.8' and value['flags']==[]
        for row in value['records']:
            assert type(row['allocated_bytes']) is int and row['allocated_bytes']>=0
            assert len(row['gc_before'])==len(row['gc_after'])==3
            assert all(type(a) is int and type(b) is int and 0<=a<=b for a,b in zip(row['gc_before'],row['gc_after']))
    else:
        assert value['runner_sha256']==pin(base/'runtime/native.py')['sha256'] and value['versions']==manifest['versions']
        assert value['python_binary']==manifest['native_files']['C:/Python313/python.exe']
        assert value['flags']==dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        assert value['native_settings']==dict(provider='CPUExecutionProvider',intra_threads=1,inter_threads=1,sequential=True,graph_optimizations='all',spinning=False)
        nonsilent=[w for row in value['records'] for w in row['result']['windows'] if w['decoding']['stop_reason']!='SilentInput']
        if family=='parakeet':assert value['reference_checks']['original_decoder_crosschecks']==len(nonsilent)
        else:
            assert value['reference_checks']['checked_arrays']==sum(2+len(w['decoding']['token_ids']) for w in nonsilent)
            assert value['reference_checks']['borrowed_function_body_sha256']==read(base/'reference-wrapper-check.json')['borrowed_function_body_sha256']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    base=a.artifact.resolve();root=base.parents[1];manifest=read(base/'manifest.json');labels=read(base/'labels.json');frozen=read(base/'frozen.json')
    assert pin(base/'labels.json')==frozen['labels']==manifest['labels']
    for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
    for name,wanted in frozen['scorer_files'].items():assert pin(Path(name))==wanted,name
    for name,version in frozen['scorer_versions'].items():assert importlib.metadata.version(name)==version,name
    # Keep this file's common module distinct from the saved scorer's normalizer.
    normalizer=load('fixed_meeting_normalizer',base/'audit-source/common.py')
    import jiwer
    scoring=original_functions(base/'audit-source/scoring.py',['distance','metrics','total'],dict(jiwer=jiwer,normalize=normalizer.normalize))
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(root/manifest['families']['whisper']['model_directory']/'tokenizer.json'))
    whisper=original_functions(base/'audit-source/whisper_audit.py',['require','window_commit','validate_recording'],dict(math=math,BEGIN=50365,END=50257))
    parakeet=original_functions(base/'audit-source/parakeet_audit.py',['require','boundary','validate'],dict(np=np,re=re))
    vocab={}
    for line in (root/manifest['families']['parakeet']['model_directory']/'vocab.txt').read_text(encoding='utf-8').splitlines():
        piece,index=line.rsplit(' ',1);vocab[int(index)]=piece.replace('\u2581',' ')
    pcms=[]
    for case in manifest['cases']:
        path=root/case['audio']['path'];assert pin(path)=={k:case['audio'][k] for k in ['bytes','sha256']}
        with wave.open(str(path),'rb') as source:pcm=np.frombuffer(source.readframes(case['samples']),dtype='<i2').astype(np.float32)/np.float32(32768)
        assert hashlib.sha256(pcm.tobytes()).hexdigest()==case['pcm_sha256'];pcms.append(pcm)
    comparisons=[];scores=[];values={};cache={};complete=True
    for family in manifest['schedule']:
        values[family]={}
        for engine,folder in [('ort','native'),('managed','managed')]:
            value=read(base/f'process-{folder}-{family}-run/worker/result.json');worker(value,manifest,base,engine,family);values[family][engine]=value
            for i,(case,row) in enumerate(zip(manifest['cases'],value['records'],strict=True)):
                result=row['result']
                if family=='whisper':whisper['validate_recording'](result,case,tokenizer)
                else:parakeet['validate'](result,case,pcms[i],vocab)
                complete=complete and result['stop_reason']=='Completed'
                if i==2:continue
                key=(labels['cases'][i]['text'],result['text'])
                if key not in cache:cache[key]=scoring['metrics'](*key)
                assert cache[key]['reference']==labels['cases'][i]['normalized_reference']
                scores.append(dict(family=family,engine=engine,name=case['name'],stop_reason=result['stop_reason'],**cache[key]))
        for i,case in enumerate(manifest['cases']):
            comparisons.append(dict(family=family,name=case['name'],**compare(values[family]['managed']['records'][i]['result'],values[family]['ort']['records'][i]['result'],family)))
    aggregates=[dict(family=f,engine=e,**scoring['total']([r for r in scores if r['family']==f and r['engine']==e])) for f in manifest['schedule'] for e in ['ort','managed']]
    output=dict(all_requests_completed=complete,public_comparison_passed=all(r['passed'] for r in comparisons),
        application_passed=complete and all(r['passed'] for r in comparisons),comparisons=comparisons,scores=scores,aggregates=aggregates)
    write(a.output,output);print('Audited',len(comparisons),'comparisons and',len(scores),'human-score rows; application pass',output['application_passed'])


if __name__=='__main__':main()
