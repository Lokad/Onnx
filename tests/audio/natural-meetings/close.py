"""Close terminal natural-ASR evidence without hiding failed public comparisons."""
from pathlib import Path
import argparse
import copy
import importlib.metadata
import json
import math
import re
import shutil
import subprocess
import time
import wave
import hashlib
import numpy as np
import jiwer
from tokenizers import Tokenizer
from common import load,pin,read,write
from audit import worker,compare,original_functions
from audit_resources import inspect,local_terminal
from collect import HOST,KEY
from evidence import directory,failed_attempt,recovery_ready,profile


def refuses(action):
    try:action()
    except (AssertionError,ValueError,KeyError,TypeError,IndexError):return
    raise AssertionError('Damaged evidence accepted')


def damaged_worker(value,manifest,base,engine,family,validate):
    changes=[lambda v:v['records'].pop(),lambda v:v['records'].reverse(),
        lambda v:v.update(manifest_sha256='0'*64),lambda v:v.update(runner_sha256='0'*64),
        lambda v:v.update(affinity=1),lambda v:v.update(held_outputs_unchanged=1),
        lambda v:v['records'][0].update(input_sha256='0'*64),lambda v:v['records'][0].update(ownership=1),
        lambda v:v['records'][0].update(frequency=0),
        lambda v:v['records'][0].update(end_ticks=v['records'][0]['start_ticks']),
        lambda v:v['records'][0]['result'].update(text='damaged aggregate transcript'),
        lambda v:v['records'][0]['result'].update(processed_seconds=-1)]
    if engine=='managed':
        changes += [lambda v:v.update(core_sha256='0'*64),lambda v:v.update(data_sha256='0'*64),
            lambda v:v['records'][0].update(allocated_bytes=-1),lambda v:v['records'][0].update(gc_after=[-1,0,0])]
    else:
        changes += [lambda v:v.update(python_binary={}),lambda v:v.update(flags={}),
            lambda v:v['native_settings'].update(intra_threads=2),lambda v:v.update(versions={})]
    def check(v):
        worker(v,manifest,base,engine,family)
        for i,row in enumerate(v['records']):validate(row['result'],i)
    check(value)
    for change in changes:
        damaged=copy.deepcopy(value);change(damaged);refuses(lambda:check(damaged))
    return len(changes)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args()
    base=a.artifact.resolve();root=base.parents[1]
    assert not (base/'closed.json').exists() and not (base/'verification.json').exists()
    frozen=read(base/'frozen.json');manifest=read(base/'manifest.json');labels=read(base/'labels.json')
    for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
    for group in ['native_files','scorer_files']:
        for name,wanted in frozen[group].items():assert pin(Path(name))==wanted,name
    for name,version in frozen['scorer_versions'].items():assert importlib.metadata.version(name)==version
    for family in manifest['families'].values():
        for item in family['models'].values():assert pin(root/item['path'])=={k:item[k] for k in ['bytes','sha256']}
    checks={'labels.json':'labels','input-audit.json':'input_audit','inputs-check.json':'input_smokes',
        'reference-wrapper-check.json':'wrapper_check','retained-validator-check.json':'validator_check','audit-source-pins.json':'audit_source_pins'}
    for name,key in checks.items():assert pin(base/name)==frozen[key]
    for name in checks:
        if name not in ['labels.json','audit-source-pins.json']:assert read(base/name)['passed'] is True
    collection=read(base/'collected/collection.json');collected=read(base/'collection-check.json')
    assert collected['passed'] is True and collected['remote']['collection']==collection
    assert pin(base/'amd-results.tar.gz')==collected['remote']['archive']
    assert pin(base/'collected/collection.json')==collected['remote']['receipt']
    assert collection['all_owned_processes_terminal'] is True
    assert collection['frozen']==pin(base/'frozen.json') and collection['verified_reusable_files']==frozen['files']
    assert {p.relative_to(base/'collected').as_posix() for p in (base/'collected').rglob('*') if p.is_file()}==set(collection['files'])|{'collection.json'}
    for name,wanted in collection['files'].items():assert pin(base/'collected'/name)==wanted,name
    for family in manifest['schedule']:
        managed_directory=directory(base,'managed',family)
        expected={n for n in collection['files'] if n.startswith(managed_directory.name+'/')}
        assert {p.relative_to(base).as_posix() for p in managed_directory.rglob('*') if p.is_file()}==expected
        for name in expected:assert pin(base/name)==collection['files'][name]
    audit=read(base/'audit.json');resources=read(base/'resource-audit.json')
    assert resources['passed'] is True
    continuation=recovery_ready(base);failed=failed_attempt(base)
    assert resources['native_linux_continuation']==continuation
    assert resources['failed_native_attempt']==failed
    assert resources['resources']==[inspect(base,e,f,frozen) for e in ['native','managed'] for f in manifest['schedule']]
    normalizer=load('natural_asr_fixed_normalizer',base/'audit-source/common.py')
    assert pin(base/'audit-source/common.py')=={k:labels['normalizer'][k] for k in ['bytes','sha256']}
    scoring=original_functions(base/'audit-source/scoring.py',['total'],{})
    tokenizer=Tokenizer.from_file(str(root/manifest['families']['whisper']['model_directory']/'tokenizer.json'))
    whisper=original_functions(base/'audit-source/whisper_audit.py',['require','window_commit','validate_recording'],dict(math=math,BEGIN=50365,END=50257))
    parakeet=original_functions(base/'audit-source/parakeet_audit.py',['require','boundary','validate'],dict(np=np,re=re))
    vocab={}
    for line in (root/manifest['families']['parakeet']['model_directory']/'vocab.txt').read_text(encoding='utf-8').splitlines():
        piece,index=line.rsplit(' ',1);vocab[int(index)]=piece.replace('\u2581',' ')
    pcms=[]
    for case in manifest['cases']:
        item=case['audio'];path=root/item['path'];assert pin(path)=={k:item[k] for k in ['bytes','sha256']}
        with wave.open(str(path),'rb') as s:
            assert (s.getnchannels(),s.getsampwidth(),s.getframerate())==(1,2,16000)
            pcm=np.frombuffer(s.readframes(case['samples']),dtype='<i2').astype(np.float32)/np.float32(32768)
        assert hashlib.sha256(pcm.tobytes()).hexdigest()==case['pcm_sha256'];pcms.append(pcm)
    values={};refusals=[];comparisons=[];score_checks=[];complete=True
    expected_rows=[(f,e,c['name']) for f in manifest['schedule'] for e in ['ort','managed'] for c in manifest['cases'][:2]]
    assert [(r['family'],r['engine'],r['name']) for r in audit['scores']]==expected_rows
    for family in manifest['schedule']:
        values[family]={}
        def validate(result,i):
            if family=='whisper':whisper['validate_recording'](result,manifest['cases'][i],tokenizer)
            else:parakeet['validate'](result,manifest['cases'][i],pcms[i],vocab)
        for engine,folder in [('ort','native'),('managed','managed')]:
            value=read(directory(base,folder,family)/'worker/result.json');values[family][engine]=value
            refusals.append(dict(family=family,engine=engine,count=damaged_worker(value,manifest,base,engine,family,validate)))
            complete=complete and all(r['result']['stop_reason']=='Completed' for r in value['records'])
            for i,row in enumerate(value['records'][:2]):
                reference=normalizer.normalize(labels['cases'][i]['text']);hypothesis=normalizer.normalize(row['result']['text'])
                assert reference==labels['cases'][i]['normalized_reference']
                score=next(s for s in audit['scores'] if (s['family'],s['engine'],s['name'])==(family,engine,row['name']))
                words=jiwer.process_words(reference,hypothesis);chars=jiwer.process_characters(reference,hypothesis)
                expected=dict(reference=reference,hypothesis=hypothesis,reference_words=len(reference.split()),reference_characters=len(reference),
                    word_errors=words.substitutions+words.deletions+words.insertions,character_errors=chars.substitutions+chars.deletions+chars.insertions,
                    substitutions=words.substitutions,deletions=words.deletions,insertions=words.insertions,word_error_rate=words.wer,
                    character_error_rate=chars.cer,normalized_equal=reference==hypothesis)
                assert score==dict(family=family,engine=engine,name=row['name'],stop_reason=row['result']['stop_reason'],**expected)
                # Keep the exact word edit alignment beside every raw and normalized transcript.
                alignment=[dict(type=c.type,reference_start=c.ref_start_idx,reference_end=c.ref_end_idx,
                    hypothesis_start=c.hyp_start_idx,hypothesis_end=c.hyp_end_idx) for c in words.alignments[0]]
                score_checks.append(dict(family=family,engine=engine,name=row['name'],alignment=alignment))
        for i,case in enumerate(manifest['cases']):
            comparisons.append(dict(family=family,name=case['name'],**compare(values[family]['managed']['records'][i]['result'],values[family]['ort']['records'][i]['result'],family)))
    assert audit['comparisons']==comparisons and audit['all_requests_completed']==complete
    assert audit['public_comparison_passed']==all(r['passed'] for r in comparisons)
    repeated=None
    if failed:
        whisper['validate_recording'](failed['completed_record']['result'],manifest['cases'][0],tokenizer)
        repeated=compare(values['whisper']['ort']['records'][0]['result'],failed['completed_record']['result'],'whisper')
    assert audit['retry_repeat']==repeated
    assert audit['application_passed']==(complete and audit['public_comparison_passed'] and (repeated is None or repeated['passed']))
    assert audit['aggregates']==[dict(family=f,engine=e,**scoring['total']([r for r in audit['scores'] if r['family']==f and r['engine']==e])) for f in manifest['schedule'] for e in ['ort','managed']]
    local=local_terminal(base)
    selected,_,linux_frozen=profile(base,'native','whisper')
    remote_pins={} if selected==base else linux_frozen['native_files']
    remote_births=collection['terminal_processes']+([] if continuation is None else continuation['terminal_processes'])
    script="""import json,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
rows=json.loads(%r)
pins=json.loads(%r)
for name,wanted in pins.items():
 p=Path(name)
 with p.open('rb') as s:actual=dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(s,'sha256').hexdigest())
 assert actual==wanted,name
for row in rows:
 try:assert psutil.Process(row['pid']).create_time()!=row['birth'],row
 except psutil.NoSuchProcess:pass
print(json.dumps(dict(checked_at=time.time(),terminal_processes=rows,verified_native_files=pins)))
""" % (json.dumps(remote_births),json.dumps(remote_pins))
    response=subprocess.run(['ssh','-i',KEY,'-o','BatchMode=yes',HOST,'python3 -B -'],input=script,text=True,encoding='utf-8',capture_output=True,check=True)
    terminal=dict(local=local,amd=json.loads(response.stdout))
    write(base/'verification.json',dict(passed=True,refusals=refusals,score_checks=score_checks,terminal=terminal,application_passed=audit['application_passed']))
    snapshot=base/'postprocessing-source';snapshot.mkdir();sources={}
    for path in sorted(Path(__file__).resolve().parent.glob('*.py')):
        shutil.copyfile(path,snapshot/path.name);assert pin(path)==pin(snapshot/path.name)
        sources[path.relative_to(root).as_posix()]=pin(path)
    linux_source=Path(__file__).resolve().parent.parent/'natural-meetings-linux'
    (snapshot/'linux').mkdir()
    for path in sorted(linux_source.glob('*.py')):
        shutil.copyfile(path,snapshot/'linux'/path.name);assert pin(path)==pin(snapshot/'linux'/path.name)
        sources[path.relative_to(root).as_posix()]=pin(path)
    record=dict(schema=1,closed=True,execution_passed=True,application_passed=audit['application_passed'],closed_at=time.time(),
        source_commit=frozen['source_commit'],all_owned_processes_terminal=True,terminal=terminal,sources=sources,
        external_pins=dict(frozen['native_files'],**frozen['scorer_files']),remote_external_pins=remote_pins,
        files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()})
    write(base/'closed.json',record);print('Closed',len(record['files']),'files; application pass',record['application_passed'],'receipt',pin(base/'closed.json')['sha256'])


if __name__=='__main__':main()
