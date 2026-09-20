"""Independently verify and close completed evidence, including failed comparisons.

Run with the existing psutil/NumPy environment after collect.py and both audits.
No inference is performed and public disagreement is never a closure failure.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
import wave
import numpy as np
import psutil
from common import annotations, pin, read, select, write
from audit import compare
from audit_resources import inspect
from independent_score import score
from validation import worker, resources, damaged_worker_checks, damaged_resource_checks
from collect import HOST, KEY

COMPONENTS = ['reference_speaker_seconds', 'correct_speaker_seconds',
              'missed_speaker_seconds', 'false_alarm_speaker_seconds', 'confused_speaker_seconds']


def verify_inputs(base, manifest, dataset):
    preparation = read(base/'preparation.json')
    assert preparation['passed'] is True and read(base/'input-audit.json')['passed'] is True
    assert manifest['input_manifest_sha256'] == pin(base/'inputs/dataset.json')['sha256']
    for directory in ['inputs','originals']:
        for name,wanted in preparation[directory].items():assert pin(base/directory/name) == wanted
    for name,source in dataset['sources'].items():
        assert pin(base/'originals'/name) == {k:source[k] for k in ['bytes','sha256']}
    assert select((base/'originals/test.meetings.txt').read_text()) == [c['name'] for c in dataset['cases']]
    for case in dataset['cases']:
        name=case['name']
        rows=annotations((base/f'originals/{name}.rttm').read_text(),(base/f'originals/{name}.uem').read_text(),name)
        assert [list(row) for row in rows] == case['intervals']
        with wave.open(str(base/f'originals/{name}.Mix-Headset.wav'),'rb') as original, wave.open(str(base/'inputs'/case['path']),'rb') as crop:
            assert (crop.getnchannels(),crop.getsampwidth(),crop.getframerate(),crop.getnframes()) == (1,2,16000,9600000)
            data=crop.readframes(crop.getnframes());assert data == original.readframes(9600000)
        pcm=np.frombuffer(data,dtype='<i2').astype('<f4')/np.float32(32768)
        assert hashlib.sha256(pcm.tobytes()).hexdigest() == case['pcm_sha256']
        if name=='ES2004a':assert hashlib.sha256(pcm[:480000].tobytes()).hexdigest() == manifest['cases'][2]['pcm_sha256']


def terminal(base, collection):
    births={}
    for path in base.glob('process-*/identity.json'):
        if path.parent.name=='process-managed-run':continue  # These are Linux identities.
        value=read(path)
        for row in [value.get('supervisor'),value.get('child')]:
            if row:births[row['pid']]=row['birth']
        births.update({int(p):b for p,b in value.get('members',{}).items()})
    deployment=read(base/'deployment-native.json');births[deployment['pid']]=deployment['birth']
    for pid,birth in births.items():
        try:assert psutil.Process(pid).create_time()!=birth, ('Local owned process still live',pid,birth)
        except psutil.NoSuchProcess:pass
    script="""import sys,json,time
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
rows=json.loads(%r)
for row in rows:
 try:assert psutil.Process(row['pid']).create_time()!=row['birth'],('Remote owned process still live',row)
 except psutil.NoSuchProcess:pass
print(json.dumps(dict(checked_at=time.time(),terminal_processes=rows)))
""" % json.dumps(collection['terminal_processes'])
    output=subprocess.run(['ssh','-i',KEY,'-o','BatchMode=yes',HOST,'python3 -B -'],input=script,text=True,encoding='utf-8',capture_output=True,check=True)
    return dict(local=[dict(pid=p,birth=b) for p,b in sorted(births.items())],amd=json.loads(output.stdout))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True)
    args=parser.parse_args();base=args.artifact.resolve();root=base.parents[1]
    assert not (base/'closed.json').exists() and not (base/'verification.json').exists()
    frozen=read(base/'frozen.json');manifest=read(base/'manifest.json');dataset=read(base/'inputs/dataset.json')
    for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
    for name,wanted in frozen['native_files'].items():assert pin(Path(name))==wanted,name
    for section in ['models','native_assets','upstream','native_sources']:
        for item in manifest[section].values():assert pin(root/item['path'])=={k:item[k] for k in ['bytes','sha256']}
    verify_inputs(base,manifest,dataset)
    collection=read(base/'collected/collection.json');collection_check=read(base/'collection-check.json')
    assert collection_check['passed'] is True and collection_check['remote']['collection']==collection
    assert pin(base/'amd-results.tar.gz')==collection_check['remote']['archive']
    assert pin(base/'collected/collection.json')==collection_check['remote']['receipt']
    assert collection['all_owned_processes_terminal'] is True
    assert collection['frozen']==pin(base/'frozen.json') and collection['verified_reusable_files']==frozen['files']
    assert {p.relative_to(base/'collected').as_posix() for p in (base/'collected').rglob('*') if p.is_file()}==set(collection['files'])|{'collection.json'}
    for name,wanted in collection['files'].items():assert pin(base/'collected'/name)==wanted,name
    for path in (base/'process-managed-run').rglob('*'):
        if path.is_file():assert pin(path)==collection['files'][path.relative_to(base).as_posix()]
    assert {p.relative_to(base).as_posix() for p in (base/'process-managed-run').rglob('*') if p.is_file()}=={n for n in collection['files'] if n.startswith('process-managed-run/')}
    audit=read(base/'audit.json');resource_audit=read(base/'resource-audit.json')
    assert resource_audit['passed'] is True
    assert resource_audit['resources']==[inspect(base,engine,frozen) for engine in ['native','managed']]
    values={};refusals=[]
    for engine in ['native','managed']:
        directory=base/f'process-{engine}-run';value=read(directory/'worker/result.json')
        label='ort' if engine=='native' else engine
        worker(value,manifest,frozen,base,label)
        state=read(directory/'identity.json');samples=[json.loads(line) for line in (directory/'samples.jsonl').read_text().splitlines()]
        resources(state,samples,pin(base/'frozen.json')['sha256'],frozen['limits'],engine)
        refusals.append(dict(engine=engine,worker=damaged_worker_checks(value,manifest,frozen,base,label),
            resources=damaged_resource_checks(state,samples,pin(base/'frozen.json')['sha256'],frozen['limits'],engine)))
        values[label]=value
    assert audit['comparisons']==[dict(name=c['name'],**compare(m['result'],n['result'])) for c,m,n in zip(manifest['cases'],values['managed']['records'],values['ort']['records'],strict=True)]
    assert audit['public_comparison_passed'] == all(c['passed'] for c in audit['comparisons'])
    expected=[(c['name'],e,t) for c in dataset['cases'] for e in ['ort','managed'] for t in ['intervals','exclusive_intervals']]
    assert [(r['name'],r['engine'],r['timeline']) for r in audit['scores']]==expected
    score_checks=[]
    for row in audit['scores']:
        i=[c['name'] for c in dataset['cases']].index(row['name'])
        independent=score(dataset['cases'][i]['intervals'],values[row['engine']]['records'][i]['result'][row['timeline']],600)
        assert row['duration_seconds']==600 and row['collar_seconds']==0 and row['overlap_included'] is True
        for key in COMPONENTS+['diarization_error_rate']:assert math.isclose(row[key],independent[key],rel_tol=1e-11,abs_tol=1e-8),(row['name'],key)
        score_checks.append(dict(name=row['name'],engine=row['engine'],timeline=row['timeline'],independent=independent))
    assert [(r['engine'],r['timeline']) for r in audit['aggregates']]==[(e,t) for e in ['ort','managed'] for t in ['intervals','exclusive_intervals']]
    for row in audit['aggregates']:
        selected=[r for r in audit['scores'] if r['engine']==row['engine'] and r['timeline']==row['timeline']]
        for key in COMPONENTS:assert row[key]==math.fsum(r[key] for r in selected)
        assert row['diarization_error_rate']==sum(row[k] for k in COMPONENTS[2:])/row[COMPONENTS[0]]
    scorer_check=read(base/'scorer-validation.json');assert scorer_check['passed'] is True
    assert scorer_check['independent']==pin(Path(__file__).with_name('independent_score.py'))
    assert scorer_check['official']==pin(root/'tests/pyannote/accuracy/diarization_error.py')
    terminal_check=terminal(base,collection)
    write(base/'verification.json',dict(passed=True,refusals=refusals,independent_scores=score_checks,
        terminal=terminal_check,public_comparison_passed=audit['public_comparison_passed']))
    source=Path(__file__).parent
    snapshot=base/'postprocessing-source';snapshot.mkdir()
    sources={}
    for path in sorted(source.glob('*.py')):
        shutil.copyfile(path,snapshot/path.name)
        sources[path.relative_to(root).as_posix()]=pin(path)
        assert pin(snapshot/path.name)==pin(path)
    record=dict(schema=1,closed=True,execution_passed=True,public_comparison_passed=audit['public_comparison_passed'],
        closed_at=time.time(),source_commit=frozen['source_commit'],all_owned_processes_terminal=True,terminal=terminal_check,
        sources=sources,external_pins=frozen['native_files'],files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()})
    write(base/'closed.json',record)
    print('Closed',len(record['files']),'files; public compatibility',record['public_comparison_passed'],'receipt',pin(base/'closed.json')['sha256'])


if __name__=='__main__':main()
