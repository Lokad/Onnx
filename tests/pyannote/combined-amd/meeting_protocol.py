"""Keep strict public-output compatibility separate from human-label accuracy."""
from pathlib import Path
import argparse
import json
import math
import sys
from candidate_protocol import pin, read, write


def number(v):
    return type(v) in (int, float) and math.isfinite(v)


def inspect_public(value, samples):
    assert set(value)=={'status','windows','audio_seconds','intervals','exclusive_intervals','speakers'}
    assert value['status'] in ('Completed','NoSpeech','NoUsableEmbeddings')
    assert type(value['windows']) is int and value['windows']==max(1,math.ceil((samples/16000-10)/1)+1)
    assert number(value['audio_seconds']) and value['audio_seconds']==samples/16000
    speakers=value['speakers'];assert isinstance(speakers,list)
    assert [s['speaker'] for s in speakers]==list(range(len(speakers)))
    for s in speakers:
        assert set(s)=={'speaker','centroid','has_embedding'} and type(s['speaker']) is int and type(s['has_embedding']) is bool
        assert len(s['centroid'])==256 and all(number(v) for v in s['centroid'])
        assert s['has_embedding'] or all(v==0 for v in s['centroid'])
    for key in ('intervals','exclusive_intervals'):
        rows=value[key];assert isinstance(rows,list)
        by_speaker={}
        for row in rows:
            assert isinstance(row,list) and len(row)==3 and all(number(v) for v in row)
            start,end,label=row
            assert 0<=start<end<=samples/16000 and int(label)==label and 0<=label<len(speakers)
            by_speaker.setdefault(label,[]).append((start,end))
        for group in by_speaker.values():
            ordered=sorted(group);assert all(a[1]<=b[0] for a,b in zip(ordered,ordered[1:]))
        if key=='exclusive_intervals':
            ordered=sorted(rows);assert all(a[1]<=b[0] for a,b in zip(ordered,ordered[1:]))
    assert {int(r[2]) for r in value['intervals']}==set(range(len(speakers)))
    assert (value['status']=='Completed') == bool(value['intervals'])


def compare(actual, expected):
    failures=[];maximum=0.
    def visit(a,e,path):
        nonlocal maximum
        if isinstance(e,dict):
            assert isinstance(a,dict) and set(a)==set(e)
            for key in e:visit(a[key],e[key],path+'/'+key)
        elif isinstance(e,list):
            if len(a)!=len(e):failures.append(dict(path=path,reason='length',actual=len(a),expected=len(e)));return
            for i,(x,y) in enumerate(zip(a,e)):visit(x,y,path+'/'+str(i))
        elif number(e) and number(a):
            error=abs(a-e)/max(1,abs(e))
            if '/centroid/' in path:
                maximum=max(maximum,error);ok=error<=1e-4
            else:ok=abs(a-e)<=(1e-12 if path.startswith(('/intervals/','/exclusive_intervals/')) else 0)
            if not ok:failures.append(dict(path=path,reason='value',actual=a,expected=e,scaled_error=error))
        elif type(a)!=type(e) or a!=e:failures.append(dict(path=path,reason='discrete',actual=a,expected=e))
    visit(actual,expected,'')
    return dict(passed=not failures,maximum_centroid_error=maximum,mismatch_count=len(failures),mismatches=failures)


def inspect_worker(value,manifest,engine):
    assert value['schema']==1 and value['engine']==engine and value['held_outputs_unchanged'] and value['affinity']==4
    assert [r['name'] for r in value['records']]==[c['name'] for c in manifest['cases']]
    for row,case in zip(value['records'],manifest['cases'],strict=True):
        assert row['ownership'] and row['input_sha256']==case['pcm_sha256']
        assert all(type(row[k]) is int for k in ['start_ticks','end_ticks','frequency'])
        assert row['frequency']>0 and row['end_ticks']>row['start_ticks'] and number(row['seconds'])
        assert row['seconds']==(row['end_ticks']-row['start_ticks'])/row['frequency']
        inspect_public(row['result'],case['samples'])
    if engine=='managed':
        assert value['core_sha256']==manifest['core_sha256'] and value['data_sha256']==manifest['data_sha256']
        assert value['runtime']=='.NET 10.0.8' and not value['flags']
    else:
        assert value['versions']==manifest['pins']['versions']
        assert value['native_settings']==dict(provider='CPUExecutionProvider',intra_threads=1,inter_threads=1,sequential=True,graph_optimizations='all',spinning=False)

