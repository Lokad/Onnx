"""Prospective coverage, application checks and finite limits for AMD audio."""
import hashlib, json, math
from pathlib import Path

FAMILIES=('parakeet','pyannote','whisper')
VERSIONS=dict(numpy='2.2.4',onnxruntime='1.29.0')
LIMITS=dict(seconds=3600,rss=14*1024**3,available=1024**3,preflight=13*1024**3,
            disk=32*1024**2,preflight_disk=64*1024**2,campaign_seconds=14400)


def pin(path):
    path=Path(path)
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def check_result(actual,expected,path='',family=None):
    if isinstance(expected,dict):
        assert isinstance(actual,dict) and set(actual)==set(expected),path
        return max([check_result(actual[k],v,path+'/'+k,family) for k,v in expected.items()]+[0.])
    if isinstance(expected,list):
        assert isinstance(actual,list) and len(actual)==len(expected),path
        return max([check_result(a,e,path+'/'+str(i),family) for i,(a,e) in enumerate(zip(actual,expected))]+[0.])
    if type(expected) in (int,float):
        assert type(actual) in (int,float) and math.isfinite(actual) and math.isfinite(expected),path
        if family in ('parakeet','whisper') and type(expected) is int:assert type(actual) is int,path
        if '/centroid/' in path:
            error=abs(actual-expected)/max(1,abs(expected));assert error<=1e-4,path;return error
        tolerance=1e-12 if path.startswith(('/intervals/','/exclusive_intervals/')) else 0
        assert abs(actual-expected)<=tolerance,path
    else:assert type(actual) is type(expected) and actual==expected,path
    return 0.


def schedule(mode):
    assert mode in ('conformance','timing')
    return [(family,engine) for family in FAMILIES for engine in (('ort','managed') if mode=='conformance' else ('managed','ort','ort','managed'))]


def check_sample(sample):
    assert 0<=sample['seconds']<LIMITS['seconds']
    assert sample['available']>=LIMITS['available'] and sample['disk']>=LIMITS['disk']
    assert sample['members'] and sum(m['rss'] for m in sample['members'])<LIMITS['rss']
    assert all(m['affinity']==[2] and m['rss']>=0 for m in sample['members'])


def validate_records(value,manifest,mode):
    assert value['schema']==1 and value['family']==manifest['family'] and value['engine'] in ('managed','ort')
    assert value['conformance']==(mode=='conformance') and value['held_outputs_unchanged'] and value['affinity']==4
    assert math.isfinite(value['setup_seconds']) and value['setup_seconds']>=0
    assert not any(k.lower().startswith(('lokad_','dotnet_','complus_')) for k in value['flags'])
    wanted=[(iteration,case) for iteration in range(1 if mode=='conformance' else 4) for case in manifest['cases']]
    assert len(value['records'])==len(wanted);previous=0;first={}
    for row,(iteration,case) in zip(value['records'],wanted):
        assert row['name']==case['name'] and row['pass']==iteration and row['phase']==('warmup' if iteration==0 else 'measured')
        assert row['ownership'] and row['input_sha256']==case['raw_sha256']
        assert all(type(row[k]) is int for k in ['start_ticks','end_ticks','frequency']) and row['frequency']>0
        assert previous<=row['start_ticks']<row['end_ticks'];previous=row['end_ticks']
        assert math.isfinite(row['seconds']) and row['seconds']>0 and math.isclose(row['seconds'],(row['end_ticks']-row['start_ticks'])/row['frequency'],rel_tol=1e-14)
        error=check_result(row['result'],case['expected'],family=manifest['family'])
        assert math.isclose(row['maximum_centroid_error'],error,rel_tol=1e-12,abs_tol=1e-15)
        if row['name'] in first:assert row['result']==first[row['name']]
        else:first[row['name']]=row['result']
        if value['engine']=='ort' and manifest['family']=='whisper':
            feature=row['frontend'];assert feature['values']==384000 and feature['failed']==0 and 0<=feature['max_abs']<=1e-5
            assert len(feature['sha256'])==64 and type(feature['bits_equal']) is bool

