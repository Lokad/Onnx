from pathlib import Path
import sys
REFTOOLS=Path(__file__).resolve().parents[1]/'filterbank-reference'
sys.path.insert(0,str(REFTOOLS))
from common import *
from generate import VARIANTS, SOURCE_SHA

LIMITS=dict(seconds=600,rss=2*1024**3,available=1024**3,preflight=4*1024**3,disk=8*1024**3)
CORE=ROOT/'artifacts/e5-reduction-blocks-local-v3-20260920/bin/Lokad.Onnx.dll'
CORE_SHA='8b991fd7baaa470c45285754b20696c463dedc890a7db23dd4f0b9c7c818ccf1'
PRIORS={
 'wespeaker-full-reference-20260920':'25d45f0f2cc3c70b442fb0b1dd6ab4be4420dcf1ce0667a4c232abf83f981ce8',
 'wespeaker-window-reference-20260920':'3e5130b1a76a626d12c7108ffcab0a9b01bcbb8e72de26e50558cd6c57fc7bed',
 'wespeaker-coefficients-20260920':'e506c01ae586226900421ff4e3415b8c240f2fdc45e084f65a467b5380b9802f'}


def shapes(count):
    assert isinstance(count,int) and 400<=count<=480000
    frames=1+(count-400)//160
    return {stage:([1,frames,80] if stage=='features' else [frames,512 if stage=='windowed' else 257 if stage in ['real','imaginary','power'] else 80]) for stage in STAGES}


def load_baseline(row):
    path=ROOT/row['file'];assert pin(path)==row['pin']
    value=np.load(path,allow_pickle=False) if row['format']=='npy' else np.fromfile(path,dtype='<f4').reshape(row['shape'])
    assert value.dtype==np.float32 and list(value.shape)==row['shape'] and np.isfinite(value).all()
    return value


def check_array(value,shape):
    assert value.dtype==np.float64 and list(value.shape)==shape and np.isfinite(value).all()


def comparison(actual,expected,limit=ORIGINAL_LIMIT):
    row=metric(actual,expected,limit);delta=np.asarray(actual,dtype=np.float64)-expected
    row['squared_error']=float(np.sum(delta*delta));return row


def aggregate(rows):
    assert rows
    return dict(arrays=len(rows),values=sum(r['values'] for r in rows),failed=sum(r['failed'] for r in rows),
                max_scaled=max(r['max_scaled'] for r in rows),squared_error=sum(r['squared_error'] for r in rows))


def resource_checks(state, samples):
    assert state['complete'] and state['code']==0 and not state.get('error') and state['limits']==LIMITS
    assert [r['name'] for r in state['runs']]==['build','managed','numpy','torch']
    births=[state['supervisor']];summaries=[];ended=0
    for run in state['runs']:
        assert run['complete'] and run['code']==0 and not run.get('error') and 0<run['seconds']<600
        assert ended<=run['started']<=run['ended'];ended=run['ended']
        assert run['preflight_available']>=LIMITS['preflight'] and run['preflight_disk']>=LIMITS['disk']
        assert run['members'][str(run['child']['pid'])]==run['child']['birth']
        births += [dict(pid=int(pid),birth=birth) for pid,birth in run['members'].items()]
        rows=samples[run['name']];assert rows and len(rows)==run['samples'];prior=0
        for row in rows:
            assert prior<=row['seconds']<=run['seconds'] and row['seconds']-prior<10;prior=row['seconds']
            assert row['available']>=LIMITS['available'] and sum(m['rss'] for m in row['members'])<LIMITS['rss']
            assert len({m['pid'] for m in row['members']})==len(row['members'])
            for m in row['members']:assert m['rss']>=0 and m['affinity']==[0] and run['members'][str(m['pid'])]==m['birth']
        assert run['seconds']-prior<10
        summaries.append(dict(name=run['name'],seconds=run['seconds'],samples=len(rows),peak_group_rss=max(sum(m['rss'] for m in row['members']) for row in rows)))
    return births,summaries
