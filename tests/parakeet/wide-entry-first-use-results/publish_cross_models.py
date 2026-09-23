"""Publish complete closed cross-model results and independently verify monitor gaps."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def closed(kind):
    base=ROOT/f'artifacts/parakeet-wide-entry-first-use-{kind}-amd-20260923'
    proof=read(base/'closed.json');assert proof['passed']
    for name,wanted in proof['files'].items():assert pin(base/name)==wanted,name
    a=read(base/'analysis.json');assert proof['analysis']==pin(base/'analysis.json') and a['passed']
    gaps=[]
    for row in read(base/'collected/identity.json')['runs']:
        samples=[json.loads(s) for s in (base/'collected/logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples']>0
        intervals=[samples[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])]+[row['seconds']-samples[-1]['seconds']]
        assert all(0<=value<10 for value in intervals)
        gaps.append(dict(name=row['name'],maximum_seconds=max(intervals)))
    return dict(closure=pin(base/'closed.json'),payload=pin(base/'payload.json'),monitoring_gaps=gaps,**a)


def main():
    targets=[OUT/'cross-models-20260923.md',OUT/'cross-models-observations-20260923.json']
    assert not any(p.exists() for p in targets)
    shared=closed('shared');pyannote=closed('pyannote')
    assert shared['identities']==pyannote['identities']
    for role in ['selected','candidate']:
        assert sum(shared['results'][role+'-'+m]['arrays'] for m in ['shared','e5'])==166
        assert sum(shared['results'][role+'-'+m]['values'] for m in ['shared','e5'])==5000814
        p=pyannote['results'][role]
        assert (p['arrays'],p['values'],p['public_calls'])==(18,2917107,16)
    assert pyannote['results']['candidate']['complete_public_results_exact']
    assert all(r['exact_selected'] for mode in ['shared','e5'] for r in shared['results']['candidate-'+mode]['rows'])
    assert all(r['bit_identical'] for r in pyannote['results']['candidate']['comparisons'] if r['reference']=='production')
    maximum_shared=max(r['maximum'] for role in ['selected','candidate'] for mode in ['shared','e5'] for r in shared['results'][role+'-'+mode]['rows'])
    maximum_pyannote=max(r['maximum'] for role in ['selected','candidate'] for r in pyannote['results'][role]['comparisons'] if r['reference']=='native')
    assert max(maximum_shared,maximum_pyannote)<=1e-4
    maintenance=ROOT/'artifacts/parakeet-wide-entry-pyannote-preflight-maintenance-20260923'
    m=read(maintenance/'closed.json');assert m['passed'] and m['resumed']
    assert m['prospective']==pin(maintenance/'prospective.json') and m['paused']==pin(maintenance/'paused.json')
    observations=dict(shared=shared,pyannote=pyannote,maintenance=dict(receipt=pin(maintenance/'closed.json'),**m))
    targets[1].write_text(json.dumps(observations,indent=2,allow_nan=False)+'\n',encoding='utf8')
    lines=['# Wide-entry candidate: complete shared/e5 and Pyannote correctness','',
        'Both products pass all complete numerical and public checks. Every candidate',
        'tensor and public result matches the fresh selected product exactly.',
        'These correctness results do not establish graph or Pyannote application performance.','',
        '| Scope per product | Arrays | Values | Public calls | Maximum native scaled error |',
        '| --- | ---: | ---: | ---: | ---: |',
        f'| Shared graphs and five e5 inputs | 166 | 5,000,814 | — | {maximum_shared:.17g} |',
        f'| Complete Pyannote graphs and public diarization | 18 | 2,917,107 | 16 | {maximum_pyannote:.17g} |','',
        'Both roles independently retain the original native error bound of0.0001.',
        'Shared/e5 includes repeated contexts, facade use, memory policy, unchanged',
        'inputs and held outputs. Pyannote includes complete segmentation and embedding',
        'arrays, all public speaker assignments, timelines, statuses and centroid values.','',
        'The shared Replay consumer is reused unchanged. The rebuilt Pyannote consumer',
        'changes only its expected Data-assembly hash:95 methods remain exact and one',
        'method differs by that literal only; the public surface is unchanged.','']
    for label,a in [('Shared/e5',shared),('Pyannote',pyannote)]:
        lines.append(f"{label}: {len(a['resources'])} completed jobs, {sum(r['samples'] for r in a['resources'])} resource samples; peak RSS {max(r['peak_rss'] for r in a['resources']):,}bytes. All monitoring gaps are below10seconds.")
    lines+=['','All recorded owners are terminal. Pyannote waited at its unchanged12GiB',
        'preflight between completed workers. With no inference child, the exact',
        'supervisor was suspended at that boundary;1,822 duplicate files in closed',
        'campaigns were linked after verifying14,298 hashes, freeing219,025,408bytes',
        'of tmpfs. The active payload remained exact and the same supervisor resumed.',
        'No inference was stopped or repeated, and no resource bound changed.','',
        'Actual products: selected Core521bae17/Dataf3b9aa81, candidate Core672e5f30/Data065b7a7f.',
        'AMD CPU2 computes and CPU0 monitors, .NET10.0.8, ordinary runtime flags.',
        'Root source and BENCHMARK.md remain unchanged pending performance and release checks.','',
        '[Complete results, identities, resources and monitoring gaps](cross-models-observations-20260923.json).','',
        'Shared/e5 closure: `'+shared['closure']['sha256']+'`.',
        'Pyannote closure: `'+pyannote['closure']['sha256']+'`.']
    targets[0].write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(passed=True,shared_maximum=maximum_shared,pyannote_maximum=maximum_pyannote,
                         reports={p.name:pin(p) for p in targets})))


if __name__=='__main__':main()
