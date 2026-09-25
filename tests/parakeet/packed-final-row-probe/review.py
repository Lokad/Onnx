"""Audit exact binaries, every planned case and resources; retain failed verdicts."""
import base64
import json
from pathlib import Path
import sys
from run import BASE, ROOT, TOOLS, PRELUDE, SOURCES, pin, read, write, ssh, prepared
from cases import MODES, cases


def collected(kind):
    prepared(); folder=BASE/(kind+'-collected');spec=read(BASE/'bundle/spec.json')
    receipt=read(folder/(kind+'-collection.json'));transfer=read(BASE/(kind+'-transfer.json'))
    assert transfer['passed'] and transfer['archive']==pin(BASE/(kind+'-results.tar.gz'))
    assert transfer['collection']==pin(folder/(kind+'-collection.json'))
    assert receipt['terminal'] and receipt['code']==0
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    assert pin(folder/'spec.json')==pin(BASE/'bundle/spec.json')
    for name,wanted in spec['files'].items():assert pin(folder/name)==wanted,name
    state=read(folder/(kind+'-state.json'))
    assert state['complete'] and state['code']==0 and receipt['state']==pin(folder/(kind+'-state.json'))
    assert state['supervisor']==read(BASE/(kind+'-deployment.json'))
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    expected=['sdk-version','consumer-restore','consumer-build'] if kind=='build' else MODES
    assert [r['name'] for r in state['runs']]==expected
    resources=[];limits=spec[kind+'_limits']
    for run in state['runs']:
        assert run['complete'] and run['code']==0 and run['seconds']<limits['seconds']
        assert run['preflight']['available']>=limits['available_before'] and run['preflight']['tmpfs']>=limits['tmpfs_before']
        samples=[json.loads(s) for s in (folder/'logs'/(run['name']+'.resources.jsonl')).read_text().splitlines()]
        assert len(samples)==run['samples']>0
        for sample in samples:
            assert sample['seconds']<limits['seconds'] and sample['rss']<limits['rss']
            assert min(sample['available'],sample['tmpfs'])>=spec['minimum_free'] and sample['output']<spec['output_limit']
            assert sample['rss']==sum(m['rss'] for m in sample['members'])
            for member in sample['members']:
                assert run['members'][str(member['pid'])]==member['birth'] and member['affinity']==[2]
                assert member['threads'] and all(t==[2] for t in member['threads'])
        gaps=[samples[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])]+[run['seconds']-samples[-1]['seconds']]
        assert all(0<=g<10 for g in gaps)
        resources.append(dict(name=run['name'],samples=len(samples),seconds=run['seconds'],peak_rss=max(s['rss'] for s in samples)))
    built=read(folder/'built.json');assert built['passed'] and not built['product_rebuilt']
    for name,wanted in built['runtime'].items():assert pin(folder/'runtime'/name)==wanted
    for name,wanted in spec['runtime_files'].items():assert built['runtime'][name]==wanted
    assert set(built['runtime'])==set(spec['runtime_files'])|{'PackedFinalRowProbe.dll','PackedFinalRowProbe.deps.json','PackedFinalRowProbe.runtimeconfig.json'}
    return folder,spec,receipt,state,built,resources


def build():
    assert not (BASE/'build-review.json').exists()
    folder,spec,receipt,state,built,resources=collected('build')
    assert (folder/'logs/sdk-version.stdout').read_text().strip()=='10.0.204'
    for name in ['consumer-restore','consumer-build']:
        output=(folder/'logs'/(name+'.stdout')).read_text()+(folder/'logs'/(name+'.stderr')).read_text()
        assert ': warning ' not in output and ': error ' not in output
    output=(folder/'logs/consumer-build.stdout').read_text()
    assert '0 Warning(s)' in output and '0 Error(s)' in output
    for name in SOURCES:
        assert pin(folder/'source'/(name+'.cs'))==pin(TOOLS/(name+'.cs.txt'))
    result=dict(passed=True,product_rebuilt=False,zero_added_warnings=True,built=pin(folder/'built.json'),
        collection=pin(folder/'build-collection.json'),transfer=pin(BASE/'build-transfer.json'),
        sources={n:pin(folder/'source'/(n+'.cs')) for n in SOURCES},
        product=spec['product'],resources=resources,release_admitted=False,reviewer=pin(Path(__file__)))
    write(BASE/'build-review.json',result)
    encoded=base64.b64encode((BASE/'build-review.json').read_bytes()).decode()
    transferred=ssh(PRELUDE+f'''
from remote import verify,live,read,pin
import base64
verify();state=read(base/'build-state.json')
assert state['complete'] and state['code']==0 and not live(state['supervisor'])
assert all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
assert pin(base/'built.json')=={result['built']!r}
with (base/'build-review.json').open('xb') as f:f.write(base64.b64decode({encoded!r}))
print(json.dumps(dict(passed=True,review=pin(base/'build-review.json'))))
''')
    assert transferred['review']==pin(BASE/'build-review.json')
    write(BASE/'build-review-transferred.json',transferred)
    print(json.dumps(dict(passed=True,review=transferred['review'],resources=resources)))


def capture():
    assert not (BASE/'closed.json').exists()
    folder,spec,receipt,state,built,resources=collected('capture')
    assert pin(folder/'build-review.json')==pin(BASE/'build-review.json')
    assert read(BASE/'build-review.json')['built']==pin(folder/'built.json')
    assert read(BASE/'build-collected/build-state.json')['ended']<state['started']
    assert not spec['prior_quantitative_attribution'] and not spec['release_admitted']
    modes={};failures=[];kernel_identities=None
    for mode in MODES:
        result=read(folder/'probe'/mode/'result.json')
        assert result['completed'] and result['mode']==mode and result['runtime']=='.NET 10.0.8'
        assert result['affinity']==4 and result['processor_count']==1
        assert all(result[name] for name in ['diagnostic_only','no_model_execution','no_application_score','no_product_change'])
        assert result['core_sha256']==spec['product']['Lokad.Onnx.dll']['sha256']
        assert result['runner_sha256']==built['runtime']['PackedFinalRowProbe.dll']['sha256']
        assert result['spec_sha256']==pin(folder/'spec.json')['sha256']
        flags={} if mode=='native' else {('DOTNET_EnableAVX512' if mode=='avx512-disabled' else 'DOTNET_EnableHWIntrinsic'):'0'}
        assert result['flags']==flags
        assert result['fma']==result['avx2']==(mode!='hardware-disabled')
        assert result['avx512']==(mode=='native')
        assert [r['name'] for r in result['kernels']]==['ShortWidePackPanelsB','ShortWideMultiply2Rows','ShortWideMultiply3Rows','ShortWideMultiplyRemainder']
        assert all(k['assembly']=='Lokad.Onnx' and k['token']>0 for k in result['kernels'])
        assert len({k['module'] for k in result['kernels']})==1
        if kernel_identities is None:kernel_identities=result['kernels']
        else:assert result['kernels']==kernel_identities
        expected=cases(mode);assert expected==spec['cases'][mode]
        assert [r['id'] for r in result['cases']]==[r['id'] for r in expected]
        assert {p.name for p in (folder/'probe'/mode).iterdir()}=={'result.json'}|{r['id']+'.json' for r in expected}
        for case,row in zip(expected,result['cases'],strict=True):
            assert read(folder/'probe'/mode/(row['id']+'.json'))==row and type(row['passed']) is bool
            if not row['passed']:
                assert row['error'];failures.append(dict(mode=mode,**row));continue
            evidence=row['evidence'];kind=case['kind']
            for name in ['n','k','m','special','option']:
                if name in case:assert evidence[name]==case[name]
            if kind in ['row','route','fallback']:
                assert evidence['bit_exact'] and evidence['inputs_immutable'] and len(evidence['output_sha256'])==64
            if kind=='row':
                assert evidence['allocated_bytes']==0 and evidence['nonzero_destination'] and evidence['guards_intact']
                assert len(evidence['packed_sha256'])==64
            elif kind=='route':
                remainder=case['m']%2!=0 and case['m']%3!=0
                assert evidence['public_matmul_exact'] and evidence['guards_intact']
                assert evidence['remainder']==remainder and evidence['new_row_calls']==int(remainder)
                assert evidence['reconstructed_bytes']==0 and not evidence['dense_operand_supplied_to_candidate']
            elif kind=='fallback':assert not evidence['packed_row_invoked']
            else:assert kind=='unavailable' and evidence['rejected'] and not evidence['packed_row_invoked']
        count=sum(not r['passed'] for r in result['cases'])
        assert result['failed']==count and result['passed']==(count==0)
        modes[mode]=result
    passed=not failures
    analysis=dict(passed=True,proof_passed=passed,diagnostic_only=True,release_admitted=False,
        no_product_change=True,no_application_score=True,no_model_execution=True,
        cases=sum(len(r['cases']) for r in modes.values()),modes=modes,failures=failures,resources=resources,
        product=spec['product'],consumer=built['runtime']['PackedFinalRowProbe.dll'],
        helper=pin(folder/'source/PackedFinalRowKernel.cs'),model_closure=spec['model_closure'],
        diagnostic_closure=spec['diagnostic_closure'],prior_quantitative_attribution=False,
        failed_release_controls=spec['failed_release_controls'],graph_integration_not_tested=True)
    assert analysis['cases']==98
    write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=True,proof_passed=passed,analysis=pin(BASE/'analysis.json'),
        collection=pin(folder/'capture-collection.json'),transfer=pin(BASE/'capture-transfer.json'),
        terminal_owners=receipt['identities'],reviewer=pin(Path(__file__)),
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),proof_passed=passed,cases=analysis['cases'],failures=failures,resources=resources),indent=2))


if __name__=='__main__':{'build':build,'capture':capture}[sys.argv[1]]()
