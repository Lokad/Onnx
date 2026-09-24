"""Reconcile all copy results and clocks before interpreting the component cost."""
from collections import defaultdict
from fractions import Fraction
import importlib.util
import json
from run import BASE, APP, ROOT, pin, read, write, prepared
from checks import resources


def main():
    prepared();assert not (BASE/'closed.json').exists()
    resource_rows=resources('capture');folder=BASE/'capture-collected';spec=read(BASE/'bundle/spec.json')
    state=read(folder/'capture-state.json');built=read(BASE/'build-collected/built.json')
    assert pin(folder/'built.json')==pin(BASE/'build-collected/built.json')
    assert pin(folder/'manifest.json')==built['manifest']
    assert pin(folder/'build-review.json')==pin(BASE/'build-review.json')
    assert read(BASE/'build-review.json')['passed'] and read(BASE/'build-review.json')['core_unchanged']
    manifest=read(folder/'manifest.json');cases=manifest['cases'];processes=[];all_clocks=[]
    assert [r['name'] for r in state['runs']]==[f'{i:02}-{m}' for i,m in enumerate(spec['order'])]
    loader=importlib.util.spec_from_file_location('cpu_accounting',APP/'collected/runtime/campaign_processes.py')
    accounting=importlib.util.module_from_spec(loader);loader.loader.exec_module(accounting)
    for index,run in enumerate(state['runs']):
        mode=spec['order'][index];output=folder/'results'/run['name'];result=read(output/'result.json')
        assert run['accounting']==accounting.foreign_fraction(run['cpu_before'],run['cpu_after'],state['supervisor']['pid'])
        assert run['accounting']['valid'] and run['accounting']['foreign_cpu_fraction']<=.01
        assert result['passed'] and result['mode']==mode and result['records']==1920 and result['measured']==1440 and result['warmup']==480
        assert result['runtime']=='.NET 10.0.8' and result['processor_count']==1 and result['affinity']==4 and result['flags']==[]
        assert result['core_sha256']==spec['core']['sha256'] and result['consumer_sha256']==built['consumer']['sha256']
        assert result['manifest_sha256']==built['manifest']['sha256'] and result['constant_sha256']==manifest['constant_sha256']
        assert all(result[k] for k in ['input_unchanged','held_outputs_unchanged','independent_outputs','checked_every_output_byte'])
        assert result['frequency']==1000000000 and 0<result['setup_start_ticks']<result['setup_end_ticks']
        clocks=read(output/'clocks.json');assert len(clocks)==1920
        durations=defaultdict(int);previous=result['setup_end_ticks'];copied=0
        for clock_index,row in enumerate(clocks):
            pass_index=clock_index//480;case_index=(clock_index%480)//24;copy=clock_index%24;c=cases[case_index]
            assert (row['pass'],row['case_index'],row['copy'])==(pass_index,case_index,copy)
            assert (row['name'],row['frames'],row['rows'],row['elements'])==(c['name'],c['frames'],2*c['frames']-1,c['bytes']//4)
            assert row['phase']==('warmup' if pass_index==0 else 'measured') and row['independent']
            assert row['output_sha256']==c['output_sha256'] and row['frequency']==result['frequency']
            assert row['thread_id']==result['thread_id'] and previous<=row['start_ticks']<row['end_ticks']
            previous=row['end_ticks'];all_clocks.append(dict(process=index,mode=mode,**row))
            if pass_index>0:durations[case_index]+=row['end_ticks']-row['start_ticks'];copied+=c['bytes']
        assert copied==3*524353536 and set(durations)==set(range(20))
        files=list(output.glob('case-*.json'));assert len(files)==80
        for pass_index in range(4):
            for case_index,c in enumerate(cases):
                value=read(output/f'case-{pass_index}-{case_index:02}.json')
                assert value['Name']==c['name'] and value['pass']==pass_index
                assert value['input_unchanged'] and value['held_outputs_unchanged']
                assert value['allocated_bytes']>=24*c['bytes']
        seconds={k:Fraction(v,3*result['frequency']) for k,v in durations.items()}
        processes.append(dict(index=index,mode=mode,corpus_seconds=float(sum(seconds.values())),
            cases=[dict(name=cases[k]['name'],seconds=float(v)) for k,v in seconds.items()],setup_seconds=result['setup_seconds']))
    controls=[];means={};case_means={}
    for mode in ['generic','helper']:
        pair=[r for r in processes if r['mode']==mode];assert len(pair)==2
        ratios=[('corpus',max(r['corpus_seconds'] for r in pair)/min(r['corpus_seconds'] for r in pair),spec['corpus_ratio_limit'])]
        means[mode]=sum(r['corpus_seconds'] for r in pair)/2
        for i,c in enumerate(cases):
            values=[r['cases'][i]['seconds'] for r in pair]
            ratios.append((c['name'],max(values)/min(values),spec['case_ratio_limit']))
            case_means[mode,i]=sum(values)/2
        controls.extend(dict(mode=mode,name=name,ratio=ratio,limit=limit,passed=ratio<=limit) for name,ratio,limit in ratios)
    difference=means['generic']-means['helper'];stable=all(r['passed'] for r in controls)
    useful=stable and difference>=spec['required_component_seconds']
    result=dict(passed=True,evidence_qualified=True,component_stable=stable,useful_component_estimate=useful,
        no_product_change=True,application_gain_claimed=False,core_unchanged=True,requests=320,calls=7680,measured_clocks=5760,
        copied_bytes_per_corpus=524353536,resources=resource_rows,processes=processes,controls=controls,
        corpus_seconds=means,component_difference_seconds=difference,required_component_seconds=spec['required_component_seconds'],
        interpretation='Standalone copying with allocation inside clocks and validation outside clocks; it is not a full-application speedup. Reshape includes an extra output view wrapper.',
        cases=[dict(name=c['name'],frames=c['frames'],generic_seconds=case_means['generic',i],helper_seconds=case_means['helper',i],
            difference_seconds=case_means['generic',i]-case_means['helper',i]) for i,c in enumerate(cases)])
    write(BASE/'clocks.json',all_clocks);write(BASE/'analysis.json',result)
    write(BASE/'closed.json',dict(passed=True,evidence_qualified=True,component_stable=stable,useful_component_estimate=useful,
        analysis=pin(BASE/'analysis.json'),clocks=pin(BASE/'clocks.json'),collection=pin(folder/'capture-collection.json'),
        transfer=pin(BASE/'capture-transfer.json'),build_review=pin(BASE/'build-review.json'),auditor=pin(__file__),
        terminal_owners=[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]))
    print(json.dumps(dict(evidence_qualified=True,stable=stable,useful_component_estimate=useful,corpus_seconds=means,
        difference_seconds=difference,required_seconds=spec['required_component_seconds'],
        failed_controls=[r for r in controls if not r['passed']],resources=resource_rows,closure=pin(BASE/'closed.json'))))


if __name__=='__main__':main()
