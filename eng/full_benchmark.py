"""Fixed seven-model current-package/ORT run, including validation, within one hour."""
import datetime as dt
import json,math,os,shutil,statistics,time
from pathlib import Path

import audio_worker as audio
import benchmark_worker as common
import campaign_evidence as evidence
import gemma_worker as gemma
import graph_worker as graph

CASES=('pyannote','parakeet','e5-30tok','gemma2-128tok','dinov3','resnet50','gpt2')
PAIR_COUNT=6
T_CRITICAL=2.570581835636305
PROTOCOL='full-seven-models-v1'


def jobs():
    result=[]
    for case in CASES:
        for engine in ('lok','ort'):result.append(dict(case=case,engine=engine,phase='conformance',pair=0))
    for case in CASES:
        for pair in range(1,(PAIR_COUNT if case in ('pyannote','parakeet') else 2)+1):
            for engine in (('lok','ort') if pair%2 else ('ort','lok')):
                result.append(dict(case=case,engine=engine,phase='timing',pair=pair))
    for index,item in enumerate(result,1):item['index']=index
    return result


def remaining(deadline,clock=None):
    value=deadline-(time.monotonic() if clock is None else clock())
    evidence.require(value>0,'full benchmark duration limit exceeded')
    return value


def summarize(workers):
    expected=jobs()
    evidence.require(len(workers)==len(expected),'full benchmark incomplete')
    for item,wanted in zip(workers,expected,strict=True):
        evidence.require(all(item[k]==wanted[k] for k in wanted),'full benchmark schedule changed')
        values=item['samples_ms']
        if wanted['phase']=='conformance':
            evidence.require(len(values)==(2 if wanted['case']=='gemma2-128tok' else 0),'full conformance sample count changed')
            evidence.require(all(math.isfinite(v) and v>0 for v in values),'invalid full conformance clocks')
            continue
        evidence.require(values and all(math.isfinite(v) and v>0 for v in values),'invalid full benchmark samples')
        count=3 if wanted['case'] in ('pyannote','parakeet') else 33 if wanted['case']=='gemma2-128tok' else 180
        evidence.require(len(values)==count,'full benchmark measured count changed')
    summary={}
    for case in CASES:
        group=[w for w in workers if w['case']==case and w['phase']=='timing']
        aggregation=statistics.median if case=='gemma2-128tok' else statistics.mean
        scores={(w['pair'],w['engine']):aggregation(w['samples_ms']) for w in group}
        pairs=range(1,(PAIR_COUNT if case in ('pyannote','parakeet') else 2)+1)
        lok=[scores[p,'lok'] for p in pairs];ort=[scores[p,'ort'] for p in pairs]
        if case in ('pyannote','parakeet'):
            ratios=[a/b for a,b in zip(lok,ort,strict=True)]
            logs=[math.log(r) for r in ratios];center=statistics.mean(logs)
            margin=T_CRITICAL*statistics.stdev(logs)/math.sqrt(PAIR_COUNT)
            ratio=math.exp(center);interval=[math.exp(center-margin),math.exp(center+margin)]
            summary[case]=dict(lok_ms=statistics.mean(lok),ort_ms=statistics.mean(ort),ratio=ratio,
                interval_95=interval,df=5,t_critical=T_CRITICAL,paired_ratios=ratios,
                target_pass=ratio<1.03 and interval[1]<1.03,
                aggregation='process mean; geometric paired ratio; two-sided log-Student-t95%')
        else:
            a,b=statistics.median(lok),statistics.median(ort)
            summary[case]=dict(lok_ms=a,ort_ms=b,ratio=a/b,aggregation='median of two process scores per engine')
        summary[case].update(lok_process_ms=lok,ort_process_ms=ort)
    return summary


def load_fixtures(path,build,deadline):
    config=json.loads(path.read_text(encoding='utf-8-sig'))
    evidence.require(config['schema']==1 and set(config['audio'])=={'parakeet','pyannote'},'invalid full fixture manifest')
    def binding(item):
        target=(path.parent/item['path']).resolve()
        evidence.require(evidence.sha256(target)==item['sha256'],'full fixture changed: '+str(target))
        return target
    paths={case:binding(item) for case,item in config['audio'].items()}
    paths.update(graphs=binding(config['graphs']),gemma=binding(config['gemma']))
    audio_fixtures={}
    for case in ('pyannote','parakeet'):
        value,targets=audio.manifest(paths[case],case)
        evidence.require(value['schema']==2 and value['conditioning_passes']==2 and value['measured_passes']==3,'full audio schedule must be1first/2conditioning/3measured')
        audio_fixtures[case]=(value,targets)
    graphs=json.loads(paths['graphs'].read_text(encoding='utf-8-sig'))
    evidence.require(graphs['core']==build['core_sha256'] and {c['key'] for c in graphs['cases']}==set(graph.CASES)
                     and len(graphs['cases'])==len(graph.CASES),'full graph coverage/core differs')
    gemma_manifest=gemma.fixture(paths['gemma'].parent)
    evidence.require(config['assets'],'full benchmark asset inventory missing')
    assets={}
    for item in config['assets']:
        remaining(deadline)
        target=(path.parent/item['path']).resolve()
        evidence.require(str(target) not in assets,'repeated full asset')
        evidence.require(target.stat().st_size==item['bytes'] and evidence.sha256(target)==item['sha256'],'full asset changed: '+str(target))
        assets[str(target)]=item['sha256']
    for case in graphs['cases']:
        required=[Path(case['model'])]
        required += [(paths['graphs'].parent/item['file']).resolve() for item in (*case['inputs'],*case['outputs']) if 'file' in item]
        sidecar=Path(case['model']).with_name('model.onnx_data')
        if sidecar.exists():required.append(sidecar)
        evidence.require(all(str(p.resolve()) in assets for p in required),'graph files missing from full asset inventory')
    return config,paths,audio_fixtures,graphs,gemma_manifest


def execute(args,build,output,config,paths,audio_fixtures,graphs,gemma_manifest,deadline):
    env={k:v for k,v in os.environ.items() if not k.lower().startswith(('dotnet_','complus_','lokad_'))}
    env.update({k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    workers=[];native_signatures={};verification_paths={}
    for job in jobs():
        remaining(deadline)
        case,engine,phase=job['case'],job['engine'],job['phase']
        if phase=='timing':evidence.require(len([w for w in workers if w['phase']=='conformance'])==14,'all full-output checks must precede timings')
        if case in ('pyannote','parakeet'):
            fixture,targets=audio_fixtures[case]
            item=audio.child(dict(build,runner=build['runner'].parent/audio.RUNNER),Path(config['audio_root']),output,
                'L1',job['pair'],engine,case,paths[case],fixture,targets,phase,args.dotnet,args.native_python,job['index'],overall_deadline=deadline)
        elif case=='gemma2-128tok':
            smoke=phase=='conformance'
            entry=gemma.child(dict(build,runner=build['runner'].parent/'Lokad.Onnx.EmbeddingGemma2Bench.dll'),Path(config['gemma_root']),output,
                case,engine,args.cpu,768,args.packed_weight_budget,smoke,dict(env,DOTNET_TieredCompilation='1'),
                job['index'],overall_deadline=deadline,dotnet=args.dotnet)
            record,samples,_=gemma.worker(output,entry,smoke,gemma_manifest,check_guest_noise=False)
            evidence.require(record['source_sha']==build['source_sha'] and record['core_sha256']==build['core_sha256'],'full Gemma product changed')
            item=dict(case=case,engine=engine,samples_ms=samples,files=entry,native_signature=record['native_module'],
                      setup_seconds=record['load_ticks']/record['stopwatch_frequency'])
        else:
            item=graph.child(dict(build,runner=build['runner'].parent/graph.RUNNER),output,case,engine,
                'verify' if phase=='conformance' else 'timing',job['pair'],paths['graphs'],graphs,config['native_versions'],env,
                args.cpu,args.dotnet,args.native_python,job['index'],deadline)
            if phase=='conformance':
                verification_paths[case,engine]=(output/item['result']).parent
                if engine=='ort':item['complete_output_check']=graph.compare(verification_paths[case,'lok'],verification_paths[case,'ort'])
        if engine=='ort':
            signature=item.get('native_signature')
            evidence.require(native_signatures.setdefault(case,signature)==signature,'loaded native closure changed: '+case)
        item.update(job)
        workers.append(item)
        common.save_json(output/f"completed-{job['index']:03d}.json",item)
        print(f"COMPLETE {job['index']}/58 {case} {engine} {phase}",flush=True)
    return workers


def require_host(args):
    evidence.require(os.name=='posix' and hasattr(os,'sched_setaffinity') and args.cpu==2,'full audio benchmark requires Linux CPU2')


def completed(output):
    evidence.require(not (output/'failure.json').exists(),'full benchmark has a failure receipt')
    completion=json.loads((output/'complete.json').read_text(encoding='utf-8-sig'))
    report=json.loads((output/'report.json').read_text(encoding='utf-8-sig'))
    commands=json.loads((output/'commands-before-launch.json').read_text(encoding='utf-8-sig'))
    evidence.require(completion['protocol']==report['protocol']==commands['protocol']==PROTOCOL,'full benchmark protocol differs')
    evidence.require(completion['complete'] and report['all_cases_complete'],'full benchmark is incomplete')
    elapsed=completion['elapsed_seconds']
    evidence.require(math.isfinite(elapsed) and 0<elapsed<3600,'full benchmark did not finish under60minutes')
    evidence.require(completion['report_sha256']==evidence.sha256(output/'report.json'),'full report changed')
    evidence.require(commands['jobs']==jobs() and report['summary']==summarize(report['workers']),'full benchmark schedule/summary differs')
    evidence.require(all(report[k]==commands[k] for k in ('source_sha','core_sha256')),'full benchmark product differs')
    return report,completion


def run(args):
    started=time.monotonic();started_utc=dt.datetime.now(dt.timezone.utc).isoformat()
    deadline=started+3600
    require_host(args)
    evidence.require(args.fixtures is not None and not args.case and not args.verify_only,'full mode requires its complete fixture manifest')
    evidence.require(args.dimensions==768,'full Gemma workload uses768 dimensions')
    output,prepared=args.output.resolve(),args.prepared.resolve()
    # Validate fixtures before staging, but include both operations in the total clock.
    build=common.read_build(prepared,'L1')
    config,paths,audio_fixtures,graphs,gemma_manifest=load_fixtures(args.fixtures.resolve(),build,deadline)
    evidence.require(not output.is_relative_to(paths['gemma'].parent),'full output must be outside fixture sources')
    receipt=json.loads((prepared/'full-build.json').read_text(encoding='utf-8-sig'))
    evidence.require(receipt['sdk']==build['sdk'] and receipt['target_framework']==build['target_framework'],'full caller/product build environment differs')
    evidence.require({audio.RUNNER,graph.RUNNER,'Lokad.Onnx.EmbeddingGemma2Bench.dll'}<=set(receipt['files']),'full caller set incomplete')
    evidence.require(evidence.sha256(prepared/'runner-source.zip')==receipt['source_archive_sha256'],'full caller source changed')
    for name,digest in receipt['files'].items():
        evidence.require(evidence.sha256(prepared/'runner'/name)==digest,'full caller closure changed: '+name)
    build=common.stage(prepared,output,'current',audio.RUNNER,product=True)['L1']
    common.save_json(output/'commands-before-launch.json',dict(protocol=PROTOCOL,source_sha=build['source_sha'],core_sha256=build['core_sha256'],
        product_files=build['product_files'],harness=receipt,fixture_sha256=evidence.sha256(args.fixtures.resolve()),configuration=config,
        jobs=jobs(),started_utc=started_utc,duration_limit_seconds=3600,
        schedule=dict(audio_pairs_each=6,first_use=1,conditioning=2,measured=3,non_audio_processes_each_engine=2),
        producer_files={name:evidence.sha256(Path(__file__).with_name(name)) for name in
            ('full_benchmark.py','bench.py','benchmark_worker.py','audio_worker.py','audio_native.py','audio_protocol.py','gemma_worker.py','graph_worker.py','graph_native.py')}))
    shutil.copytree(paths['gemma'].parent,output/'fixtures')
    original=os.sched_getaffinity(0)
    os.sched_setaffinity(0,{0})
    try:
        workers=execute(args,build,output,config,paths,audio_fixtures,graphs,gemma_manifest,deadline)
        summary=summarize(workers)
        remaining(deadline)
        report=dict(protocol=PROTOCOL,all_cases_complete=True,started_utc=started_utc,completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            source_sha=build['source_sha'],core_sha256=build['core_sha256'],target_framework=build['target_framework'],
            duration_limit_seconds=3600,workers=workers,summary=summary,
            note='All seven full workloads; current/native only. Fixed short sampling, all observations retained. Earlier long proofs remain separate.')
        common.save_json(output/'report.json',report)
        remaining(deadline)
        completion=dict(protocol=PROTOCOL,complete=True,elapsed_seconds=time.monotonic()-started,
            completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),report_sha256=evidence.sha256(output/'report.json'))
        common.save_json(output/'complete.json',completion)
        remaining(deadline)
    except BaseException as error:
        common.save_json(output/'failure.json',dict(protocol=PROTOCOL,complete=False,elapsed_seconds=time.monotonic()-started,
            error=str(error),action='Completed workers and failure receipts retained; no truncated success or automatic extension'))
        raise
    finally:os.sched_setaffinity(0,original)
    print(f"PASS full seven-model benchmark: {completion['elapsed_seconds']/60:.2f} minutes; {output/'report.json'}",flush=True)
