"""Thin serial coordinator for Paradee's frozen tensors and shared graph workers."""
import argparse,datetime as dt,hashlib,json,math,os,re,statistics,subprocess,time
from pathlib import Path
import xml.etree.ElementTree as ET

import graph_worker as graph
import campaign_evidence as evidence


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def qualify(path,core):
    document=ET.parse(path)
    results=document.findall('.//{*}UnitTestResult')
    expected={'Fp32ControlledNoiseMatchesOrt','Fp32OriginalExportRuns','Int8ControlledNoiseMatchesOrt','Int8OriginalExportRuns',
              'Fp32WaveformOnlyMatchesOrt','Int8WaveformOnlyMatchesOrt'}
    evidence.require(len(results)==6 and {r.attrib['testName'].split('.')[-1] for r in results}==expected,
                     'Paradee qualification must contain exactly the six model tests')
    for result in results:
        evidence.require(result.attrib['outcome']=='Passed','Paradee model qualification did not pass')
        text=''.join(result.itertext())
        evidence.require(re.search(r'core_sha256='+re.escape(core)+r'; runtime=11\.',text) is not None,
                         'Paradee qualification uses another product/runtime')
    return dict(file=str(path.resolve()),sha256=digest(path),passed=True,core_sha256=core)


def summarize(workers,cases,pairs):
    summary={}
    for case in cases:
        selected=[w for w in workers if w['case']==case and w['phase']=='timing']
        evidence.require(len(selected)==pairs*2,'incomplete Paradee process cohort')
        scores={}
        for item in selected:
            key=(item['pair'],item['engine'])
            evidence.require(key not in scores and item['samples_ms'] and all(math.isfinite(v) and v>0 for v in item['samples_ms']),
                             'invalid/duplicate Paradee process score')
            scores[key]=statistics.mean(item['samples_ms'])
        lok=[scores[p,'lok'] for p in range(1,pairs+1)]
        ort=[scores[p,'ort'] for p in range(1,pairs+1)]
        ratios=[a/b for a,b in zip(lok,ort,strict=True)]
        logs=[math.log(value) for value in ratios];center=statistics.mean(logs)
        result=dict(lok_ms=statistics.mean(lok),ort_ms=statistics.mean(ort),ratio=math.exp(center),
                    lok_process_ms=lok,ort_process_ms=ort,paired_ratios=ratios,
                    aggregation='process mean; geometric paired ratio')
        if pairs==6:
            # The same six-pair log-Student-t interval as full_benchmark.py.
            critical=2.570581835636305;margin=critical*statistics.stdev(logs)/math.sqrt(6)
            result.update(interval_95=[math.exp(center-margin),math.exp(center+margin)],df=5,t_critical=critical,
                          target_pass=math.exp(center)<1.03 and math.exp(center+margin)<1.03)
        summary[case]=result
    return summary


def main():
    started=time.monotonic();started_utc=dt.datetime.now(dt.timezone.utc).isoformat()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--runner',type=Path,required=True)
    parser.add_argument('--qualification',type=Path,required=True)
    parser.add_argument('--dotnet',type=Path,required=True)
    parser.add_argument('--native-python',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu',type=int,default=2)
    parser.add_argument('--max-seconds',type=int,default=2700)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--cases',nargs='+')
    args=parser.parse_args()
    evidence.require(0<args.max_seconds<=3600,'invalid overall duration limit')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    args.manifest=args.manifest.resolve();args.runner=args.runner.resolve()
    spec=json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    source_root=Path(__file__).resolve().parent.parent
    source_sha=subprocess.check_output(['git','-C',str(source_root),'rev-parse','HEAD'],text=True).strip()
    source_dirty=bool(subprocess.check_output(['git','-C',str(source_root),'status','--porcelain'],text=True).strip())
    evidence.require(args.pilot or not source_dirty,'A measured cohort requires a clean committed producer')
    core=digest(args.runner.parent/'Lokad.Onnx.dll')
    evidence.require(core==spec['core'],'Paradee product identity changed')
    qualification=qualify(args.qualification,core)
    cases=[item['key'] for item in spec['cases']]
    if args.cases:
        evidence.require(len(set(args.cases))==len(args.cases) and set(args.cases)<=set(cases),'unknown/duplicate Paradee cases')
        cases=[name for name in cases if name in args.cases]
    evidence.require(cases and len(cases)==len(set(cases)),'empty/duplicate Paradee workload')
    for item in spec['cases']:
        evidence.require(item.get('stochastic_outputs') is True,'Paradee timings must identify their stochastic outputs')
        evidence.require(digest(item['model'])==item['model_sha256'],'Paradee model identity changed')
        timing=item['timing']
        # Default .NET tiering continued beyond 20 seconds in fixed 90-second
        # timecourses. Keep both engines on the same conditioning schedule.
        wanted=dict(warmup_calls=3,warmup_seconds=0,measured_calls=3) if args.pilot else dict(warmup_calls=32,warmup_seconds=40,measured_calls=33)
        evidence.require(timing==wanted,'Paradee timing schedule changed')
    versions=json.loads(subprocess.check_output([str(args.native_python),'-c',
        'import json,numpy,onnxruntime; print(json.dumps(dict(numpy=numpy.__version__,onnxruntime=onnxruntime.__version__)))'],text=True))
    build=dict(runner=args.runner,core_sha256=core,target_framework='net11.0')
    env={k:v for k,v in os.environ.items() if not k.lower().startswith(('dotnet_','complus_','lokad_','lonnx_'))}
    pairs=1 if args.pilot else 6;workers=[];comparisons=[];index=0
    deadline=started+args.max_seconds
    record=dict(protocol='paradee-runtime-pilot-v1' if args.pilot else 'paradee-runtime-v2',started_utc=started_utc,
                manifest=str(args.manifest),manifest_sha256=digest(args.manifest),core_sha256=core,
                source_sha=source_sha,source_dirty=source_dirty,runner_sha256=digest(args.runner),
                native_worker_sha256=digest(Path(graph.__file__).with_name('graph_native.py')),
                qualification=qualification,native_versions=versions,
                cases=cases,pairs=pairs,cpu=args.cpu,timing=wanted,workers=workers,comparisons=comparisons)
    def child(case,engine,mode,pair):
        nonlocal index
        index+=1
        result=graph.child(build,args.output,case,engine,mode,pair,args.manifest,spec,versions,
                           env,args.cpu,args.dotnet,args.native_python,index,deadline)
        result['first_use_ms']=None
        detail=json.loads((args.output/result['result']).read_text(encoding='utf-8-sig'))
        first=detail['clocks'][0]
        result['first_use_ms']=first['ticks']*1000/first['frequency']
        workers.append(result)
        print('COMPLETE',index,case,engine,mode,pair,flush=True)
        return args.output/Path(result['result']).parent
    try:
        for case in cases:
            managed=child(case,'lok','verify',0);native=child(case,'ort','verify',0)
            comparisons.append(dict(case=case,**graph.compare(managed,native)))
        for case in cases:
            for pair in range(1,pairs+1):
                for engine in (('lok','ort') if pair%2 else ('ort','lok')):child(case,engine,'timing',pair)
        record.update(complete=True,summary=summarize(workers,cases,pairs),elapsed_seconds=time.monotonic()-started)
        evidence.require(record['elapsed_seconds']<args.max_seconds,'Paradee comparison exceeded duration limit')
        evidence.require(digest(args.manifest)==record['manifest_sha256'],'Paradee manifest changed during collection')
        evidence.require(digest(args.runner.parent/'Lokad.Onnx.dll')==core,'Paradee product changed during collection')
        evidence.require(digest(args.runner)==record['runner_sha256'],'Paradee caller changed during collection')
        evidence.require(digest(Path(graph.__file__).with_name('graph_native.py'))==record['native_worker_sha256'],'Paradee native worker changed during collection')
        signatures={w['native_signature']['sha256'] for w in workers if w['engine']=='ort'}
        evidence.require(len(signatures)==1,'Paradee native binary changed between workers')
        (args.output/'complete.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(json.dumps(record['summary'],indent=2),flush=True)
    except BaseException as error:
        record.update(complete=False,error=repr(error),elapsed_seconds=time.monotonic()-started)
        (args.output/'failure.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        raise


if __name__=='__main__':main()
