"""Publish closed M43 results; update the release table only after actual-root qualification."""
import csv,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tests/parakeet/first-use-kernels-graphs-amd'))
from protocol import pin,read

def closed(suffix):
    base=ROOT/'artifacts'/suffix;proof=read(base/'closed.json');assert proof['passed']
    for name,wanted in proof['files'].items():assert pin(base/name)==wanted,name
    return base,proof,read(base/'analysis.json')

def text(name,lines):
    path=OUT/name;assert not path.exists();path.write_text('\n'.join(lines)+'\n',encoding='utf8')

def data(name,value):
    path=OUT/name;assert not path.exists();path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')

def csv_file(name,rows):
    with (OUT/name).open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def graphs():
    base,proof,a=closed('parakeet-first-use-kernels-graphs-amd-20260923')
    from statistics import summarize
    from protocol import ORDER
    for row in a['performance']:
        assert {k:v for k,v in row.items() if k!='key'}==summarize({role:read(base/'collected'/('timing-'+row['key']+'-'+role)/'output/result.json') for role in ORDER})
    assert a['clocks']==5832 and a['measured']==2880
    destination=OUT/'graphs-clocks-20260923.csv';assert not destination.exists();shutil.copyfile(base/'clocks.csv',destination)
    setup=[dict(process=row['name'],seconds=read(base/'collected'/row['name']/'output/result.json')['setup_seconds']) for row in read(base/'collected/identity.json')['runs']]
    assert len(setup)==72;data('graphs-setup-20260923.json',setup)
    data('graphs-observations-20260923.json',dict(closure=pin(base/'closed.json'),payload=pin(base/'payload.json'),**a))
    lines=['# M43 graph comparison with the selected release and Microsoft ORT','',
        '**Release qualification passes.**' if proof['admitted'] else '**Release qualification is incomplete; inspect the repeatability and regression checks below.**','',
        '| Case | Selected ms | Candidate ms | Microsoft ORT ms | Candidate / ORT | Candidate / selected |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in a['performance']:
        lines.append(f"| {r['key']} | {r['current']*1000:.6f} | {r['candidate']*1000:.6f} | {r['ort']*1000:.6f} | {r['ratio']:.6f} | {r['candidate_over_current']:.6f} |")
    lines+=['','AMD EPYC 9V74 CPU 2, monitoring CPU 0; .NET 10.0.8, ORT 1.29.0 CPU.',
        'The unchanged consumer measures Reset, Execute and owned output arrays;',
        'ORT measures session.run and its returned arrays. Load/preparation, input',
        'creation and validation are separate. ORT uses one intra/inter-op thread,',
        'sequential execution and all graph optimizations. No profiler or override.','',
        'Each case first has three fresh numerical processes of three calls. Then',
        'six fresh timing processes run selected, candidate, ORT, ORT, candidate,',
        'selected, with 60 warmups and 60 measurements each. All 5,832 clocks,',
        '2,880 measurements and 72 setup intervals are retained. Exact clock fractions',
        'give equal process weights; no trimming, campaign pooling or retry.','',
        'Every candidate output matches selected bytes. All outputs independently',
        'match fresh ORT within abs(actual-reference)/max(1,abs(reference)) <=1e-4.',
        'Shapes, finiteness, inputs, held outputs and determinism also pass.',
        f"Repeatability: {sum(c['passed'] for r in a['performance'] for c in r['controls'])}/24 controls pass (max/min <=1.10).",
        f"Regression: {sum(r['regression_passed'] for r in a['performance'])}/8 gates pass (candidate/selected <=1.05).",'',
        'Before timing, the supervisor was paused between numerical workers at a',
        'memory preflight. Linking identical closed tensors/binaries preserved all',
        '10,696 checked hashes and paths, reclaiming 570,101,760 tmpfs bytes. Complete',
        'payload verification preceded resumption; no timed worker overlapped the',
        'maintenance, and the 12 GiB preflight bound stayed unchanged.','',
        f"All owners are terminal; {sum(r['samples'] for r in a['resources']):,} resource observations pass. Peak RSS {max(r['peak_rss'] for r in a['resources']):,} bytes.",'',
        '[All clocks](graphs-clocks-20260923.csv), [all setups](graphs-setup-20260923.json),',
        '[complete controls and resources](graphs-observations-20260923.json).','',
        'Closure: `'+pin(base/'closed.json')['sha256']+'`.']
    text('graphs-20260923.md',lines)

def pyannote():
    base,proof,a=closed('parakeet-first-use-kernels-pyannote-app-amd-20260923')
    timing=[r['name'] for r in read(base/'collected/identity.json')['runs'] if r['name'].startswith('timing-')]
    clocks=[];setup=[]
    for name in timing:
        value=read(base/'collected'/name/'output/result.json');setup.append(dict(process=name,seconds=value['setup_seconds']))
        clocks.extend(dict(process=name,**{k:r[k] for k in ['name','pass','phase','start_ticks','end_ticks','frequency','seconds']}) for r in value['records'])
    assert len(clocks)==96 and sum(r['phase']=='measured' for r in clocks)==72
    csv_file('pyannote-clocks-20260923.csv',clocks);csv_file('pyannote-setup-20260923.csv',setup)
    data('pyannote-observations-20260923.json',dict(closure=pin(base/'closed.json'),**a))
    lines=['# M43 complete Pyannote comparison and long meetings','',
        '**Regression gates pass.**' if proof['admitted'] else '**Regression gates fail; this candidate is not admitted for release.**','',
        '| Fixture | Selected s | Candidate s | Microsoft ORT s | Candidate / ORT |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in a['table']:lines.append(f"| {r['name']} | {r['selected']['seconds']:.6f} | {r['candidate']['seconds']:.6f} | {r['ort']['seconds']:.6f} | {r['ratios_to_ort']['candidate']:.6f} |")
    lines+=['','AMD EPYC 9V74 CPU 2, monitoring CPU 0, .NET 10.0.8 and ORT 1.29.0 CPU.',
        'The clock includes frontend, segmentation, embeddings, clustering and owned',
        'public results. Setup and validation are separate. Native settings, inputs',
        'and consumers are unchanged. No profiler or implementation override.','',
        'Six fresh timing processes run selected, candidate, ORT, ORT, candidate,',
        'selected: one warmup and three measurements per fixture. Every one of 96',
        'requests, 72 measurements and six setup intervals is retained. Exact clock',
        'fractions give equal process weights; no trimming, pooling or retries.','',
        f"All {sum(r['passed'] for r in a['performance']['controls'])}/12 stability controls pass: dialogue max/min <=1.10, crops <=1.20.",
        f"All {sum(r['passed'] for r in a['performance']['gates'])}/4 regression gates pass: candidate/selected <=1.05.",
        'All managed public results match exactly. Native conformance passes the',
        'original limits. Both ten-minute meetings and the thirty-second recovery',
        'preserve native decisions and every selected result, including centroids.','',
        f"All owners are terminal; {sum(r['samples'] for r in a['resources']):,} resource observations pass. Peak RSS {max(r['peak_rss'] for r in a['resources']):,} bytes.",'',
        '[All clocks](pyannote-clocks-20260923.csv), [setup](pyannote-setup-20260923.csv),',
        '[complete results and controls](pyannote-observations-20260923.json).','',
        'Closure: `'+pin(base/'closed.json')['sha256']+'`.']
    text('pyannote-20260923.md',lines)

def root():
    base,proof,a=closed('parakeet-first-use-kernels-root-amd-20260923')
    assert a['root_source_verified'] and a['inventory']['implementation_flags_equal']
    data('root-observations-20260923.json',dict(closure=pin(base/'closed.json'),**a))
    text('root-20260923.md',['# M43 actual root and package qualification','',
        'All 421 integrated root inputs match the admitted isolated source. A normal',
        'SDK 10.0.204 build preserves all 3,181 Core and 697 Data method bodies,',
        'implementation flags and public interfaces against the measured candidate.','',
        '| Complete suite | Normal pass / skip | AVX512 disabled pass / skip |',
        '| --- | ---: | ---: |',
        '| Backend | 3,449 / 41 | 3,369 / 121 |',
        '| Tensor | 343 / 0 | 343 / 0 |','',
        'Every test outcome matches its prospective census. Disabling AVX512 skips',
        'the 83 hardware-specific cases and activates three unsupported-hardware',
        'checks; all other outcomes are unchanged. No new skip or test failure.','',
        'The normal NuGet package contains the exact built Core and only the existing',
        'Google.Protobuf 3.33.5 dependency. An independent PackageReference consumer',
        'passes model import, matrix and convolution calls, prepared spatial and',
        'Winograd execution, unchanged inputs and owned-output checks.','',
        f"Measured Core `{a['measured']['Lokad.Onnx.dll']['sha256']}`.",
        f"Built Core `{a['built']['Lokad.Onnx.dll']['sha256']}`.",
        f"Built Data `{a['built']['Lokad.Onnx.Data.dll']['sha256']}`.",'',
        f"All workers are terminal. {sum(r['samples'] for r in a['resources']):,} resource observations pass; peak RSS {max(r['peak_rss'] for r in a['resources']):,} bytes.",
        'This qualification provides no new performance measurement.','',
        '[Full census, package and resource evidence](root-observations-20260923.json).','',
        'Closure: `'+pin(base/'closed.json')['sha256']+'`.'])

def benchmark():
    rb,rp,ra=closed('parakeet-first-use-kernels-root-amd-20260923')
    pb,pp,pa=closed('parakeet-first-use-kernels-app-amd-20260923')
    yb,yp,ya=closed('parakeet-first-use-kernels-pyannote-app-amd-20260923')
    gb,gp,ga=closed('parakeet-first-use-kernels-graphs-amd-20260923')
    assert pp['admitted'] and yp['admitted'] and gp['admitted'] and ra['root_source_verified']
    assert pa['identities']['candidate']==ya['identities']['candidate']==ra['measured']
    assert read(gb/'payload.json')['products']['candidate']['Lokad.Onnx.dll']==ra['measured']['Lokad.Onnx.dll']
    for name,wanted in read(rb/'bundle/evidence/root-applied.json')['source_files'].items():assert pin(ROOT/name)==wanted,name
    files=['src/Lokad.Onnx/MathOps.cs','src/Lokad.Onnx/TensorOps.MatMul.cs','src/Lokad.Onnx/Zzz.ShortWideMatMul.cs']
    assert not subprocess.check_output(['git','diff','HEAD','--name-only','--',*files],cwd=ROOT,text=True).strip()
    commit=subprocess.check_output(['git','log','-1','--format=%h','--',*files],cwd=ROOT,text=True).strip()
    rows=[]
    for label,work,row in [('Parakeet TDT 0.6B V3','Transcribe 20 clips / 213.265 seconds of audio',next(r for r in pa['table'] if r['is_corpus'])),
                           ('Pyannote Community-1','Complete diarization of a 30-second dialogue',next(r for r in ya['table'] if r['audio_seconds']==30))]:
        rows.append(f"| {label} | {work} | {row['candidate']['seconds']:.6f} | {row['ort']['seconds']:.6f} | **{row['ratios_to_ort']['candidate']:.3f}** | Qualified |")
    lookup={r['key']:r for r in ga['performance']}
    for key,label,work in [('e5-30tok','multilingual-e5-small','One 30-token forward pass'),('dinov3','DINOv3 ViT-S/16','One 224x224 image, full weights'),('resnet50','ResNet50','One 224x224 image, feature export'),('gpt2','GPT-2','Four-token prefill, empty past state')]:
        r=lookup[key];assert r['qualified'];rows.append(f"| {label} | {work} | {r['candidate']:.6f} | {r['ort']:.6f} | **{r['ratio']:.3f}** | Qualified |")
    rows+=['| DINOv2-small | 224x224 image | — | — | — | Excluded: numerical agreement gate |',
        '| Whisper Large V3 Turbo | Speech transcription | — | — | — | Supported; current-release comparison deferred |']
    path=ROOT/'BENCHMARK.md';document=path.read_text();start=document.index('| Parakeet TDT');end=document.index('\n\nEvery numerical timing row',start)
    document=document[:start]+'\n'.join(rows)+document[end:]
    start=document.index('The selected product is source');end=document.index('\n## What is timed',start)
    document=document[:start]+f'''The selected product is source `{commit}`, measured as Core `c00a25b4` and
Data `9b623be9`. The [normal root and package qualification](tests/parakeet/first-use-kernels-results/root-20260923.md)
verifies all compiled methods, implementation flags and public interfaces, full
backend/tensor suites in both instruction modes and independent NuGet consumption.
Ordinary mode passes 3,449 backend and 343 tensor tests; the hardware-dependent
skip census in each mode is recorded in that report.
''' +document[end:]
    document=document.replace('Each case uses four fresh processes in Lokad,ORT,ORT,Lokad order.',
        'Each comparison also includes the previous selected product: six fresh\nprocesses run previous, candidate, ORT, ORT, candidate, previous. The table\nreports the qualified candidate, which is the current repository product.')
    document=document.replace('tests/pyannote/winograd-product-results/application-20260923.md','tests/parakeet/first-use-kernels-results/pyannote-20260923.md')
    document=document.replace('tests/parakeet/winograd-baseline-amd/results-20260923.md','tests/parakeet/first-use-kernels-app-amd/results-20260923.md').replace('All 42 repeatability controls','All 63 repeatability controls')
    document=document.replace('tests/benchmarks/release-results/results-20260923.md','tests/parakeet/first-use-kernels-results/graphs-20260923.md')
    document=document.replace('tests/benchmarks/release-amd-v2/README.md','tests/parakeet/first-use-kernels-graphs-amd/README.md')
    document=document.replace('tests/pyannote/winograd-product-app-amd/README.md','tests/parakeet/first-use-kernels-pyannote-app-amd/README.md')
    document=document.replace('tests/parakeet/winograd-baseline-amd/README.md','tests/parakeet/first-use-kernels-app-amd/README.md')
    for name in ['graphs-20260923.md','pyannote-20260923.md','root-20260923.md']:assert (OUT/name).exists()
    path.write_text(document,encoding='utf8')

if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['graphs','pyannote','root','benchmark']
    globals()[sys.argv[1]]()
