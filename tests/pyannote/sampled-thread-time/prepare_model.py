"""Freeze a diagnostic copy of the original public consumer and exact product DLLs."""
import difflib
from common import *


def main():
    assert not (BASE / 'model-prepared.json').exists() and not (BASE / 'consumer').exists()
    assert pin(BASE / 'toy-closed.json')['sha256'] == 'f6ff187cc29acf7e9aa8951c89ee3eeb816afcd0b20aa32c03867a8041640ae1'
    toy = read(BASE / 'toy-closed.json')
    assert toy['passed']
    verify_spec(toy)
    for identity in toy['identities']:
        terminal(identity)
    assert pin(QUALIFIED / 'closed.json')['sha256'] == 'e3fa3f17df5ec4731bc75ebf519389c571cc4f8d9537d6098f9e9f4ce0a28a7a'
    qualified = read(QUALIFIED / 'closed.json')
    assert qualified['passed']
    verify(qualified['files'])
    for identity in qualified['identities']:
        terminal(identity)
    runtime = BASE / 'runtime'
    shutil.copytree(QUALIFIED / 'application-runtime', runtime)
    assert pin(runtime / 'Lokad.Onnx.dll')['sha256'] == CORE and pin(runtime / 'Lokad.Onnx.Data.dll')['sha256'] == DATA
    source = BASE / 'consumer'
    source.mkdir()
    original = ROOT / 'tests/audio/comparison/Program.cs'
    before = original.read_text(encoding='utf8')
    program = before
    def replace(old, new):
        nonlocal program
        assert program.count(old) == 1, old
        program = program.replace(old, new)
    replace('if(args.Length!=4)throw new ArgumentException("root manifest output conformance|timing");',
        'if(args.Length!=5)throw new ArgumentException("root manifest output timing sampled|control");\nbool sampled=args[4]=="sampled";if(!sampled && args[4]!="control")throw new ArgumentException("diagnostic mode");')
    replace('string family=manifest.GetProperty("family").GetString()!;',
        'string family=manifest.GetProperty("family").GetString()!;Require(family=="pyannote" && !conformance,"Only the original pyannote timing workload");\n'
        f'Require(Sha(typeof(ComputationalGraph).Assembly.Location)=="{CORE}","Qualified Core");\n'
        f'Require(Sha(typeof(Community1Diarizer).Assembly.Location)=="{DATA}","Qualified Data");')
    replace('    Require(Hash(c.Pcm)==c.Hash,"Input changed before request");',
        '    if(pass==warmup && records.Count==warmup*cases.Length)DiagnosticControl.Wait(output,sampled);\n'
        '    if(pass>=warmup && sampled)Require(DiagnosticEvents.Log.IsEnabled(),"Collector stopped before measured request");\n'
        '    Require(Hash(c.Pcm)==c.Hash,"Input changed before request");')
    replace('    long start=Stopwatch.GetTimestamp();object actual;',
        '    process.Refresh();long userBefore=process.UserProcessorTime.Ticks,systemBefore=process.PrivilegedProcessorTime.Ticks;uint threadId=DiagnosticControl.GetCurrentThreadId();\n'
        '    if(pass>=warmup)DiagnosticEvents.Log.Boundary("begin",c.Name,pass);\n'
        '    long start=Stopwatch.GetTimestamp();object actual;')
    replace('    else actual=pyannote!.Diarize(c.Pcm,16000,CancellationToken.None);',
        '    else actual=pass<warmup?SampledRequests.Warmup(pyannote!,c.Pcm):c.Name switch\n'
        '    {\n        "dialogue-30s"=>SampledRequests.Full(pyannote!,c.Pcm),\n'
        '        "dialogue-0-10s"=>SampledRequests.FirstCrop(pyannote!,c.Pcm),\n'
        '        "dialogue-10-20s"=>SampledRequests.SecondCrop(pyannote!,c.Pcm),\n'
        '        "dialogue-20-30s"=>SampledRequests.ThirdCrop(pyannote!,c.Pcm),\n'
        '        _=>throw new InvalidDataException("Fixture marker")\n    };')
    replace('    JsonElement normalized=actual switch',
        '    process.Refresh();long userAfter=process.UserProcessorTime.Ticks,systemAfter=process.PrivilegedProcessorTime.Ticks;\n'
        '    if(pass>=warmup)DiagnosticEvents.Log.Boundary("end",c.Name,pass);\n'
        '    if(pass>=warmup && sampled)Require(DiagnosticEvents.Log.IsEnabled(),"Collector stopped during measured request");\n'
        '    JsonElement normalized=actual switch')
    replace('allocated_bytes=allocatedAfter-allocated};',
        'allocated_bytes=allocatedAfter-allocated,cpu_user_ticks=userAfter-userBefore,cpu_system_ticks=systemAfter-systemBefore,cpu_frequency=TimeSpan.TicksPerSecond,thread_id=threadId};')
    replace('new{schema=1,family,engine="managed",conformance,records,setup_seconds=setupSeconds,',
        'new{passed=true,sampled,schema=1,family,engine="managed",conformance,records,setup_seconds=setupSeconds,')
    (source / 'Program.cs').write_text(program, encoding='utf8')
    shutil.copy2(TOOLS / 'Diagnostic.cs', source / 'Diagnostic.cs')
    shutil.copy2(ROOT / 'tests/Shared/NpySupport.cs', source / 'NpySupport.cs')
    project_source = ROOT / 'tests/audio/comparison/AudioBenchmark.csproj'
    project = project_source.read_text(encoding='utf8')
    assert project.count('../../Shared/NpySupport.cs') == 1
    project = project.replace('../../Shared/NpySupport.cs', 'NpySupport.cs')
    assert project.count('<Compile Include="Program.cs"/>') == 1
    project = project.replace('<Compile Include="Program.cs"/>', '<Compile Include="Program.cs"/><Compile Include="Diagnostic.cs"/>')
    (source / 'SampledAudio.csproj').write_text(project, encoding='utf8')
    save(BASE / 'consumer-adaptation.json', dict(original=pin(original), project=pin(project_source),
        diff=''.join(difflib.unified_diff(before.splitlines(True), program.splitlines(True))),
        scope='Only diagnostic markers, attachment barrier, extra counters and exact runtime guards; all original public/native/input/ownership checks retained.'))
    state = new_state()
    save(BASE / 'model-preparation.json', state)
    try:
        flags = monitor.FLAGS + ['-p:NuGetAudit=false', '-p:FrozenProductDirectory='+str(runtime)]
        for name, command in [
            ('consumer-restore', ['dotnet', 'restore', source / 'SampledAudio.csproj', *flags, '--source', FEED, '--packages', BASE / 'packages']),
            ('consumer-build', ['dotnet', 'build', source / 'SampledAudio.csproj', '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'])]:
            monitor.worker(state, BASE / 'model-preparation.json', name, command, ROOT, [0], 8, 8, 900, True, None)
        binary = source / 'bin/Release/net10.0'
        for name in ['Lokad.Onnx', 'Lokad.Onnx.Data', 'Google.Protobuf', 'FastBertTokenizer', 'Lokad.Tokenizers', 'SixLabors.ImageSharp']:
            assert pin(binary / (name+'.dll')) == pin(runtime / (name+'.dll'))
        for suffix in ['dll', 'deps.json', 'runtimeconfig.json']:
            shutil.copy2(binary / ('SampledAudio.'+suffix), runtime / ('SampledAudio.'+suffix))
        files = dict(toy['files'])
        for path in [BASE / 'toy-closed.json', QUALIFIED / 'closed.json', QUALIFIED / 'dialogue-output/result.json', INPUT,
                     original, project_source, ROOT / 'tests/audio/comparison/audit.py', ROOT / 'tests/Shared/NpySupport.cs']:
            files[rel(path)] = pin(path)
        manifest = read(INPUT)
        for asset in [*manifest['models'].values(), *manifest['native_assets'].values(), *manifest['upstream'].values(), manifest['reference'], *[c['pcm'] for c in manifest['cases']]]:
            p = ROOT / asset['path']
            assert pin(p) == {k:asset[k] for k in ['bytes','sha256']}
            files[rel(p)] = pin(p)
        for folder in [source, runtime, TOOLS]:
            for p in folder.rglob('*'):
                if p.is_file() and 'obj' not in p.relative_to(folder).parts:
                    files[rel(p)] = pin(p)
        save(BASE / 'model-prepared.json', dict(passed=True, files=files, external_files=toy['external_files'],
            core=CORE, data=DATA, consumer=pin(runtime / 'SampledAudio.dll'), jobs=['control', 'sampled-a', 'sampled-b'],
            limits=dict(preflight_gib=10, aggregate_rss_gib=8, available_gib=1, disk_gib=20, output_gib=1, seconds=900),
            scope='Complete public sampled-thread diagnostics only; no speedup, CPU-time attribution or AMD admission from trace weights.'))
        state['code'] = 0
        print(dict(prepared=pin(BASE / 'model-prepared.json')), flush=True)
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(BASE / 'model-preparation.json', state)


if __name__ == '__main__':
    main()
