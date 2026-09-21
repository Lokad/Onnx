"""Read the new captures using the exact already-qualified offline exporter."""
from gc_common import *


def main():
    assert not GC_BASE.exists()
    files = {}
    for path in [BASE / 'model-closed.json', OLD_GC / 'closed.json']:
        if path.parent == OLD_GC:
            assert pin(path)['sha256'] == '4191974e3e6863ef3359412ec6a1c50f8bfde65eb89a078652ccd8f4aedbc8c1'
        proof = read(path)
        assert proof['passed']
        verify_spec(proof)
        for identity in proof['identities']:
            terminal(identity)
        files[rel(path)] = pin(path)
    GC_BASE.mkdir()
    (GC_BASE / 'logs').mkdir()
    reader = GC_BASE / 'reader'
    shutil.copytree(OLD_GC / 'source/bin/Release/net10.0', reader)
    for name in ['Microsoft.Diagnostics.Tracing.TraceEvent.dll', 'Microsoft.Diagnostics.FastSerialization.dll', 'Microsoft.Diagnostics.NETCore.Client.dll']:
        assert pin(reader / name) == pin(BASE / 'tracer' / name)
    for path in [*reader.iterdir(), *TOOLS.glob('*.py'), GC_TOOLS / 'analyze_v3.py', GC_TOOLS / 'common.py',
        GC_TOOLS / 'test_analysis_v3.py', OLD_GC / 'unit-tests-v3.json']:
        if path.is_file():
            files[rel(path)] = pin(path)
    for name in ['sampled-a', 'sampled-b']:
        for stem in ['capture.nettrace', 'result.json', 'ready.json']:
            path = BASE / name / stem
            files[rel(path)] = pin(path)
    save(GC_BASE / 'prepared.json', dict(passed=True, files=files, inference_executed=False,
        scope='Offline GC attribution of newly closed integrated-runtime captures; reuse the exact exporter and tested parser.'))
    monitor.BASE = GC_BASE
    state = new_state()
    save(GC_BASE / 'processes.json', state)
    try:
        for name in ['sampled-a', 'sampled-b']:
            monitor.worker(state, GC_BASE / 'processes.json', name,
                ['dotnet', reader / 'Export.dll', BASE / name / 'capture.nettrace', GC_BASE / name], ROOT,
                [0], 2, 2, 300, False, GC_BASE / name)
            print(name, 'offline GC export passed', flush=True)
        verify(files)
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(GC_BASE / 'processes.json', state)


if __name__ == '__main__':
    main()
