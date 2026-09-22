"""Build one isolated normal product and qualify original/wide/spatial raw and layer probes."""
import difflib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import traceback
from transform import SOURCE, once, transform

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-spatial-weight-reuse-20260922'
CONTROL = ROOT/'artifacts/pyannote-blocked-spatial-composition-v3-20260922/runtime'
ROOT_PROOF = ROOT/'artifacts/pyannote-blocked-spatial-root-20260922'
AMD = ROOT/'artifacts/pyannote-blocked-spatial-product-amd-v3-20260922'
PROBES = ROOT/'artifacts/pyannote-blocked-spatial-product-amd-v2-20260922'
FIXTURES = ROOT/'artifacts/pyannote-blocked-spatial-fixtures-20260922/output'
FEED = ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
CORE = '3c2f16b08856426d3dfeff07f1638dd76cee7f06b65bbee230e8e0789679206f'
HEAD = '9533cd67'
spec = importlib.util.spec_from_file_location('spatial_weight_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor); monitor.BASE = BASE
pin, read, save, verify, terminal = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal


def previous():
    for folder, sha in [(ROOT_PROOF, '301fe7a2291f4302aa7b4d24622e72bfd723af129920c753cb86819b9051018f'),
                        (AMD, '88ec9b71d6e7807440ccd65d6148ead8c585a0633ce755898e174b8501b77197')]:
        assert pin(folder/'closed.json')['sha256'] == sha
        proof = read(folder/'closed.json'); assert proof['passed']
        origin = ROOT if folder == ROOT_PROOF else folder
        for name, wanted in proof['files'].items(): assert pin(origin/name) == wanted, name
        for identity in proof.get('identities', proof.get('local_identities', [])): terminal(identity)
    assert pin(CONTROL/'Lokad.Onnx.dll')['sha256'] == CORE
    assert pin(CONTROL/'Lokad.Onnx.Data.dll')['sha256'] == '6318cf48691470b908eec4c4d09c558172e43ce3b04bca9039c68966998a684b'


def product_review():
    value = read(BASE/'instructions.json'); assert value['inventory_complete']
    rows = value['observations']; assert len(rows) == 2
    core, data = rows
    assert core['assembly'] == 'Lokad.Onnx.dll' and data['assembly'] == 'Lokad.Onnx.Data.dll'
    for row in rows:
        assert row['public_surface_equal'] and not row['removed']
        assert row['before_sha256'] == pin(CONTROL/row['assembly'])['sha256']
        assert row['after_sha256'] == pin(BASE/'runtime'/row['assembly'])['sha256']
    assert core['methods'] == 3161 and core['unchanged_methods'] == 3160
    changed, = core['differences']; assert not core['added']
    assert changed.startswith('Lokad.Onnx.ConvBlockedSpatial::Kernel512::')
    assert data['methods'] == data['unchanged_methods'] == 697 and not data['added'] and not data['differences']
    return dict(passed=True, changed=changed, added=[], unchanged_core=3160, unchanged_data=697, public_surface_equal=True)


def canonical_method(serialized):
    value = json.loads(serialized.replace(CORE, '<qualified-core>').replace(pin(BASE/'runtime/Lokad.Onnx.dll')['sha256'], '<qualified-core>'))
    instructions = value['instructions']
    offsets = {r['offset']: i for i, r in enumerate(instructions)}
    for region in value['exceptions']:
        for kind in ['Try', 'Handler']:
            offset = region[kind+'Offset']; end = offset+region[kind+'Length']
            region[kind+'Offset'] = offsets[offset]; region[kind+'Length'] = offsets[end]-offsets[offset]
        if region['filter'] >= 0: region['filter'] = offsets[region['filter']]
    branch_prefixes = ('br', 'beq', 'bge', 'bgt', 'ble', 'blt', 'bne', 'leave')
    for i, row in enumerate(instructions):
        assert row['opcode'] != 'switch'
        if row['opcode'].startswith(branch_prefixes):
            displacement = int.from_bytes(bytes.fromhex(row['operand']), 'little', signed=True)
            row['operand'] = 'target:'+str(offsets[instructions[i+1]['offset']+displacement])
    for row in instructions: del row['offset']
    return value


def consumer_review(mode):
    value = read(BASE/(mode+'-instructions.json')); assert value['inventory_complete']
    row, = value['observations']
    assert row['public_surface_equal'] and not row['removed'] and not row['added']
    owner = 'ModelProbe' if mode == 'layers' else 'Probe'
    names = {key.split('::')[1] for key in row['differences']}
    assert names == ({'Main', 'Supplemental'} if mode == 'wide' else {'Main'}), (mode, names)
    assert all(key.startswith(owner+'::') for key in row['differences'])
    if mode not in ['wide', 'span']:
        key, = row['differences']
        before, after = row['normalized_methods'][key], row['candidate_methods'][key]
        assert before.count(CORE) >= 1 and before.replace(CORE, pin(BASE/'runtime/Lokad.Onnx.dll')['sha256']) == after
    if mode == 'span':
        key, = row['differences']
        before = canonical_method(row['normalized_methods'][key])
        after = canonical_method(row['candidate_methods'][key])
        instructions = before['instructions']
        indexes = [i for i, r in enumerate(instructions) if ' Range(Int32, Int32)' in r['operand']]
        i, = indexes
        assert instructions[i-2] == dict(opcode='ldc.i4.1', operand='')
        assert instructions[i-1] == dict(opcode='ldc.i4.s', operand='0D')
        instructions[i-2] = dict(opcode='ldc.i4.s', operand='15')
        assert before == after
    return dict(passed=True, mode=mode, methods=row['methods'], unchanged=row['unchanged_methods'], changed=row['differences'])


def results():
    candidate = pin(BASE/'runtime/Lokad.Onnx.dll')['sha256']; reports = {}
    for mode in ['raw', 'wide', 'span', 'layers']:
        value = read(BASE/'output'/(mode+'-256.json'))
        assert value['passed'] and value['core'] == candidate and value['lanes'] == 8 and not value['flags']
        assert value['no_performance_measurement']
        if mode != 'layers':
            assert value['geometries'] == 312 and value['layout_cases'] == 331 and value['cases'] == 2648
            assert len(value['observations']) == 2648 and len(value['supplemental']) == 20 and value['rejected'] == 10
            assert value['graph_controls'] == 2668 and value['graph_candidates'] == 5336 and len(value['graph_cases']) == 2668
            assert value['finite_kernel_cases'] == 2496 and value['nonfinite_fallback_cases'] == 152
            assert value['graph_differences'] == value['failed_cases'] == value['scalar_differences'] == value['production_differences'] == 0
            assert {r['m'] for r in value['observations']} == ({64, 128} if mode == 'wide' else {32, 48})
            assert {r['w'] for r in value['observations']} == set(range(21, 34) if mode == 'span' else range(1, 14))
            assert {r['stride'] for r in value['observations']} == {1, 2}
            assert len({tuple(r[k] for k in ['c', 'm', 'h', 'w', 'stride']) for r in value['observations']}) == 312
            if mode == 'raw':
                original = read(AMD/'payload/windows-raw.json')
                for field in ['observations', 'supplemental', 'graph_cases']: assert value[field] == original[field], field
        else:
            assert value['cases'] == value['layer_graphs'] == 108 and len(value['graph_dispatch']) == 216
            assert value['values'] == 119823360 and value['native_failures'] == value['differences'] == 0
            original = read(AMD/'payload/windows-layers.json')
            for field in ['observations', 'graph_dispatch']: assert value[field] == original[field], field
        reports[mode] = {key: value[key] for key in ['passed', 'cases', 'values', 'lanes']}
    return reports


def main():
    assert not BASE.exists(); previous()
    head = subprocess.check_output(['git', 'rev-parse', HEAD], cwd=ROOT, text=True).strip()
    BASE.mkdir(); (BASE/'logs').mkdir(); (BASE/'output').mkdir()
    source = BASE/'source'; source.mkdir()
    archive = BASE/'source.tar'
    subprocess.run(['git', 'archive', '--format=tar', '--output', str(archive), head, '--',
                    'src', 'global.json', 'LICENSE.txt', 'icon.png', 'README.md', 'CHANGELOG.md'], cwd=ROOT, check=True)
    with tarfile.open(archive) as tar: tar.extractall(source, filter='data')
    before = (source/SOURCE).read_text(); candidate, diff = transform(before)
    (source/SOURCE).write_text(candidate, encoding='utf8')
    (BASE/'candidate.patch').write_text(diff, encoding='utf8')
    shutil.copy2(ROOT/'.agent/m21-pyannote-spatial-weight-reuse-20260922.md', BASE/'prospective-plan.md')
    files = {p.as_posix(): pin(p) for folder in [source, TOOLS] for p in folder.rglob('*') if p.is_file()}
    for p in [archive, BASE/'candidate.patch', BASE/'prospective-plan.md', MONITOR, ROOT_PROOF/'closed.json', AMD/'closed.json',
              ROOT/'tests/parakeet/portable-models/common.py', CONTROL/'Lokad.Onnx.dll', CONTROL/'Lokad.Onnx.Data.dll']:
        files[p.as_posix()] = pin(p)
    bridge = ROOT_PROOF/'bridge/bin/Release/net10.0/Bridge.dll'
    consumer_bridge = PROBES/'source/bridge/bin/Release/net10.0/Bridge.dll'
    for path in [bridge, consumer_bridge]:
        for p in path.parent.iterdir():
            if p.is_file(): files[p.as_posix()] = pin(p)
    save(BASE/'inputs.json', dict(files=files, source_commit=head, no_performance_measurement=True))
    own = monitor.psutil.Process(); state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    state_path = BASE/'controller.json'; save(state_path, state)
    flags = monitor.FLAGS+['-p:NuGetAudit=false']; jobs = {}

    def run(name, command, numerical=False):
        jobs[name] = [12 if numerical else 8, 8, 900, numerical]; save(BASE/'jobs.json', jobs)
        monitor.worker(state, state_path, name, command, ROOT, [0], jobs[name][0], 8, 900,
                       not numerical, BASE/'output' if numerical else source)
        print(name, 'passed', flush=True)

    try:
        project = source/'src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj'
        run('cli-restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE/'packages'])
        run('cli-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'])
        shutil.copytree(source/'src/Lokad.Onnx.CLI/bin/Release/net10.0', BASE/'runtime')
        run('inventory', ['dotnet', bridge, CONTROL, BASE/'runtime', BASE/'instructions.json'])
        save(BASE/'instruction-review.json', product_review())
        core = pin(BASE/'runtime/Lokad.Onnx.dll')['sha256']; consumer_files = {}
        for mode in ['raw', 'wide', 'span', 'layers']:
            old = PROBES/'source'/('raw' if mode in ['wide', 'span'] else mode)
            target = BASE/'consumers'/mode; target.mkdir(parents=True)
            for p in old.iterdir():
                if p.is_file():
                    consumer_files[p.as_posix()] = pin(p); shutil.copy2(p, target/p.name)
            filename = 'ModelProbe.cs' if mode == 'layers' else 'Probe.cs'
            old_text = (target/filename).read_text(); text = once(old_text, CORE, core)
            if mode == 'wide':
                text = once(text, 'foreach (int m in new[] { 32, 48 })', 'foreach (int m in new[] { 64, 128 })')
                text = once(text, 'const int c = 32, m = 32, h = 3, w = 7, count = m * h * w;',
                                  'const int c = 32, m = 64, h = 3, w = 7, count = m * h * w;')
            if mode == 'span':
                text = once(text, 'Enumerable.Range(1, 13)', 'Enumerable.Range(21, 13)')
            (target/filename).write_text(text, encoding='utf8')
            (BASE/(mode+'-consumer.diff')).write_text(''.join(difflib.unified_diff(old_text.splitlines(True), text.splitlines(True), fromfile='qualified/'+filename, tofile=mode+'/'+filename)), encoding='utf8')
            consumer_files.update({p.as_posix(): pin(p) for p in target.iterdir() if p.is_file()})
        consumer_files.update({p.as_posix(): pin(p) for p in FIXTURES.iterdir() if p.is_file()})
        save(BASE/'consumer-inputs.json', dict(files=consumer_files, core=pin(BASE/'runtime/Lokad.Onnx.dll')))
        consumer_reviews = []
        for mode in ['raw', 'wide', 'span', 'layers']:
            target = BASE/'consumers'/mode; project, = target.glob('*.csproj')
            run(mode+'-restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE/'packages'])
            run(mode+'-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'])
            assembly = 'LayerGraphs.dll' if mode == 'layers' else 'Lokad.Onnx.Backend.Tests.dll'
            built = target/'bin/Release/net10.0'; assert pin(built/'Lokad.Onnx.dll')['sha256'] == core
            run(mode+'-inventory', ['dotnet', consumer_bridge, AMD/'payload/runtime', built, BASE/(mode+'-instructions.json'), assembly])
            consumer_reviews.append(consumer_review(mode))
        save(BASE/'consumer-review.json', consumer_reviews)
        for mode in ['raw', 'wide', 'span', 'layers']:
            executable = BASE/'consumers'/mode/'bin/Release/net10.0'/('LayerGraphs.dll' if mode == 'layers' else 'Lokad.Onnx.Backend.Tests.dll')
            arguments = ['256']+([FIXTURES] if mode == 'layers' else [])+[BASE/'output'/(mode+'-256.json')]
            run(mode+'-256', ['dotnet', executable, *arguments], True)
        verify(files); verify(consumer_files)
        reports = results()
        save(BASE/'verified.json', dict(passed=True, core=pin(BASE/'runtime/Lokad.Onnx.dll'), data=pin(BASE/'runtime/Lokad.Onnx.Data.dll'),
            product=product_review(), consumers=consumer_reviews, reports=reports, no_performance_measurement=True, avx512_qualified=False))
        state['code'] = 0; print(json.dumps(reports), flush=True)
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(state_path, state)


if __name__ == '__main__': main()
