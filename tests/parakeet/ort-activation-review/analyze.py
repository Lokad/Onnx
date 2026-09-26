"""Identify the sampled SiLU leaf from retained evidence; never run inference."""
from bisect import bisect_right
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-ort-activation-review-20260926'
SAMPLES = ROOT / 'artifacts/parakeet-ort-native-samples-20260924'
SOURCE = ROOT / 'artifacts/parakeet-packed-final-row-gap-20260925'
REVISION = '2e2543fbe9fae542f921d47a72d21d5a4ef0b710'
OBJDUMP = Path('C:/Strawberry/c/bin/objdump.exe')
READELF = Path('C:/Strawberry/c/bin/readelf.exe')
START, END = 0xf6b550, 0xf6b9cd


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,
                    sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def command(args):
    return subprocess.run([str(a) for a in args], check=True, capture_output=True,
                          timeout=45, creationflags=subprocess.CREATE_NO_WINDOW).stdout


def verify_closure(base, expected):
    assert pin(base / 'closed.json')['sha256'] == expected
    closed = read(base / 'closed.json')
    assert closed['passed'] and closed['analysis'] == pin(base / 'analysis.json')
    return closed, read(base / 'analysis.json')


def elf_sections(binary):
    assert binary[:6] == b'\x7fELF\x02\x01'
    offset = struct.unpack_from('<Q', binary, 40)[0]
    size, count, names = struct.unpack_from('<HHH', binary, 58)
    assert size == 64 and 0 < count < 100 and names < count
    headers = [struct.unpack_from('<IIQQQQIIQQ', binary, offset + i * size)
               for i in range(count)]
    table = headers[names]
    strings = binary[table[4]:table[4] + table[5]]
    return {strings[h[0]:strings.index(b'\0', h[0])].decode():
            dict(address=h[3], offset=h[4], bytes=h[5]) for h in headers}


def disassemble(path, first, last, binary):
    output = command([OBJDUMP, '-d', '-M', 'intel', '--insn-width=16',
                      f'--start-address={first}', f'--stop-address={last}', path]).decode()
    rows = {}
    for line in output.splitlines():
        match = re.match(r'^\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} )+)\s+(.+)$', line)
        if match:
            address = int(match[1], 16)
            raw = bytes.fromhex(match[2])
            assert binary[address:address + len(raw)] == raw
            rows[address] = dict(bytes=raw.hex(), instruction=match[3])
    assert rows and min(rows) == first
    for a, b in zip(rows, list(rows)[1:]):
        assert a + len(bytes.fromhex(rows[a]['bytes'])) == b
    assert max(rows) + len(bytes.fromhex(rows[max(rows)]['bytes'])) == last
    return output, rows


def analyze():
    closed, original = verify_closure(SAMPLES,
        '546b3a58af8814772f0a5b816dbcbece710facc01980acebeb2fc93d7f0dcca9')
    _, source = verify_closure(SOURCE,
        '17f0e96862a89606b7e6f56ac5b3ba13fcd36507efa7388bf8c15fff5e0899a7')
    assert source['source_revision'] == REVISION
    assert closed['inspection'] == pin(SAMPLES / 'native-inspection.json')
    assert closed['transfer'] == pin(SAMPLES / 'transfer.json')
    assert closed['auditor'] == pin(SAMPLES / 'audit-final.py')
    transfer = read(SAMPLES / 'transfer.json')
    assert transfer['passed'] and transfer['terminal'] == pin(SAMPLES / 'terminal.json')
    terminal = read(SAMPLES / 'terminal.json')
    inputs = {}
    for name in ['perf.data', 'perf.script', 'target.maps', 'requests/result.json']:
        path = SAMPLES / 'collected' / name
        assert pin(path) == terminal['files'][name]
        inputs[str(path.relative_to(ROOT))] = pin(path)
    receipt = read(BASE / 'binary-receipt.json')
    prospective = read(BASE / 'prospective.json')
    assert receipt['passed'] and receipt['no_inference_overlap'] and receipt['same_owner_resumed']
    assert receipt['source_unchanged'] and receipt['inference_calls'] == 0
    assert receipt['prospective'] == pin(BASE / 'prospective.json')
    assert prospective['helper'] == pin(OUT / 'copy_binary.py')
    assert prospective['sample_closure'] == pin(SAMPLES / 'closed.json')
    binary_path = BASE / 'onnxruntime_pybind11_state.so'
    assert receipt['binary'] == receipt['local'] == original['loaded_binary'] == pin(binary_path)
    binary = binary_path.read_bytes()
    sections = elf_sections(binary)
    for name, first, last in [('.text', 0xd09e00, END), ('.rodata', 0x178a800, 0x191d0c8)]:
        row = sections[name]
        assert row['address'] == row['offset'] and row['address'] <= first < last <= row['address'] + row['bytes']
    note = sections['.note.gnu.build-id']
    namesize, descsize, kind = struct.unpack_from('<III', binary, note['offset'])
    assert (namesize, descsize, kind) == (4, 20, 3)
    assert binary[note['offset'] + 12:note['offset'] + 16] == b'GNU\0'
    build_id = binary[note['offset'] + 16:note['offset'] + 36].hex()
    assert build_id == '3f20b61967f5eab0aff0b64364583da1845ca5e7'
    native_sources = {}
    for name, wanted in source['native_sources'].items():
        path = SOURCE / 'native-source' / name
        assert pin(path) == wanted
        assert command(['git', '-C', ROOT / 'external/onnxruntime', 'show', f'{REVISION}:{name}']) == path.read_bytes()
        native_sources[name] = wanted
    cpp = (SOURCE / 'native-source/onnxruntime/core/mlas/lib/intrinsics/avx512/silu_avx512f.cpp').read_text()
    constants = read(BASE / 'silu-constants.json')
    declarations = dict(re.findall(r'static constexpr float (\w+) = ([^;]+);', cpp))
    assert set(declarations) == {row['name'] for row in constants} and len(constants) == 14
    leaf_text, leaf = disassemble(binary_path, START, END, binary)
    caller_text, caller = disassemble(binary_path, 0xd09e00, 0xd0a917, binary)
    for row in constants:
        address = int(row['address'], 16)
        assert declarations[row['name']] == row['source']
        assert struct.pack('<f', float(row['source'].removesuffix('f'))).hex() == row['bits']
        assert binary[address:address + 4].hex() == row['bits']
        assert f'# {address:x} ' in leaf_text
    for address in [0x191cd2c, 0x17bbdf0]:
        assert struct.unpack_from('<f', binary, address)[0] == 1.0
    assert binary[START:END] == (BASE / 'silu-leaf.bin').read_bytes()
    assert pin(BASE / 'silu-leaf.bin')['sha256'] == '6d00d8dbaed97a36a014c16b2d295be657cbcedd3daeb5a5d9be56219bd27caf'
    required = {
        0xf6b610: 'vmovups zmm20', 0xf6b618: 'vmovups zmm18',
        0xf6b620: 'vminps', 0xf6b62d: 'vmaxps', 0xf6b633: 'vmulps',
        0xf6b68d: 'vdivps', 0xf6b693: 'vaddps', 0xf6b699: 'vmaxps',
        0xf6b69f: 'vminps', 0xf6b6a5: 'vmulps', 0xf6b6ab: 'vmovaps zmm16{k1},zmm20',
        0xf6b70e: 'vdivps', 0xf6b726: 'vmulps', 0xf6b72c: 'vmovaps zmm0{k1},zmm18',
        0xf6b73a: 'add    rax,0x20', 0xf6b8ef: 'vmovups', 0xf6b9af: 'vmovups',
    }
    assert all(leaf[a]['instruction'].startswith(op) for a, op in required.items())
    assert sum('vfmadd' in row['instruction'] for a, row in leaf.items() if 0xf6b645 <= a < 0xf6b67b) == 9
    assert not any(row['instruction'].startswith('call') for row in leaf.values())
    assert '# 17bbdf0 ' in caller[0xd0a329]['instruction']
    assert caller[0xd0a331]['instruction'].startswith('mov    edx,0x1000')
    assert caller[0xd0a350]['instruction'].startswith('shl    rdi,0xc')
    assert caller[0xd0a36e]['instruction'].startswith('ucomiss')
    assert caller[0xd0a375]['instruction'].startswith('jp     d0a37d ')
    assert caller[0xd0a377]['instruction'].startswith('je     d0a540 ')
    assert caller[0xd0a55d]['instruction'].startswith('call')
    assert '# 1cb90b8 ' in caller[0xd0a55d]['instruction']
    assert len(bytes.fromhex(caller[0xd0a55d]['bytes'])) == 6
    # Independently recover function boundaries from unwind records.
    frames = command([READELF, '-wf', binary_path]).decode()
    frame_rows = [line for line in frames.splitlines() if re.search(
        r'pc=0*f6b550\.\.0*f6b9cd|pc=0*d09e00\.\.0*d0a917', line)]
    assert len(frame_rows) == 2
    sys.path.insert(0, str(ROOT / 'tests/parakeet/ort-diagnosis-amd'))
    spec = importlib.util.spec_from_file_location('retained_sample_parser', SAMPLES / 'audit-final.py')
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)  # Only use its pure parsers; never execute main.
    raw, addresses = parser.raw_records(SAMPLES / 'collected/perf.data')
    assert json.loads(json.dumps(raw)) == original['raw'] and raw['lost'] == 0
    maps = []
    for line in (SAMPLES / 'collected/target.maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5] == prospective['path'] and 'x' in parts[1]:
            first, last = [int(s, 16) for s in parts[0].split('-')]
            maps.append((first, last, int(parts[2], 16)))
    assert len(maps) == 1
    first, last, file_offset = maps[0]
    requests = read(SAMPLES / 'collected/requests/result.json')['records']
    windows = sorted((r['start_ticks'], r['end_ticks']) for r in requests if r['phase'] == 'measured')
    assert len(windows) == 60 and all(a[1] <= b[0] for a, b in zip(windows, windows[1:]))
    starts = [a for a, b in windows]
    selected = []
    seen = set()
    measured = weight = 0
    for sample in parser.samples(SAMPLES / 'collected/perf.script'):
        key = (sample['pid'], sample['tid'], sample['stamp'])
        assert key in addresses and key not in seen
        seen.add(key)
        index = bisect_right(starts, sample['stamp']) - 1
        if index < 0 or sample['stamp'] >= windows[index][1]:
            continue
        assert sample['pid'] == terminal['state']['target']['pid']
        measured += 1
        weight += sample['period']
        ip = addresses[key]
        if not first <= ip < last:
            continue
        offset = ip - first + file_offset
        if not START <= offset < END:
            continue
        assert sample['frames'][0]['ip'] == ip and offset in leaf
        assert sample['frames'][0]['dso'] == prospective['path']
        stack = []
        for frame in sample['frames'][:7]:
            assert frame['dso'] == prospective['path'] and first <= frame['ip'] < last
            stack.append(hex(frame['ip'] - first + file_offset))
        assert stack[1] == '0xd0a562'  # Unwind-adjusted address lies within the indirect call.
        selected.append(dict(period=sample['period'], request=index, offset=hex(offset), stack=stack))
    assert seen == set(addresses) and len(seen) == original['total_samples']
    assert measured == original['measured_samples'] and weight == original['measured_period_ns']
    assert selected == read(BASE / 'silu-samples.json')
    counts = Counter(row['offset'] for row in selected)
    period = sum(row['period'] for row in selected)
    assert len(selected) == 103 and len(counts) == 18 and period == 517587875
    value = dict(passed=True, new_inference_calls=0, source_revision=REVISION,
        sample_closure=pin(SAMPLES / 'closed.json'), source_closure=pin(SOURCE / 'closed.json'),
        binary=pin(binary_path), build_id=build_id, binary_receipt=pin(BASE / 'binary-receipt.json'),
        native_sources=native_sources, raw_inputs=inputs, constants=constants,
        leaf=dict(name='MlasSiluKernelAvx512F', start=hex(START), end=hex(END),
                  **pin(BASE / 'silu-leaf.bin'), instruction_count=len(leaf),
                  identification='source, exact constants, instructions, caller and observed samples',
                  source_rebuild_byte_match=False),
        samples=dict(count=len(selected), requests=len({r['request'] for r in selected}),
                     measured_requests=len(windows), measured_samples=measured,
                     period_ns=period, total_measured_period_ns=weight, share=period / weight,
                     estimated_seconds_per_corpus=period / 3e9, instructions=dict(counts)),
        caller=dict(start='0xd09e00', end='0xd0a917', call='0xd0a55d', call_end='0xd0a563',
                    sampled_frame='0xd0a562', dispatch_slot='0x1cb90b8', alpha=1.0, chunk=4096),
        graph=dict(quickgelu_nodes=len(source['activations']),
                   families=dict(Counter(a['family'] for a in source['activations'])),
                   older_profile_silu_seconds=source['activation_total']['ort_seconds']),
        attribution_only=True, per_node_sample_join=False, new_candidate_selected=False,
        managed_source_review={str(p.relative_to(ROOT)): pin(p) for p in [
            ROOT / 'src/Lokad.Onnx/MathOps.cs',
            ROOT / 'src/Lokad.Onnx/CPUExecutionProvider.Elementwise.cs',
            ROOT / 'tests/parakeet/vector-sigmoid-source/Sigmoid.cs.txt']},
        analyst=pin(Path(__file__)),
        tools={str(p): pin(p) for p in [OBJDUMP, READELF]})
    assert value['graph']['quickgelu_nodes'] == 72
    return value, {'reviewed-leaf-disassembly.txt': leaf_text,
                   'reviewed-caller-disassembly.txt': caller_text,
                   'reviewed-unwind.json': json.dumps(frame_rows, indent=2)}


def main():
    assert sys.argv[1:] in [[], ['--publish']]
    publishing = sys.argv[1:] == ['--publish']
    targets = [BASE / 'analysis.json', BASE / 'closed.json',
               *[BASE / name for name in ['reviewed-leaf-disassembly.txt',
                                          'reviewed-caller-disassembly.txt', 'reviewed-unwind.json']]]
    if publishing:
        assert not any(path.exists() for path in targets), 'Preserve closed analyses'
    value, text_files = analyze()
    if publishing:
        for name, content in text_files.items():
            with (BASE / name).open('x', encoding='utf8', newline='\n') as stream:
                stream.write(content)
        with (BASE / 'analysis.json').open('x', encoding='utf8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
        evidence = ['analysis.json', 'prospective.json', 'binary-receipt.json',
                    'silu-leaf.bin', 'silu-samples.json', 'silu-constants.json', *text_files]
        with (BASE / 'closed.json').open('x', encoding='utf8') as stream:
            json.dump(dict(passed=True, new_inference_calls=0, files={name: pin(BASE / name)
                          for name in evidence}, analyst=pin(Path(__file__))), stream, indent=2)
    print(json.dumps(dict(passed=True, published=publishing, leaf=value['leaf'], samples=value['samples'])))


if __name__ == '__main__':
    main()
