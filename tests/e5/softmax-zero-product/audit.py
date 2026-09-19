"""Independently audit the terminal product qualification and model comparison."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import struct
import tarfile
import xml.etree.ElementTree as ET
import numpy as np
from evaluate import CRITERIA, evaluate, require, schedule


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def local_asset(root, remote):
    prefix = 'artifacts/e5-current-comparison-20260919/result/'
    if remote.startswith(prefix):
        remote = remote.replace(prefix, 'artifacts/e5-current-comparison-20260919/collected/result/', 1)
    return root / remote


def input_hash(inputs):
    require(set(inputs) == {'input_ids', 'attention_mask', 'token_type_ids'}, 'Input names differ')
    lengths = {len(v) for v in inputs.values()}
    require(len(lengths) == 1, 'Input lengths differ')
    stream = bytearray(b'LOKAD-CAMPAIGN-INPUTS-1\0') + struct.pack('<i', len(inputs))
    for key, values in sorted(inputs.items()):
        require(all(type(v) is int for v in values), 'Input type differs')
        encoded = key.encode('utf-8')
        stream += struct.pack('<i', len(encoded)) + encoded
        stream += struct.pack('<iiiiq', 7, 2, 1, len(values), len(values))
        stream += struct.pack('<' + 'q' * len(values), *values)
    return hashlib.sha256(stream).hexdigest()


def scaled_error(actual, expected):
    require(actual.size == expected.size and actual.size > 0, 'Output size differs')
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), 'Nonfinite output')
    result = float(np.max(np.abs(actual.astype(float) - expected.astype(float)) /
                          np.maximum(1., np.abs(expected.astype(float)))))
    require(result <= 1e-4, 'Native output agreement failed')
    return result


def validate_model(value, meta, bundle, case, role, flags):
    require(value['schema'] == 3 and value['protocol'] == meta['protocol'] and value['diagnostic_only'], 'Model protocol differs')
    require((value['name'], value['mode'], value['role'], value['optimization'], value['native'], value['parallelism']) ==
            (case, 'memory', role, 'Memory', None, 1), 'Public execution differs')
    require(value['settings'] == flags and value['affinity'] == 4, 'Worker flags/affinity differ')
    require(value['runtime'] == '.NET 10.0.8' and value['avx2'] and value['avx512'] and value['vector_width'] == 8, 'Runtime/ISA differs')
    require(value['core_sha256'] == meta['core_sha256'] and value['probe_sha256'] == meta['probe_sha256'], 'Product/probe differs')
    require(value['runner_sha256'] == bundle['files']['model/bin/Lokad.Onnx.Campaign.dll']['sha256'], 'Canonical producer differs')
    oracle = meta['oracles'][case]
    require(value['fixture_sha256'] == oracle['sha256'], 'Oracle manifest differs')
    require(all(value[k] == oracle['manifest'][k] for k in ('model_sha256', 'tokenizer_sha256', 'input_sha256')), 'Model/workload differs')
    require(input_hash(value['inputs']) == value['input_sha256'], 'Actual input bytes differ')
    require(value['inputs_unchanged'] is True and value['held_outputs_unchanged'] is True, 'Input/output ownership failed')
    require(value['frequency'] == 1_000_000_000, 'Clock frequency differs')
    for key in ('load_ticks', 'first_execute_ticks'):
        require(type(value[key]) is int and value[key] > 0, 'Invalid first/load time')
    condition = value['conditioning']
    require(condition and all(type(t) is int and t > 0 for t in condition), 'Invalid conditioning')
    seconds = sum(condition) / value['frequency']
    require(seconds >= 30 and (sum(condition[:-1]) / value['frequency']) < 30 and
            math.isclose(seconds, value['conditioning_seconds'], abs_tol=1e-8), 'Conditioning boundary differs')
    blocks = {}
    for boundary, include in (('execute', False), ('request', True)):
        block = value[boundary]
        require(block['include_reset'] is include and len(block['ticks']) == 33 and
                all(type(t) is int and t > 0 for t in block['ticks']), 'Timing block differs')
        require(type(block['allocated_bytes']) is int and block['allocated_bytes'] >= 0, 'Invalid allocation')
        gc = [b - a for a, b in zip(block['before']['gc'], block['after']['gc'], strict=True)]
        require(len(gc) == 3 and all(type(v) is int and v >= 0 for v in gc), 'Invalid GC delta')
        blocks[boundary] = dict(samples_ms=[t / 1e6 for t in block['ticks']],
                                allocated_bytes_per_call=block['allocated_bytes'] / 33, gc=gc)
    return dict(**blocks, conditioning_calls=len(condition), conditioning_seconds=seconds,
                load_ms=value['load_ticks'] / 1e6, first_ms=value['first_execute_ticks'] / 1e6)


def verify_archive(base, collected, phase):
    download = read(base / (phase + '-download.json'))
    archive = base / (phase + '-results.tar.gz')
    require(sha(archive) == download['sha256'] and archive.stat().st_size == download['bytes'], 'Downloaded archive differs')
    receipt_path = collected / ('collection-' + phase + '.json')
    require(sha(receipt_path) == download['collection_sha256'], 'Collection receipt differs')
    receipt = read(receipt_path)
    names = set()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            require(member.isfile() and member.name not in names and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts, 'Invalid archive member')
            names.add(member.name)
            with tar.extractfile(member) as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            path = collected / member.name
            require(sha(path) == digest and path.stat().st_size == member.size, 'Extracted bytes differ')
    require(names == {p.relative_to(collected).as_posix() for p in collected.rglob('*') if p.is_file()}, 'Extracted inventory differs')
    require(names == set(receipt['files']) | {receipt_path.name}, 'Collection inventory differs')
    for name, pin in receipt['files'].items():
        path = collected / name
        require(path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'], 'Collection bytes differ: ' + name)
    require(receipt['checkout'] == '172181fc5ab4eb2bdc2eb7f37e80d25e482a0887', 'VM checkout changed')
    require(sha(collected / 'bundle.json') == sha(base / 'payload/bundle.json'), 'Frozen manifest differs')
    bundle = read(collected / 'bundle.json')
    for name, pin in bundle['files'].items():
        require(sha(collected / name) == sha(base / 'payload' / name) == pin['sha256'], 'Frozen payload differs: ' + name)
    return receipt, bundle


def shared_audit(directory, root, meta, flags):
    value = read(directory / 'managed.json')
    reference = root / 'artifacts/shared-regression-20260918/reference'
    require(value['passed'] and value['core_sha256'] == meta['core_sha256'] and value['runtime'] == '10.0.8' and value['flags'] == flags, 'Shared execution differs')
    require(value['reference_sha256'] == sha(reference / 'manifest.json'), 'Shared references differ')
    expected = []
    for model in read(reference / 'manifest.json')['models']:
        for scenario in model['scenarios']:
            for step, row in enumerate(scenario['steps']):
                for output in row['outputs']:
                    expected.append((model['key'], scenario['name'], step, output))
    require(len(value['rows']) == len(expected) == 106, 'Shared coverage differs')
    maxima = {}; bits = []; count = 0
    for actual, (model, scenario, step, output) in zip(value['rows'], expected, strict=True):
        require((actual['model'], actual['scenario'], actual['step'], actual['name']) == (model, scenario, step, output['name']), 'Shared row differs')
        path = directory / actual['file']; native = reference / output['file']
        require(sha(native) == output['sha256'] and sha(path) == actual['sha256'], 'Shared array identity differs')
        want = np.load(native, allow_pickle=False)
        require(list(want.shape) == output['shape'] and want.dtype == np.float32, 'Shared native geometry differs')
        got = np.fromfile(path, '<f4'); error = scaled_error(got, want.reshape(-1))
        require(actual['values'] == want.size and actual['failed_values'] == 0 and error == actual['max_scaled_error'], 'Shared numerical report differs')
        maxima[model] = max(maxima.get(model, 0.), error); bits.append(sha(path)); count += got.size
    require(count == 1_286_766, 'Shared value coverage differs')
    return dict(arrays=106, values=count, maxima=maxima, output_sha256=bits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--phase', choices=('qual', 'model'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); base = args.artifact.resolve(); root = base.parents[1]
    require(not args.output.exists(), 'Existing immutable audit output')
    collected = base / ('collected-' + args.phase); out = collected / ('result-' + args.phase)
    receipt, bundle = verify_archive(base, collected, args.phase)
    meta = read(collected / 'provenance.json')
    require(meta['criteria'] == CRITERIA and meta['schedule'] == schedule(), 'Prospective comparison changed')
    for pin in meta['assets'].values():
        path = local_asset(root, pin['remote'])
        require(path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'], 'Pinned local asset differs')
    identity = read(out / 'identity.json'); deployment = read(collected / ('deployment-' + args.phase + '.json'))
    jobs = meta['qualification'] if args.phase == 'qual' else schedule()
    require(identity['complete'] and identity['phase'] == args.phase and len(identity['runs']) == len(jobs) and
            identity['supervisor_affinity'] == [0] and (collected / ('complete-' + args.phase + '.txt')).read_text().strip() == '0', 'Incomplete campaign')
    require(deployment['pid'] == identity['supervisor'] and deployment['start'] == identity['supervisor_start'] and
            receipt['supervisor'] == dict(pid=identity['supervisor'], start=identity['supervisor_start']), 'Supervisor identities differ')
    require(receipt['terminal_workers'] == [dict(pid=r['pid'], start=r['start_identity'], code=r['code']) for r in identity['runs']], 'Terminal identities differ')
    require(len({r['pid'] for r in identity['runs']}) == len(jobs), 'Repeated worker PID')
    require('AMD EPYC 9V74' in (out / 'cpuinfo.txt').read_text() and '10.0.204' in (out / 'dotnet-info.txt').read_text(), 'Host/SDK differs')
    spec = importlib.util.spec_from_file_location('frozen_processes', collected / 'campaign_processes.py')
    processes = importlib.util.module_from_spec(spec); spec.loader.exec_module(processes)
    workers = []; qualifications = {}; bits = {}; telemetry = dict(maximum_foreign_cpu_fraction=0., maximum_steal_fraction=0., maximum_rss=0)
    for index, (row, job) in enumerate(zip(identity['runs'], jobs, strict=True)):
        require(row['index'] == index and row['job'] == job and row['code'] == 0 and 0 < row['seconds'] < 600 and row['started'] < row['ended'], 'Worker identity or bound differs')
        if index:
            require(identity['runs'][index - 1]['ended'] <= row['started'], 'Workers overlap')
        directory = out / row['tag']; kind = job['kind'] if args.phase == 'qual' else 'model'
        flags = job['flags'].copy() if args.phase == 'qual' else ({'LOKAD_ONNX_SOFTMAX_ZERO_BLOCKS': '1'} if job['role'] == 'candidate' else {})
        if kind == 'codegen':
            flags.update(DOTNET_JitDisasm='*SoftmaxMaskedFloatSpanPtrZeroBlocks*', DOTNET_JitStdOutFile='/home/vermorel/Onnx/artifacts/softmax-zero-product-20260919/result-qual/product-codegen/codegen.txt')
        require(row['flags'] == flags and row['kind'] == kind and row['command'][:4] == ['taskset', '-c', '2', 'dotnet'], 'Worker command/flags differ')
        limit = (8 if kind in ('tests', 'shared') else 6) * 1024**3
        require(row['limit_rss'] == limit and row['samples'] and row['members'][str(row['pid'])]['start'] == row['start_identity'], 'Resource/birth record differs')
        for sample in row['samples']:
            require(all(m['affinity'] == '2' and m['start'] == row['members'][str(m['pid'])]['start'] for m in sample['members']), 'Group identity/affinity differs')
            rss = sum(m['rss'] for m in sample['members']); require(rss < limit, 'Memory bound exceeded')
            telemetry['maximum_rss'] = max(telemetry['maximum_rss'], rss)
        account = processes.foreign_fraction(read(directory / 'pre.json'), read(directory / 'post.json'), identity['supervisor'])
        require(account == row['accounting'], 'Process accounting differs')
        telemetry['maximum_foreign_cpu_fraction'] = max(telemetry['maximum_foreign_cpu_fraction'], account['foreign_cpu_fraction'])
        before = [int(x) for x in (directory / 'cpu-pre.txt').read_text().splitlines()[0].split()[1:]]
        after = [int(x) for x in (directory / 'cpu-post.txt').read_text().splitlines()[0].split()[1:]]
        delta = [b - a for a, b in zip(before, after, strict=True)]
        require(len(delta) >= 8 and all(x >= 0 for x in delta) and sum(delta[:8]) > 0, 'CPU counters differ')
        telemetry['maximum_steal_fraction'] = max(telemetry['maximum_steal_fraction'], delta[7] / sum(delta[:8]))
        if kind == 'tests':
            trx = ET.parse(directory / 'tests.trx'); ns = {'t': 'http://microsoft.com/schemas/VisualStudio/TeamTest/2010'}
            counters = trx.find('.//t:Counters', ns).attrib; results = trx.findall('.//t:UnitTestResult', ns)
            require(counters['total'] == counters['executed'] == counters['passed'] == '280' and len(results) == 280 and all(v.attrib['outcome'] == 'Passed' for v in results), 'AMD contract tests failed')
            require(sum('SoftmaxZeroBlockTests' in v.attrib['testName'] for v in results) == 24, 'New test coverage differs')
            qualifications[row['tag']] = dict(passed=280, zero_block_tests=24)
        elif kind == 'shared':
            qualifications[row['tag']] = shared_audit(directory, root, meta, flags)
        else:
            case = 'e5-30pad128' if kind == 'codegen' else job['name']; role = 'codegen' if kind == 'codegen' else job['role']
            value = read(directory / 'model.json'); record = validate_model(value, meta, bundle, case, role, flags)
            oracle = meta['oracles'][case]; maximum = 0.
            for stage in ('before', 'after'):
                error = 0.
                for number, output in enumerate(oracle['manifest']['outputs']):
                    native = local_asset(root, str(Path(oracle['remote']).parent / output['file']).replace('\\', '/'))
                    path = directory / ('model.json.' + stage + '-' + str(number) + '.f32')
                    require(sha(native) == output['sha256'], 'Native array changed')
                    actual = np.fromfile(path, '<f4'); expected = np.fromfile(native, '<f4')
                    require(actual.size == math.prod(output['dims']), 'e5 output geometry differs')
                    error = max(error, scaled_error(actual, expected)); bits.setdefault((case, number), set()).add(sha(path))
                require(error == value[stage + '_error'], 'e5 reported numerical error differs'); maximum = max(maximum, error)
            workers.append(job | record | dict(index=index, maximum_scaled_error=maximum,
                peak_rss=max(sum(m['rss'] for m in s['members']) for s in row['samples']), pid=row['pid']))
    require(all(len(values) == 1 for values in bits.values()), 'e5 output bits changed across workers/settings')
    if args.phase == 'qual':
        require(qualifications['shared-default']['output_sha256'] == qualifications['shared-enabled']['output_sha256'], 'Shared output bits differ off/on')
        review = read(base / 'codegen-review.json')
        require(review['inspected'] is True and review['core_sha256'] == meta['core_sha256'] and review['codegen_sha256'] == sha(out / 'product-codegen/codegen.txt') and review['full_tier1_bytes'] == 3744, 'Actual product code inspection missing')
        verdict = dict(passed=True, product_codegen_inspected=True, codegen_review_sha256=sha(base / 'codegen-review.json'), qualifications=qualifications)
    else:
        gate = read(collected / 'qualification-audit.json')
        require(gate['passed'] and gate['product_codegen_inspected'] and sha(collected / 'qualification-audit.json') == identity['gate_sha256'] == sha(base / 'qualification-audit.json'), 'Qualification gate differs')
        require(gate['identity_sha256'] == sha(collected / 'result-qual/identity.json') and gate['bundle_manifest_sha256'] == sha(collected / 'bundle.json'), 'Qualification receipt binding differs')
        verdict = evaluate(workers, telemetry)
    result = dict(schema=1, phase=args.phase, integrity_passed=True, core_sha256=meta['core_sha256'], **verdict,
                  telemetry=telemetry, workers=workers, measured_calls=len(workers) * 66 if args.phase == 'model' else 0,
                  maximum_scaled_error=max(w['maximum_scaled_error'] for w in workers), outputs_bit_identical=True,
                  archive_sha256=sha(base / (args.phase + '-results.tar.gz')), identity_sha256=sha(out / 'identity.json'),
                  collection_sha256=sha(collected / ('collection-' + args.phase + '.json')), bundle_manifest_sha256=sha(collected / 'bundle.json'),
                  audit_sha256=sha(__file__), evaluate_sha256=sha(Path(__file__).with_name('evaluate.py')),
                  scope='Empirical product mechanism checks; no root M1 confidence or fresh ORT latency claim')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(args.phase, 'passed', result['passed'], 'max error', result['maximum_scaled_error'], 'telemetry', telemetry)


if __name__ == '__main__':
    main()
