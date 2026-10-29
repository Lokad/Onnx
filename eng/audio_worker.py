"""Complete audio application workers for the existing paired benchmark command."""
import datetime as dt
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import audio_protocol as protocol
import benchmark_worker as common
import campaign_evidence as evidence

RUNNER = 'Lokad.Onnx.AudioBench.dll'


def manifest(path, family):
    value = protocol.read(path)
    protocol.definition(value)
    evidence.require(value['family'] == family, 'wrong audio fixture family')
    cases = value['cases']
    evidence.require(len({c['name'] for c in cases}) == len(cases), 'duplicate audio case')
    targets = [c['name'] for c in cases if family == 'parakeet' or c['samples'] == 480000]
    evidence.require((len(cases), len(targets)) == ((20, 20) if family == 'parakeet' else (4, 1)), 'audio workload coverage changed')
    return value, targets


def measured(value, fixture, targets, phase):
    protocol.validate_records(value, fixture, phase)
    if phase == 'conformance':
        return []
    # Retain all measured full-workload sums. Conditioning and first-use stay
    # in the raw records and never enter the sustained process mean.
    samples = [sum(r['seconds'] for r in value['records'] if r['pass'] == i and r['name'] in targets) * 1000
               for i, label in enumerate(protocol.schedule(fixture, phase)) if label == 'measured']
    evidence.require(len(samples) == fixture['measured_passes'] and all(math.isfinite(x) and x > 0 for x in samples), 'missing audio samples')
    return samples


def child(build, root, output, role, pair, engine, family, fixture_path, fixture, targets, phase, dotnet, python, index):
    import psutil
    stem = f'audio-{index:03d}-{family}-{role.lower()}-{engine}-{phase}'
    destination = output / stem
    env = {k: v for k, v in os.environ.items() if not k.lower().startswith(('dotnet_', 'complus_', 'lokad_'))}
    env.update({k: '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']})
    prefix = [str(dotnet), str(build['runner'])] if engine == 'lok' else [str(python), str(Path(__file__).with_name('audio_native.py'))]
    command = prefix + [str(root), str(fixture_path), str(destination), phase]
    started = dt.datetime.now(dt.timezone.utc)
    print('Starting ' + stem, flush=True)
    peak, failure = 0, None
    with (output / (stem + '.log')).open('x') as log:
        # Pin in pre-exec too: a monitor must never catch taskset before it has
        # changed its inherited CPU-0 affinity. The CLR starts on CPU 2.
        process = common.spawn_child(command, 2, cwd=root, env=env, stdout=log, stderr=log,
                                     preexec_fn=lambda: os.sched_setaffinity(0, {2}))
        try:
            timeout = protocol.definition(fixture)[2]
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                evidence.require(time.monotonic() < deadline, f'audio process exceeded {timeout} seconds')
                try:
                    child_process = psutil.Process(process.pid)
                    evidence.require(child_process.cpu_affinity() == [2], 'audio process affinity changed')
                    peak = max(peak, child_process.memory_info().rss)
                    evidence.require(peak < 14 * 1024**3 and psutil.virtual_memory().available > 1024**3, 'audio worker memory limit')
                    for thread in child_process.threads():
                        try:
                            evidence.require(os.sched_getaffinity(thread.id) == {2}, 'audio thread affinity changed')
                        except ProcessLookupError:
                            pass
                except psutil.NoSuchProcess:
                    pass
                time.sleep(.5)
            code = process.wait()
        except BaseException as error:
            failure = error
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            code = process.returncode
    supervision = dict(command=command, pid=process.pid, exit_code=code, started_utc=started.isoformat(),
                       completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(), peak_resident_bytes=peak,
                       failure=None if failure is None else str(failure))
    common.save_json(output / (stem + '.supervision.json'), supervision)
    if failure is not None:
        raise failure
    evidence.require(code == 0, 'audio worker failed: ' + stem)
    value = protocol.read(destination / 'result.json')
    evidence.require(value['producer'] == protocol.definition(fixture)[0], 'wrong audio producer')
    evidence.require(value['manifest_sha256'] == evidence.sha256(fixture_path), 'audio fixture changed')
    if engine == 'lok':
        evidence.require(value['runner_sha256'] == evidence.sha256(build['runner']), 'loaded audio caller changed')
        for name, digest in build['product_files'].items():
            evidence.require(evidence.sha256(build['runner'].parent / name) == digest, 'staged product changed')
        evidence.require(value['core_sha256'] == build['core_sha256'], 'loaded core changed')
        evidence.require(value['data_sha256'] == build['product_files']['Lokad.Onnx.Data.dll'], 'loaded Data changed')
        evidence.require(value['sdk'] == build['sdk'], 'caller/core build SDK differs')
        framework = build['target_framework'][3:]
        evidence.require(value['runtime'].startswith('.NET ' + framework + '.'), 'wrong managed audio runtime')
    else:
        evidence.require(value['versions'] == fixture['native_versions'] and value['native_binaries'] == fixture['native_binaries'], 'native closure changed')
    samples = measured(value, fixture, targets, phase)
    native_signature = {name: binding['sha256'] for name, binding in value.get('numeric_libraries', {}).items()}
    return dict(case=family, pair=pair, role=role, engine=engine, aggregation='mean', samples_ms=samples,
                source_sha=build['source_sha'], core_sha256=build['core_sha256'], setup_seconds=value['setup_seconds'],
                peak_resident_bytes=peak, runtime=value['runtime'], native_signature=native_signature, result=destination.name + '/result.json',
                result_sha256=evidence.sha256(destination / 'result.json'), supervision=stem + '.supervision.json')


def run(args, summarize, order):
    evidence.require(sys.platform.startswith('linux') and args.cpu == 2, 'pinned audio fixtures require Linux CPU 2')
    evidence.require(args.fixtures is not None and not args.case, 'audio needs --fixtures <manifest.json>; its complete case set is fixed')
    output, prepared, root = args.output.resolve(), args.prepared.resolve(), args.root.resolve()
    fixture_path = args.fixtures.resolve()
    fixture, targets = manifest(fixture_path, args.model)
    harness = None
    if fixture['schema'] == 2:
        harness = protocol.read(prepared / 'runner-build.json')
        evidence.digest(harness['source_sha'], 'audio caller source', 40)
        evidence.require(evidence.sha256(prepared / 'runner-source.zip') == harness['source_archive_sha256'], 'audio caller source archive changed')
        evidence.require(evidence.sha256(prepared / 'runner' / RUNNER) == harness['runner_sha256'], 'audio caller changed')
    builds = common.stage(prepared, output, 'comparison', RUNNER, product=True)
    if harness is not None:
        evidence.require(all(build['sdk'] == harness['sdk'] and build['target_framework'] == harness['target_framework'] for build in builds.values()), 'caller/product build environment differs')
        common.save_json(output / 'runner-build.json', harness)
    common.save_json(output / 'fixture.json', fixture)
    # The supervisor is separate from the single CPU used by workers.
    os.sched_setaffinity(0, {0})
    sources = {name: evidence.sha256(Path(__file__).with_name(name)) for name in
               ('bench.py', 'benchmark_worker.py', 'audio_worker.py', 'audio_native.py', 'audio_protocol.py', 'campaign_evidence.py', 'parse_baseline.py')}
    common.save_json(output / 'producer-files.json', sources)
    index, verification, workers = 0, [], []
    for role, engine in [('L0', 'lok'), ('L1', 'lok'), ('L0', 'ort')]:
        index += 1
        verification.append(child(builds[role], root, output, role, 0, engine, args.model, fixture_path, fixture, targets,
                                  'conformance', args.dotnet, args.native_python, index))
    common.save_json(output / 'verification.json', dict(fixture_sha256=evidence.sha256(fixture_path), workers=verification))
    if args.verify_only:
        print('PASS audio conformance: ' + str(output / 'verification.json'), flush=True)
        return
    for pair, roles in enumerate(order, 1):
        for role_index, role in enumerate(roles):
            for engine in (('lok', 'ort') if role_index == 0 else ('ort', 'lok')):
                index += 1
                item = child(builds[role], root, output, role, pair, engine, args.model, fixture_path, fixture, targets,
                             'timing', args.dotnet, args.native_python, index)
                if engine == 'ort':
                    evidence.require(item['native_signature'] == verification[-1]['native_signature'], 'loaded native libraries changed')
                workers.append(item)
                common.save_json(output / f'completed-{index:03d}.json', item)
    report = dict(kind='engineering-comparison', model=args.model, pairs=args.pairs,
                  protocol=protocol.definition(fixture)[0], schedule=protocol.definition(fixture)[1],
                  contract='complete public API; declared phases retained; process mean of measured full-workload sums; setup and first-use separate',
                  fixture_sha256=evidence.sha256(fixture_path), target_cases=targets, verification=verification,
                  producer_files=sources, harness=harness,
                  builds={role: {key: build[key] for key in ('source_sha', 'sdk', 'target_framework', 'core_sha256', 'product_files', 'source_archive_sha256', 'data_source_archive_sha256')}
                          for role, build in builds.items()},
                  workers=workers, summary=summarize(workers))
    common.save_json(output / 'report.json', report)
    print('PASS full audio comparison: ' + str(output / 'report.json'), flush=True)
