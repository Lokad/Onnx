"""Routine paired CPU comparisons with existing validated workers; no calibration or forced cooldown."""
import argparse
import math
import os
from pathlib import Path
import shutil
import statistics

import campaign_evidence as evidence
import isolated_evidence as isolated
import run_common_campaign as common
import run_embeddinggemma2 as gemma
import run_isolated_e5 as e5


def visits(pairs):
    evidence.require(pairs >= 2 and pairs % 2 == 0, 'use an even number of pairs, at least two')
    return [('L0', 'L1') if pair % 2 == 0 else ('L1', 'L0') for pair in range(pairs)]


def summarize(workers):
    evidence.require(workers, 'no completed workers')
    groups = {}
    for worker in workers:
        samples = worker['samples_ms']
        evidence.require(samples and all(math.isfinite(value) and value > 0 for value in samples), 'missing/invalid timings')
        key = (worker['case'], worker['pair'], worker['role'], worker['engine'])
        evidence.require(key not in groups, 'duplicate worker')
        groups[key] = statistics.median(samples)
    result = {}
    for name in sorted({key[0] for key in groups}):
        pairs = sorted({key[1] for key in groups if key[0] == name})
        arms = {}
        for role in ('L0', 'L1'):
            lok = [groups[name, pair, role, 'lok'] for pair in pairs]
            ort = [groups[name, pair, role, 'ort'] for pair in pairs]
            arms[role] = dict(lok_ms=statistics.median(lok), ort_ms=statistics.median(ort),
                lok_ort=statistics.median(a / b for a, b in zip(lok, ort)),
                lok_process_medians_ms=lok, ort_process_medians_ms=ort,
                lok_spread=(max(lok) - min(lok)) / statistics.median(lok),
                ort_spread=(max(ort) - min(ort)) / statistics.median(ort))
        ratios = [groups[name, pair, 'L1', 'lok'] / groups[name, pair, 'L0', 'lok'] for pair in pairs]
        result[name] = dict(arms=arms, paired_candidate_control=ratios,
            candidate_control=statistics.median(ratios))
    return result


def run(args):
    output, prepared = args.output.resolve(), args.prepared.resolve()
    common.ROOT = args.root.resolve()
    order = visits(args.pairs)
    cases = args.case or ['e5-30tok' if args.model == 'e5' else 'gemma2-128tok']
    evidence.require(len(cases) == len(set(cases)) and all(name in (isolated.CASES if args.model == 'e5' else gemma.CASES) for name in cases), 'invalid/repeated case')
    evidence.require(0 <= args.cpu < 64, 'invalid CPU')
    runner = common.RUNNER if args.model == 'e5' else 'Lokad.Onnx.EmbeddingGemma2Bench.dll'
    builds = common.stage(prepared, output, 'comparison', runner)
    env = dict(os.environ)
    for key in list(env):
        if key.lower() in ('dotnet_tieredcompilation', 'complus_tieredcompilation'):
            del env[key]
    env['DOTNET_TieredCompilation'] = '1'
    records = []
    fixtures = {}
    if args.model == 'gemma':
        evidence.require(args.fixtures is not None, '--fixtures is required for Gemma')
        source = args.fixtures.resolve().parent
        manifest = gemma.fixture(source)
        evidence.require(not output.is_relative_to(source), 'output must be outside fixtures')
        shutil.copytree(source, output / 'fixtures')
    else:
        for name in cases:
            fixture = output / (name + '-oracle')
            entry, oracle = e5.child_conditioned(builds['L0'], output, 'L0', 0, 'oracle', name, fixture, args.cpu, False, env)
            isolated.validate_fixture_for_protocol(output, isolated.binding(fixture / 'fixture.json', output), oracle, isolated.CONDITIONED_PROTOCOL)
            fixtures[name] = fixture
    signatures = {}
    index = 0
    for pair, roles in enumerate(order, 1):
        for role_index, role in enumerate(roles):
            for name in cases:
                for engine in (('lok', 'ort') if role_index == 0 else ('ort', 'lok')):
                    index += 1
                    if args.model == 'gemma':
                        entry = gemma.child(builds[role], common.ROOT, output, name, engine, args.cpu,
                            args.dimensions, args.packed_weight_budget, False, env, index)
                        record, samples, supervision = gemma.worker(output, entry, False, manifest)
                        evidence.require(record['environment']['sdk'] == builds[role]['sdk'], 'worker/core SDK differs')
                        signature = {key: record[key] for key in ('environment', 'runner_sha256', 'assets', 'input_sha256', 'fixture_sha256', 'dimensions', 'packed_weight_budget')}
                    else:
                        entry, record = e5.child_conditioned(builds[role], output, role, pair, engine,
                            name, fixtures[name], args.cpu, False, env)
                        samples = record['raw']
                        signature = {key: record[key] for key in ('environment', 'runner_sha256', 'model_sha256', 'input_sha256', 'fixture_sha256', 'oracle_native_sha256')}
                    evidence.require(signatures.setdefault(name, signature) == signature, 'worker workload/environment identity changed')
                    records.append(dict(case=name, pair=pair, role=role, engine=engine, samples_ms=samples,
                        source_sha=record['source_sha'], core_sha256=record['core_sha256'], files=entry))
                    # Keep every completed worker even when a later one fails.
                    common.save_json(output / f'completed-{index:03d}.json', records[-1])
    report = dict(kind='engineering-comparison', model=args.model, pairs=args.pairs,
        note='Descriptive paired timings; all calls retained. No calibrated release performance claim.',
        signatures=signatures, workers=records, summary=summarize(records))
    common.save_json(output / 'report.json', report)
    for name, result in report['summary'].items():
        arm = result['arms']['L1']
        print(f"{name}: candidate {arm['lok_ms']:.2f} ms; ORT {arm['ort_ms']:.2f} ms; "
              f"candidate/control {result['candidate_control']:.4f}; process spread {arm['lok_spread']:.1%}")
    print('PASS engineering comparison: ' + str(output / 'report.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('e5', 'gemma'), required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=common.ROOT)
    parser.add_argument('--fixtures', type=Path)
    parser.add_argument('--case', action='append')
    parser.add_argument('--cpu', type=int, required=True)
    parser.add_argument('--pairs', type=int, default=2)
    parser.add_argument('--dimensions', type=int, choices=(128, 256, 512, 768), default=768)
    parser.add_argument('--packed-weight-budget', type=int, default=576 * 1024 * 1024)
    args = parser.parse_args()
    try:
        run(args)
    except (evidence.EvidenceError, OSError, ValueError) as error:
        parser.exit(1, f'Benchmark failed; completed worker files are retained: {error}\n')


if __name__ == '__main__':
    main()
