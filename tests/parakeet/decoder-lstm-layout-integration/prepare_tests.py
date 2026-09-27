"""Extract the two portable LSTM layout facts without changing their bodies."""
import difflib
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
SOURCE = ROOT/'artifacts/parakeet-decoder-lstm-layout-contracts-v3-amd-20260927'
DIAGNOSIS = ROOT/'artifacts/parakeet-decoder-lstm-layout-scalar-baseline-20260927'
QUALIFIED = ROOT/'artifacts/parakeet-decoder-packed-row-root-amd-20260927'
OUT = ROOT/'artifacts/parakeet-decoder-lstm-layout-integration-tests-20260927'
FACTS = ['BoundariesRetainBitsGuardsAndZeroAllocation', 'ExceptionalValuesRetainPayloadsAndOperandOrder']


def read(path): return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def render(source):
    start = source.index('    const int H = 640, Outputs = 4 * H;')
    stop = source.index('    [Fact]\n    public void EveryCapturedProjectionUsesTheActualPreparedLayout()')
    body = source[start:stop]
    sha, = [line for line in body.splitlines(True) if line.startswith('    static string Sha(')]
    body = body.replace(sha, '')
    flat_start, flat_end = body.index('    static float[] Flat('), body.index('    static float[] Pack(')
    body = body[:flat_start]+body[flat_end:]
    header = '''using System.Runtime.InteropServices;
using System.Security.Cryptography;

namespace Lokad.Onnx.Backend.Tests;

// Same helpers and fact bodies as the validated LSTM layout contracts.
// Captured-model and exact-VM identity checks stay in their original campaign.
public class PreparedLstmProjectionTests
{
'''
    value = header+body+'}\n'
    assert value.count('[Fact]') == 2
    original_facts = source[source.index('    [Fact]'):stop]
    assert value[value.index('    [Fact]'):-2] == original_facts
    assert not any(word in value for word in ['JsonDocument', 'FileStream', 'Environment.', 'ProcessorAffinity'])
    return value


def review():
    inputs = {}
    for folder, name, digest in [
        (SOURCE, 'failed.json', '59f6f738dcd71beec865e46c511a9bd151207a26d2901c259f015b387a27f63f'),
        (DIAGNOSIS, 'closed.json', '63c97822999a74921c2e8a3c64e0af9ec6682a6829ba83de52bba353b93b85ec'),
        (QUALIFIED, 'closed.json', 'd0a78cdd3106d6a72a41303f879bbb2f9ea3bd6a298d778333015f38ccdac246')]:
        assert pin(folder/name)['sha256'] == digest
        inputs[(folder/name).relative_to(ROOT).as_posix()] = pin(folder/name)
    failure = read(SOURCE/'failed.json')
    assert not failure['passed'] and failure['terminal'] and failure['evidence_verified']
    original = SOURCE/'frozen-tools/LstmLayoutContracts.cs'
    stage = SOURCE/'bundle/stage.json'
    for path in [original, stage]:
        assert pin(path) == failure['files'][path.relative_to(SOURCE).as_posix()]
    fixture = TOOLS/'PreparedLstmProjectionTests.cs.txt'
    assert fixture.read_text(encoding='utf8') == render(original.read_text(encoding='utf8'))
    proof = read(DIAGNOSIS/'closed.json'); analysis = read(DIAGNOSIS/'analysis.json')
    assert proof['passed'] and proof['diagnostic_only'] and not proof['original_campaign_passed']
    assert proof['analysis'] == pin(DIAGNOSIS/'analysis.json')
    assert not analysis['contract_regression_found'] and not analysis['original_campaign_passed']
    assert analysis['matched_existing_scalar_cases'] == 172 and analysis['matched_scalar_rejections'] == 18
    assert analysis['total_projection_values'] == 5836800 and analysis['projection_hashes_equal_across_modes']
    assert set(analysis['modes']) == {'normal', 'noavx512', 'scalar'}
    for mode, row in analysis['modes'].items():
        assert row['added_passed'] == 4 and row['projections'] == 760
        assert (row['passed'], row['failed']) == ((158, 18) if mode == 'scalar' else (176, 0))
    policy = 'tests/Lokad.Onnx.Tensors.Tests/NoOptionalParametersTests.cs'
    root_proof = read(QUALIFIED/'closed.json')
    applied = QUALIFIED/'bundle/evidence/root-applied.json'
    assert pin(applied) == root_proof['files']['bundle/evidence/root-applied.json']
    before = read(applied)['source_files']; source = read(stage)['source']
    assert len(before) == 441 and len(source) == 443
    assert pin(ROOT/policy) == before[policy] == source[policy]
    for path in [original, stage, DIAGNOSIS/'analysis.json', applied, fixture, Path(__file__), ROOT/policy]:
        inputs[path.relative_to(ROOT).as_posix()] = pin(path)
    return dict(passed=True, product_unchanged=True, root_applied=False, compiled_or_executed=False,
        source_stage=pin(stage), original=pin(original), fixture=pin(fixture), script=pin(Path(__file__)),
        facts=FACTS, raw_synthetic_cases=61, fact_bodies_unchanged=True,
        original_capture_and_vm_guards_retained=True, original_campaign_passed=False,
        scalar_diagnosis=pin(DIAGNOSIS/'closed.json'), source_policy_test=before[policy], inputs=inputs)


def main():
    assert sys.argv[1:] in [[], ['--prepare']]
    value = review()
    if sys.argv[1:]:
        assert not OUT.exists(); OUT.mkdir()
        original = (SOURCE/'frozen-tools/LstmLayoutContracts.cs').read_text(encoding='utf8')
        fixture = (TOOLS/'PreparedLstmProjectionTests.cs.txt').read_text(encoding='utf8')
        (OUT/'PreparedLstmProjectionTests.cs').write_text(fixture, encoding='utf8', newline='\n')
        patch = ''.join(difflib.unified_diff(original.splitlines(True), fixture.splitlines(True),
            fromfile='closed/LstmLayoutContracts.cs', tofile='PreparedLstmProjectionTests.cs'))
        (OUT/'review.patch').write_text(patch, encoding='utf8', newline='\n')
        with (OUT/'prepared.json').open('x', encoding='utf8', newline='\n') as stream:
            json.dump(dict(**value, patch=pin(OUT/'review.patch')), stream, indent=2); stream.write('\n')
    else:
        assert read(OUT/'prepared.json') == dict(**value, patch=pin(OUT/'review.patch'))
        assert pin(OUT/'PreparedLstmProjectionTests.cs') == value['fixture']
    print(json.dumps({k:value[k] for k in ['passed','facts','raw_synthetic_cases','root_applied','compiled_or_executed']}))


if __name__ == '__main__': main()
