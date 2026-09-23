"""Bind original fallback body with first-use optimization and the unchanged M47 copy helper."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-pad-first-use-build-amd-20260923'
SOURCE = ROOT / 'artifacts/parakeet-pad-first-use-source-20260923'
sys.path.insert(0, str(ROOT / 'tests/parakeet/pad-first-use-build-amd'))
from checks import HELPER, PAD, PADCORE, inventory
from protocol import pin, read


def main():
    proof = read(BASE / 'closed.json'); assert proof['passed']
    for name, wanted in proof['files'].items(): assert pin(BASE / name) == wanted, name
    assert pin(SOURCE / 'prepared.json')['sha256'] == 'e82940ee6dba61226f51989167fcd7e825b582e9e2c5a234160dd7e644070a94'
    analysis = read(BASE / 'analysis.json')
    assert analysis['source_prepared'] == pin(SOURCE / 'prepared.json')
    instructions = read(BASE / 'collected/inventory/instructions.json')
    result = inventory(instructions, analysis['measured'], analysis['built'])
    assert result == analysis['inventory'] and result['public_pad']['instructions'] == 463
    row = instructions['observations'][0]
    assert PADCORE not in row['differences'] and row['method_flags_after'][PADCORE] == 512 and row['method_flags_before'][PADCORE] == 0
    prior = ROOT / 'artifacts/parakeet-pad-dispatch-build-amd-20260923'
    assert pin(prior / 'closed.json')['sha256'] == '7fbee9acdcedfa3e86ad32b573c36a5bbd493be050d798509b0afd1acc80c9db'
    prior_inventory = prior / 'collected/inventory/instructions.json'
    assert pin(prior_inventory) == read(prior / 'closed.json')['files']['collected/inventory/instructions.json']
    old = read(prior_inventory)['observations'][0]
    assert row['candidate_methods'][HELPER] == old['candidate_methods'][HELPER]
    assert row['normalized_methods'][PADCORE] == old['normalized_methods'][PADCORE]
    helper = json.loads(row['candidate_methods'][HELPER])
    assert helper['exceptions'] == []
    calls = [r['operand'] for r in helper['instructions'] if r['opcode'].startswith('call')]
    fallback = 'Lokad.Onnx.CPUExecutionProvider::Lokad.Onnx.DenseTensor`1[T] PadCore[T](Lokad.Onnx.Tensor`1[T], Int32[], Int32[], T, Boolean)'
    assert calls.count(fallback) == 2
    assert calls.count('Lokad.Onnx.Tensor`1[T]::Lokad.Onnx.DenseTensor`1[T] ToDenseTensor()') == 1
    assert calls.count('Lokad.Onnx.DenseTensor`1[T]::Lokad.Onnx.DenseTensor`1[T] OfShape(Int32[])') == 1
    assert calls.count('System.Span`1[T]::Void Fill(T)') == 1
    assert calls.count('System.Span`1[T]::Void CopyTo(System.Span`1[T])') == 1
    assert calls.count('System.Span`1[T]::System.Span`1[T] Slice(Int32, Int32)') == 2
    assert sum(r['opcode'] == 'div' for r in helper['instructions']) == 1
    assert all(r['opcode'] not in ['rem', 'cpblk', 'initblk', 'localloc'] for r in helper['instructions'])
    result.pop('helper_source_review_pending')
    value = dict(**result, build=pin(BASE / 'closed.json'),
        inventory=pin(BASE / 'collected/inventory/instructions.json'), source=pin(SOURCE / 'prepared.json'),
        candidate=analysis['built'], m47_helper_body_exact=True, padcore_implementation_flag=512, helper_instructions=len(helper['instructions']), helper_calls=calls,
        helper_review='Eligibility is resolved before materialization; two original fallback calls, one materialization and one owned filled allocation on the copy path. Original data dimensions and flat source mapping preserved. Empty destination/width return before division; row offsets are bounded by validated destination length. No unsafe or floating arithmetic added.',
        generator=pin(Path(__file__)), root_product_changed=False, model_qualification_pending=True, performance_pending=True)
    with (OUT / 'composition-20260923.json').open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2); f.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
