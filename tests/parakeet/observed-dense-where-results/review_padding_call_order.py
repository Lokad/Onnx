"""Reconcile actual Pad chronology and predict fallback call counts, without inference."""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-observed-padding-call-order-20260924'
PROFILE = ROOT/'artifacts/parakeet-observed-dense-where-profile-resume-amd-20260924'
INITIAL = ROOT/'artifacts/parakeet-observed-dense-where-profile-amd-20260924'
ROUTES = ROOT/'artifacts/parakeet-observed-padding-routes-20260924'
APP = ROOT/'artifacts/parakeet-observed-dense-where-app-amd-20260924'
SOURCE = ROOT/'artifacts/parakeet-observed-dense-where-source-20260924'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def main():
    assert not BASE.exists()
    assert pin(PROFILE/'closed.json')['sha256'] == '8a6f509a210641650a9c057f472417dde7b4acc320f86eeb3d34ecbced4cc2e6'
    proof = read(PROFILE/'closed.json')
    assert proof['passed'] and proof['analysis'] == pin(PROFILE/'analysis.json')
    phases = read(PROFILE/'analysis.json')['phases']
    assert pin(ROUTES/'closed.json')['sha256'] == 'eedc796f9c2982a969212e486a9e83913877320688762b90cfe8ae6b73de87ef'
    rp = read(ROUTES/'closed.json')
    assert rp['passed'] and rp['analysis'] == pin(ROUTES/'analysis.json')
    routes = read(ROUTES/'analysis.json')
    assert routes['selected_layout_observations'] == 3840
    assert all(r['eligible'] for r in routes['observations'])
    shape = 'src/Lokad.Onnx/CPUExecutionProvider.Shape.cs'
    prepared = read(SOURCE/'prepared.json')
    assert pin(ROOT/shape) == prepared['source'][shape] == prepared['before'][shape]
    # Four dtype arms, each making one call; no dispatcher in the measured product.
    source = (ROOT/shape).read_text(encoding='utf8')
    assert source.count('return Success(op, PadCore(') == 4 and 'PadDispatch' not in source
    manifest_path = APP/'collected/manifests/current-parakeet.json'
    app_proof = read(APP/'closed.json')
    assert app_proof['passed'] and app_proof['admitted']
    assert app_proof['files']['collected/manifests/current-parakeet.json'] == pin(manifest_path)
    manifest = read(manifest_path)
    names = [c['name'] for c in manifest['cases']]
    assert len(names) == 20 and (manifest['warmup_passes'], manifest['measured_passes']) == (1, 3)
    rows = []
    inputs = {}
    orders = []
    for role, campaign, leg, collection_pin in [
        ('selected', INITIAL, 'control', proof['initial']['collection']),
        ('candidate', PROFILE, 'wall', proof['collection']),
    ]:
        folder = campaign/'capture-collected'
        assert pin(folder/'capture-collection.json') == collection_pin
        receipt = read(folder/'capture-collection.json')
        assert receipt['terminal'] and receipt['code'] == (1 if role == 'selected' else 0)
        inputs[role] = collection_pin

        def checked(relative):
            path = folder/relative
            assert pin(path) == receipt['files'][relative], relative
            return read(path)

        result = checked(leg+'/result.json')
        metadata = checked(leg+'/graphs.json')
        assert result['passed'] and len(result['records']) == 80
        nodes = {g:{n['id']:n for n in v['nodes']} for g,v in metadata.items()}
        expected_order = [(g,n['name']) for g,v in metadata.items() for n in v['nodes'] if n['op'] == 'Pad']
        assert len(expected_order) == 49 and expected_order[0] == ('nemo128.onnx','n6_2')
        assert all(g == 'encoder-model.onnx' for g,_ in expected_order[1:])
        orders.append(expected_order)
        frontend_measured = 0
        previous = 0
        for index, row in enumerate(result['records']):
            assert (row['name'],row['pass'],row['phase']) == (
                names[index % 20],index//20,'warmup' if index < 20 else 'measured')
            assert previous < row['start_ticks'] < row['end_ticks']
            previous = row['end_ticks']
            value = checked(f'{leg}/phase-{index:03}.json')
            assert (value['name'],value['pass'],value['frequency'],value['mode']) == (
                row['name'],row['pass'],row['frequency'],'wall')
            assert [c['graph'] for c in value['calls']] == [
                'nemo128.onnx','encoder-model.onnx',
                *(['decoder_joint-model.onnx']*row['result']['decoder_calls'])]
            observed = []
            last = row['start_ticks']
            for call in value['calls']:
                graph = call['graph']
                assert last <= call['start_ticks'] < call['end_ticks'] <= row['end_ticks']
                last = call['start_ticks']
                assert [n['NodeId'] for n in call['nodes']] == [n['id'] for n in metadata[graph]['nodes']]
                for node in call['nodes']:
                    assert last <= node['StartTicks'] <= node['EndTicks'] <= call['end_ticks']
                    last = node['EndTicks']
                    n = nodes[graph][node['NodeId']]
                    if n['op'] == 'Pad':
                        observed.append((graph,n['name'],node['EndTicks']-node['StartTicks']))
                last = call['end_ticks']
            assert [(g,n) for g,n,_ in observed] == expected_order
            front_ticks = observed[0][2]
            if index >= 20:
                frontend_measured += front_ticks
            rows.append(dict(role=role,request=index,name=row['name'],iteration=row['pass'],phase=row['phase'],
                frontend_ticks=front_ticks,frequency=row['frequency'],frontend_seconds=front_ticks/row['frequency'],
                observed_pad_calls=49,
                source_predicted_original_padcore_calls_before_frontend=49*index,
                source_predicted_dispatcher_padcore_calls_before_frontend=index))
        expected_front, = [n for n in phases[role]['node_rows'] if n['graph'] == 'frontend' and n['op'] == 'Pad']
        assert frontend_measured == expected_front['ticks']
        assert math.isclose(frontend_measured/(3*row['frequency']),routes['frontend_profile_seconds'][role],rel_tol=1e-14)
    assert orders[0] == orders[1] and len(rows) == 160

    path = ROOT/'tests/parakeet/pad-runtime-diagnostic-amd/census.py'
    spec = importlib.util.spec_from_file_location('retained_pad_census',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = module.census()['cases']
    assert len(cases) == 12
    original = fallback = 0
    screen = []
    for case in cases:
        rank = len(case['shape']); pads = case['pads']
        # The synthetic source tensors are ordinary dense row-major tensors.
        eligible = (case['mode'] != 'reflect' and pads[rank-1] >= 0 and pads[2*rank-1] >= 0
            and all(pads[d] == pads[rank+d] == 0 for d in range(rank-1)))
        screen.append(dict(name=case['name'],shape=case['shape'],pads=pads,eligible=eligible,
            original_calls_before_case=original,dispatcher_fallback_calls_before_case=fallback,
            original_calls_before_measurement=original+600,
            dispatcher_fallback_calls_before_measurement=fallback+(0 if eligible else 600)))
        original += 780
        fallback += 0 if eligible else 780
    reflection, = [r for r in screen if r['name'] == 'reflection']
    assert (reflection['original_calls_before_measurement'],reflection['dispatcher_fallback_calls_before_measurement']) == (9180,2160)
    result = dict(passed=True,profile=pin(PROFILE/'closed.json'),routes=pin(ROUTES/'closed.json'),
        collections=inputs,manifest=pin(manifest_path),managed_pad_source=pin(ROOT/shape),
        original_census=pin(path),requests=160,observed_pad_calls=7840,order=orders[0],
        rows=rows,synthetic_case_order=screen,
        first_measured_application_frontend=dict(request=20,original_padcore_calls_before=980,
            dispatcher_fallback_calls_before=20),
        new_inference_calls=0,source_predicted_dispatcher_counts=True,
        dispatcher_executed_on_current_product=False,jit_state_observed=False,new_optimization_selected=False)
    paths = [OUT/f'padding-call-order-20260924{suffix}' for suffix in ['.md','.json','.csv']]
    assert not any(p.exists() for p in paths)
    BASE.mkdir()
    (BASE/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    (BASE/'closed.json').write_text(json.dumps(dict(passed=True,analysis=pin(BASE/'analysis.json'),
        analyzer=pin(Path(__file__)),new_inference_calls=0),indent=2)+'\n',encoding='utf8')
    with paths[1].open('x',encoding='utf8') as stream:
        json.dump(dict(closure=pin(BASE/'closed.json'),**result),stream,indent=2)
    with paths[2].open('x',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    document = '''# Actual Parakeet padding call order

All 160 retained application requests run frontend reflection first, then all
48 encoder pads. Every one of the 7,840 Pad intervals is accounted for, including
warmup; decoder calls contain no Pad. Twenty complete clips precede the first
measured pass, with the same order in the selected and masking-candidate profiles.

That order predicts a substantial change in fallback warmup if the retained
row-copy dispatcher were used: the first measured frontend would follow **20**
PadCore calls instead of **980**. In the old synthetic screen, reflection's first
measurement instead follows **2,160** dispatcher-fallback calls, or **9,180**
original PadCore calls. Its reflection tensor is also a different case,
`[1,64,225]` with four values added at each end, versus the real frontend's
two-axis padding `[0,256,0,256]`.

These dispatcher counts follow source guards and previously captured layouts.
No dispatcher ran on the current product, and the retained wall trace does not
record JIT compilation events. Call counts do not identify a compilation tier
or establish the cause of a timing difference. The existing masking profiles
use the unchanged PadCore in both products.

The next padding diagnostic therefore needs the complete application order,
including all warmup requests and the first measured reflection calls. An
isolated warmed reflection kernel cannot establish that behavior. Preserve the
old screen's failed verdict; no new variant or performance admission follows
this review. Current masking release qualification remains the prerequisite.

[All 160 request clocks and predicted counts](padding-call-order-20260924.csv),
[complete chronology and source identities](padding-call-order-20260924.json),
[actual routes](padding-routes-20260924.md),
[earlier compilation observations](../pad-runtime-diagnostic-results/report-20260923.md).

No inference ran and no product changed. Closure: `CLOSURE`.
'''.replace('CLOSURE',pin(BASE/'closed.json')['sha256'])
    with paths[0].open('x',encoding='utf8') as stream:
        stream.write(document)
    print(json.dumps(dict(passed=True,closed=pin(BASE/'closed.json'),requests=160,
        observed_pad_calls=7840,first_measured=result['first_measured_application_frontend'],
        synthetic_reflection=reflection,new_inference_calls=0)))


if __name__ == '__main__':
    main()
