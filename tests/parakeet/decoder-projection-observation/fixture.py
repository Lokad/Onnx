"""Derive one original decoder input from retained qualification, without inference or writes."""
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/parakeet-rational-sigmoid-models-amd-20260927'
CLOSURE = '3d2e01b5aca3f66c434d5d1e8124c2b0d08327bc14dd37409c0e1884619769d8'
CORE = '946ddfb66492c48a0fc6078ecbe1957ac494ff5d9ff0be42259d70e66e3b1f24'
DATA = 'dbe959361209bbc20db9bd566f037f58e01b9cc40d807c5745dbe2f4e1c1aca6'
OUTPUTS = {'outputs': ('Float', [1, 1, 1, 8198]), 'prednet_lengths': ('Int32', [1]),
           'output_states_1': ('Float', [2, 1, 640]), 'output_states_2': ('Float', [2, 1, 640])}


def identity(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def construct(reference, selected, load):
    """The caller supplies closed result documents and an identity-checked byte loader."""
    assert selected['passed'] and selected['application_passed']
    assert selected['core_sha256'] == CORE and selected['data_sha256'] == DATA
    case, = [c for c in reference['cases'] if c['name'] == 'english-16k']
    row, = [r for r in selected['rows'] if r['name'] == case['name']]
    step = case['steps'][0]
    assert (step['frame'], step['target'], step['token'], step['duration']) == (0, 8192, 8192, 3)
    assert set(step['outputs']) == set(OUTPUTS)
    lookup = {(v['label'], v['output']): v for v in row['comparisons']}
    assert len(lookup) == len(row['comparisons']), 'Ambiguous original comparison'
    blobs = {}

    def descriptor(name, dtype, shape, raw):
        assert dtype in ['Float', 'Int32']
        elements = 1
        for dim in shape:
            assert type(dim) is int and dim > 0
            elements *= dim
        assert len(raw) == 4 * elements
        if dtype == 'Float':
            # Reject infinities and all NaNs without converting/re-encoding any float.
            assert all(bits & 0x7f800000 != 0x7f800000 for (bits,) in struct.iter_unpack('<I', raw))
        file = 'fixture/' + name + '.bin'
        assert file not in blobs
        blobs[file] = raw
        return dict(file=file, dtype=dtype, shape=shape, **identity(raw))

    def original(label, name):
        item = lookup[label, name]
        assert item['passed']
        raw = load(item)
        assert identity(raw)['sha256'] == item['sha256'], 'Retained tensor identity'
        return item, raw

    encoder, raw = original('encoder', 'outputs')
    assert encoder['dtype'] == 'Float' and encoder['shape'] == [1, 1024, 74]
    assert len(raw) == 1024 * 74 * 4
    vector = b''.join(raw[(i * 74) * 4:(i * 74 + 1) * 4] for i in range(1024))
    zeros = bytes(2 * 640 * 4)
    assert step['state_input_sha256'] == [identity(zeros)['sha256']] * 2, 'Original zero-state first step'
    feeds = {'encoder_outputs': descriptor('encoder_outputs', 'Float', [1, 1024, 1], vector),
             'targets': descriptor('targets', 'Int32', [1, 1], struct.pack('<i', step['target'])),
             'target_length': descriptor('target_length', 'Int32', [1], struct.pack('<i', 1)),
             'input_states_1': descriptor('input_states_1', 'Float', [2, 1, 640], zeros),
             'input_states_2': descriptor('input_states_2', 'Float', [2, 1, 640], zeros)}
    outputs = {}
    for name, (dtype, shape) in OUTPUTS.items():
        item, data = original('step-0', name)
        assert (item['dtype'], item['shape']) == (dtype, shape), name
        outputs[name] = descriptor('expected-' + name, dtype, shape, data)
    spec = dict(case=case['name'], step=0, frame=step['frame'], token=step['token'], duration=step['duration'],
                inputs=feeds, outputs=outputs, measured_core_sha256=CORE, measured_data_sha256=DATA,
                model='/home/vermorel/Onnx/models/parakeet-tdt-0.6b-v3/decoder_joint-model.onnx',
                model_sha256=reference['assets']['files']['decoder_joint-model.onnx']['sha256'],
                source_encoder=encoder, model_changed=False, new_inference_calls=0)
    assert len(blobs) == 9 and sum(map(len, blobs.values())) < 64 * 1024
    # core_sha256 is deliberately absent. A future campaign must bind the qualified
    # actual root and prove its compiled equivalence before the observer can run.
    return spec, blobs


def inspect():
    closed_raw = (BASE / 'closed.json').read_bytes()
    assert identity(closed_raw)['sha256'] == CLOSURE
    closed = json.loads(closed_raw)
    assert closed['passed']
    evidence = {}

    def read(name):
        path = BASE / name
        assert path.resolve().is_relative_to(BASE.resolve())
        raw = path.read_bytes()
        assert identity(raw) == closed['files'][name], name
        evidence[name] = identity(raw)
        return raw

    reference = json.loads(read('collected/parakeet-reference/manifest.json'))
    selected = json.loads(read('collected/candidate-native-512/result.json'))
    spec, blobs = construct(reference, selected,
        lambda item: read('collected/candidate-native-512/result.json.tensors/' + item['file']))
    spec['provenance'] = dict(closure=identity(closed_raw), files=evidence)
    return spec, blobs


if __name__ == '__main__':
    spec, blobs = inspect()
    print(json.dumps(dict(passed=True, case=spec['case'], step=spec['step'], inputs=spec['inputs'],
                         outputs=spec['outputs'], files=len(blobs), bytes=sum(map(len, blobs.values())),
                         new_inference_calls=0, campaign_prepared=False), indent=2))
