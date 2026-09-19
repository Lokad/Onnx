"""Complete native projected-vector references and explicit public mask policy; no downloads."""
from pathlib import Path
from functools import partial
import argparse, ast, hashlib, importlib.util, inspect, json, types
import numpy as np
import onnxruntime as ort
import torch
import torchaudio
from torchaudio.compliance import kaldi

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model-directory', type=Path, required=True)
parser.add_argument('--reference-source', type=Path, required=True)
parser.add_argument('--frontend-reference', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
out = args.output
out.mkdir(parents=True, exist_ok=False)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
lfsha = lambda p: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
here = Path(__file__).resolve().parent
pins = json.loads((here / 'pins.json').read_text(encoding='utf-8'))
for name, version in [('numpy', np.__version__), ('torch', torch.__version__), ('torchaudio', torchaudio.__version__), ('onnxruntime', ort.__version__)]:
    assert version == pins[name], name
source = args.reference_source / 'src/pyannote/audio/models/embedding/wespeaker/__init__.py'
pool_source = args.reference_source / 'src/pyannote/audio/models/blocks/pooling.py'
assert lfsha(source) == pins['frontend_source_lf_sha256']
assert lfsha(pool_source) == pins['pooling_source_lf_sha256']
assert lfsha(Path(inspect.getfile(kaldi))) == pins['kaldi_source_lf_sha256']
encoder = args.model_directory / 'embedding_encoder.onnx'
assert sha(encoder) == pins['encoder_sha256']
prepare_path = here.parents[2] / 'eng/prepare-wespeaker-projection.py'
spec = importlib.util.spec_from_file_location('prepare_projection', prepare_path)
prepare = importlib.util.module_from_spec(spec); spec.loader.exec_module(prepare)
projection = prepare.prepare(args.model_directory, out / 'projection.onnx')
torch.set_num_threads(1); torch.set_num_interop_threads(1)
method = next(n for n in ast.walk(ast.parse(source.read_text(encoding='utf-8'))) if isinstance(n, ast.FunctionDef) and n.name == 'compute_fbank')
module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
namespace = {'torch': torch}; exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
settings = dict(num_mel_bins=80, frame_length=25.0, frame_shift=10.0, round_to_power_of_two=True, snip_edges=True,
    dither=0.0, sample_frequency=16000, window_type='hamming', use_energy=False)
shim = types.SimpleNamespace(hparams=types.SimpleNamespace(fbank_centering_span=None), _fbank=partial(kaldi.fbank, **settings))
pool_namespace = {}; exec(compile(pool_source.read_text(encoding='utf-8'), str(pool_source), 'exec'), pool_namespace)
pool = pool_namespace['StatsPool']()
w = torch.from_numpy(np.load(args.model_directory / 'resnet_seg_1_weight.npy', allow_pickle=False))
b = torch.from_numpy(np.load(args.model_directory / 'resnet_seg_1_bias.npy', allow_pickle=False))
options = ort.SessionOptions(); options.intra_op_num_threads = options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL; options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
options.add_session_config_entry('session.intra_op.allow_spinning', '0'); options.add_session_config_entry('session.inter_op.allow_spinning', '0')
session = ort.InferenceSession(str(encoder), options, providers=['CPUExecutionProvider'])
affine = ort.InferenceSession(str(out / 'projection.onnx'), options, providers=['CPUExecutionProvider'])
frontend = json.loads((args.frontend_reference / 'manifest.json').read_text(encoding='utf-8'))
assert frontend['generator_lf_sha256'] == lfsha(here.parent / 'frontend/generate_reference.py')
assert frontend['pins_lf_sha256'] == lfsha(here.parent / 'frontend/pins.json')
files = {}; cases = []

def save(name, array):
    a = np.ascontiguousarray(array, np.float32)
    assert np.isfinite(a).all(), name
    path = out / (name + '.npy'); np.save(path, a, allow_pickle=False)
    files[path.name] = dict(sha256=sha(path), shape=list(a.shape), dtype='float32', bytes=path.stat().st_size)
    return path.name

for name in pins['recordings']:
    parent_name = 'noise-16000' if name.startswith('short-') else name
    original = next(c for c in frontend['cases'] if c['name'] == parent_name)
    path = args.frontend_reference / original['input']
    assert sha(path) == frontend['files'][original['input']]['sha256']
    pcm = np.load(path, allow_pickle=False).reshape(-1)
    if name.startswith('short-'): pcm = pcm[:int(name.split('-')[1])].copy()
    features = namespace['compute_fbank'](shim, torch.from_numpy(pcm.reshape(1, 1, -1))).numpy()
    frames = (features.shape[1] + 7) // 8
    encoded = None if frames < 2 else session.run(None, {'fbank_features': features})[0]
    if encoded is not None:
        assert encoded.shape == (1, 2560, frames)
        assert np.array_equal(encoded, session.run(None, {'fbank_features': features})[0])
    pcm_file = save(name + '-pcm', pcm); length = features.shape[1] + 3
    masks = [None, (np.arange(length) % 7 < 3).astype(np.float32), ((np.arange(length) % 11 + 1) / 11).astype(np.float32),
        np.zeros(length, np.float32), np.eye(1, frames, dtype=np.float32).reshape(-1)]
    for kind, mask in zip(pins['masks'], masks):
        weights = np.ones(frames, np.float32) if mask is None else mask[np.arange(frames) * len(mask) // frames]
        positive = int(np.count_nonzero(weights > 0)); valid = positive >= 2
        vector_file = None; error = None
        if valid:
            tensor_mask = None if mask is None else torch.from_numpy(mask.reshape(1, -1))
            pooled = pool(torch.from_numpy(encoded), tensor_mask)
            vector = torch.nn.functional.linear(pooled, w, b).numpy()
            assert np.array_equal(vector, torch.nn.functional.linear(pool(torch.from_numpy(encoded), tensor_mask), w, b).numpy())
            projected = affine.run(None, {'pooled': pooled.numpy()})[0]
            error = float((np.abs(projected.astype(np.float64) - vector) / np.maximum(1, np.abs(vector.astype(np.float64)))).max())
            assert error <= pins['tolerance']
            vector_file = save(name + '-' + kind + '-vector', vector)
        cases.append(dict(name=name + '-' + kind, recording=name, mask_kind=kind, pcm=pcm_file,
            mask=None if mask is None else save(name + '-' + kind + '-mask', mask), vector=vector_file,
            status='Completed' if valid else 'InsufficientFrames', frames=frames, positive=positive, native_affine_error=error))
    print(name, frames, flush=True)
record = dict(scope='Public short-recording WeSpeaker vector and insufficient-frame policy, not diarization',
    pins=pins, pins_lf_sha256=lfsha(here/'pins.json'), generator_lf_sha256=lfsha(Path(__file__)),
    prepare_lf_sha256=lfsha(prepare_path), projection=projection, frontend_manifest_sha256=sha(args.frontend_reference/'manifest.json'),
    cases=cases, files=files)
(out/'manifest.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print('Complete', len(cases), 'cases', sum(c['vector'] is not None for c in cases), 'vectors')
