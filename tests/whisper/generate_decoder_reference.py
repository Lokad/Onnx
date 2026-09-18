"""Pinned Whisper Turbo decoder oracle. Synthetic hidden states, not transcription.

Native ORT generates its own caches; the managed replay must advance its own
outputs using the recorded links. Numeric oracle dependencies are development only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import onnxruntime as ort

REVISION = '360ebcde2559d60bb474678be3c1de9ef347d01a'
GENERATION_CONFIG_SHA256 = '16f95291d2f47c944d3c2b19390bba7965666555c1ea2a0bdc850d1fab45612f'
assert (np.__version__, ort.__version__) == ('2.2.4', '1.29.0')


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.add_session_config_entry('session.inter_op.allow_spinning', '0')
    models = {}
    sessions = {}
    # These file hashes are from the Hub's content records at the pinned revision.
    expected = json.loads((Path(__file__).parent / 'decoder-assets.json').read_text())
    for key, asset in expected.items():
        path = args.models / asset['path']
        assert path.stat().st_size == asset['bytes'] and sha(path) == asset['sha256'], key
        models[key] = asset
        sessions[key] = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    files = {}

    def save(relative, value):
        path = args.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, value, allow_pickle=False)
        files[relative] = dict(sha256=sha(path), shape=list(value.shape), dtype=str(value.dtype), bytes=path.stat().st_size)
        return relative

    configuration = args.models / 'generation_config.json'
    assert sha(configuration) == GENERATION_CONFIG_SHA256, 'generation configuration differs from the pinned export'
    generation = json.loads(configuration.read_text())
    prefix = [generation['decoder_start_token_id'], generation['lang_to_id']['<|en|>'],
              generation['task_to_id']['transcribe'], generation['no_timestamps_token_id']]
    scenarios = []
    for name, encoder_length, tokens in [('short-prefix1', 7, prefix[:1]), ('full-prefix4', 1500, prefix)]:
        rng = np.random.default_rng(20260918 + encoder_length)
        hidden = rng.normal(0, .05, (1, encoder_length, 1280)).astype(np.float32)
        initial = {'input_ids': np.array([tokens], np.int64), 'encoder_hidden_states': hidden}
        steps = []
        native_values = []
        for step_index in range(5):
            key = 'first' if step_index == 0 else 'past'
            session = sessions[key]
            inputs = {}
            bindings = {}
            if step_index == 0:
                for input_name, value in initial.items():
                    inputs[input_name] = value
                    bindings[input_name] = dict(file=save(f'{name}/step0-input-{input_name}.npy', value))
            else:
                # Fixed, valid vocabulary IDs isolate arithmetic/cache behavior
                # from greedy token selection, which is a separate qualification.
                inputs['input_ids'] = np.array([[100 + step_index * 731]], np.int64)
                bindings['input_ids'] = dict(file=save(f'{name}/step{step_index}-input-ids.npy', inputs['input_ids']))
                for descriptor in session.get_inputs():
                    input_name = descriptor.name
                    if not input_name.startswith('past_key_values.'): continue
                    output_name = input_name.replace('past_key_values.', 'present.', 1)
                    source_step = 0 if '.encoder.' in input_name else step_index - 1
                    inputs[input_name] = native_values[source_step][output_name]
                    bindings[input_name] = dict(step=source_step, output=output_name)
            assert set(inputs) == {i.name for i in session.get_inputs()}
            before = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in inputs.items()}
            started = time.perf_counter()
            result = dict(zip([o.name for o in session.get_outputs()], session.run(None, inputs)))
            seconds = time.perf_counter() - started
            assert before == {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in inputs.items()}
            assert all(np.isfinite(v).all() for v in result.values())
            outputs = {k: save(f'{name}/step{step_index}-output-{k}.npy', v) for k, v in result.items()}
            steps.append(dict(model=key, inputs=bindings, outputs=outputs, oracle_seconds=seconds))
            native_values.append(result)
            print(name, step_index, {k: list(v.shape) for k, v in result.items()}, flush=True)
        scenarios.append(dict(name=name, encoder_length=encoder_length, prefix=tokens, steps=steps))
    manifest = dict(schema=1, description='Synthetic hidden-state decoder component qualification; not audio preprocessing or transcription',
        repo='onnx-community/whisper-large-v3-turbo', revision=REVISION, models=models, files=files, scenarios=scenarios,
        provenance=dict(numpy=np.__version__, onnxruntime=ort.__version__, provider='CPUExecutionProvider', threads=1,
                        execution='sequential', optimization='all', seed=20260918, generator_sha256=sha(Path(__file__)),
                        generation_config_sha256=GENERATION_CONFIG_SHA256),
        scaled_absolute_tolerance=1e-4)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__': main()
