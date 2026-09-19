"""Application-only native Parakeet results for a frozen labeled PCM manifest.

The independently carried greedy loop is checked against the pinned upstream
implementation. This deliberately does not replace the complete tensor lane.
"""
from pathlib import Path
import argparse
import collections
import importlib.util
import json
import re
import time
import types
import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('models', 'audio', 'reference-source', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if (np.__version__, ort.__version__) != ('2.2.4', '1.29.0'):
        raise ValueError('Native dependency versions differ')
    helper_path = Path(__file__).parents[2] / 'parakeet/transcribe/generate_reference.py'
    spec = importlib.util.spec_from_file_location('parakeet_tensor_reference', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    sha = helper.sha
    assets_path = helper_path.with_name('assets.json')
    assets = json.loads(assets_path.read_text(encoding='utf-8'))
    for name, pin in assets['files'].items():
        path = args.models / name
        if path.stat().st_size != pin['bytes'] or sha(path) != pin['sha256']:
            raise ValueError('Model differs: ' + name)
    source = args.reference_source / 'src/onnx_asr/asr.py'
    if helper.source_sha(source) != assets['reference']['asr_lf_sha256']:
        raise ValueError('Upstream source differs')
    namespace = dict(np=np, re=re, TimestampedResult=collections.namedtuple('TimestampedResult', 'text timestamps tokens logprobs'))
    upstream_loop = helper.upstream_method(source.read_text(encoding='utf-8'), '_AsrWithTransducerDecoding', '_decoding', namespace)
    upstream_text = helper.upstream_method(source.read_text(encoding='utf-8'), '_AsrWithDecoding', '_decode_tokens', namespace)
    vocab = {}
    for line in (args.models / 'vocab.txt').read_text(encoding='utf-8').splitlines():
        piece, index = line.rsplit(' ', 1)
        vocab[int(index)] = piece.replace('\u2581', ' ')
    if len(vocab) != 8193 or vocab[8192] != '<blk>':
        raise ValueError('Vocabulary differs')
    audio = json.loads(args.audio.read_text(encoding='utf-8'))
    if audio['schema'] != 1 or audio['sample_rate'] != 16000 or len(audio['cases']) != 20:
        raise ValueError('Expected the fixed twenty-case manifest')
    if len({c['name'] for c in audio['cases']}) != 20:
        raise ValueError('Duplicate recording')
    args.output.mkdir(parents=True, exist_ok=False)
    options = ort.SessionOptions()
    options.log_severity_level = 4
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    for setting in ('session.intra_op.allow_spinning', 'session.inter_op.allow_spinning'):
        options.add_session_config_entry(setting, '0')
    sessions = {name: ort.InferenceSession(str(args.models / file), options, providers=['CPUExecutionProvider'])
                for name, file in assets['graphs'].items()}

    def run(model, feeds):
        before = {k: v.tobytes() for k, v in feeds.items()}
        values = sessions[model].run(None, feeds)
        if any(v.tobytes() != before[k] for k, v in feeds.items()) or not all(np.isfinite(v).all() for v in values):
            raise ValueError('Native input mutation or nonfinite output')
        return dict(zip([v.name for v in sessions[model].get_outputs()], values))

    cases = []
    for index, original in enumerate(audio['cases'] + audio['cases'][:1]):
        start = time.perf_counter()
        pcm_path = args.audio.parent / original['pcm']
        if sha(pcm_path) != original['pcm_sha256']:
            raise ValueError('PCM identity differs')
        pcm = np.load(pcm_path, allow_pickle=False)
        if pcm.dtype != np.float32 or pcm.shape != (original['samples'],) or not np.isfinite(pcm).all() or not np.any(pcm):
            raise ValueError('Invalid nonsilent PCM')
        prepared = run('frontend', dict(waveforms=pcm[None, :], waveforms_lens=np.array([len(pcm)], np.int64)))
        encoded = run('encoder', dict(audio_signal=prepared['features'], length=prepared['features_lens']))
        hidden = encoded['outputs']
        frames = int(encoded['encoded_lengths'][0])
        if hidden.shape != (1, 1024, frames) or frames != (len(pcm) // 160 + 8) // 8:
            raise ValueError('Unexpected encoder length')

        def zero():
            return np.zeros((2, 1, 640), np.float32), np.zeros((2, 1, 640), np.float32)

        def decode(previous, states, vector):
            return run('decoder', dict(encoder_outputs=np.ascontiguousarray(vector[None, :, None]),
                targets=np.array([[previous[-1] if previous else 8192]], np.int32), target_length=np.array([1], np.int32),
                input_states_1=states[0], input_states_2=states[1]))

        state = zero()
        tokens, positions, durations, steps = [], [], [], []
        frame = emitted = 0
        while frame < frames and len(tokens) < 4096:
            values = decode(tokens, state, hidden[0, :, frame])
            if values['outputs'].shape != (1, 1, 1, 8198) or not np.array_equal(values['prednet_lengths'], [1]):
                raise ValueError('Decoder contract')
            logits = values['outputs'].reshape(-1)
            token, duration = int(logits[:8193].argmax()), int(logits[8193:].argmax())
            steps.append(dict(frame=frame, target=tokens[-1] if tokens else 8192, token=token, duration=duration))
            if token != 8192:
                tokens.append(token)
                positions.append(frame)
                durations.append(duration)
                state = values['output_states_1'], values['output_states_2']
                emitted += 1
            if duration:
                frame += duration
                emitted = 0
            elif token == 8192 or emitted == 10:
                frame += 1
                emitted = 0

        def reference_decode(previous, states, vector):
            values = decode(previous, states, vector)
            logits = values['outputs'].reshape(-1)
            return logits[:8193], int(logits[8193:].argmax()), (values['output_states_1'], values['output_states_2'])

        shim = types.SimpleNamespace(use_low_precision=False, _blank_idx=8192, _vocab_size=8193, _max_tokens_per_step=10,
            _create_state=zero, _decode=reference_decode, _vocab=vocab,
            DECODE_SPACE_PATTERN=re.compile(r'\A\s|\s\B|(\s)\b'), window_step=.01, _subsampling_factor=8)
        upstream_tokens, upstream_frames, _ = next(upstream_loop(shim, hidden.transpose(0, 2, 1), np.array([frames], np.int64)))
        if tokens != upstream_tokens[:len(tokens)] or positions != upstream_frames[:len(tokens)] or (frame >= frames and tokens != upstream_tokens):
            raise ValueError('Independent upstream trajectory differs')
        expected = dict(text=upstream_text(shim, tokens, positions, None).text, token_ids=tokens,
            frame_indices=positions, duration_frames=durations, stop_reason='EndOfAudio' if frame >= frames else 'TokenLimit',
            encoded_frames=frames, decoder_calls=len(steps))
        cases.append(dict(**original, repeat=index == 20, expected=expected, steps=steps,
                          upstream_crosscheck=True, seconds=time.perf_counter() - start))
        print(original['name'], 'repeat' if index == 20 else '', json.dumps(expected, ensure_ascii=True), flush=True)
        (args.output / 'progress.json').write_text(json.dumps(cases, indent=2) + '\n', encoding='utf-8')
    if cases[0]['expected'] != cases[-1]['expected']:
        raise ValueError('Repeated native request differs')
    result = dict(schema=1, scope='labeled-parakeet-application-only', assets=assets, audio_manifest_sha256=sha(args.audio),
        generator_sha256=sha(Path(__file__)), helper_sha256=sha(helper_path), reference_raw_sha256=sha(source),
        numpy=np.__version__, onnxruntime=ort.__version__, native_settings=dict(provider='CPUExecutionProvider', threads=1,
        execution='sequential', optimization='all', spinning=False), cases=cases)
    (args.output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
