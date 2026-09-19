"""PCM-to-text Whisper application using pinned NumPy features and CPU ORT graphs."""
from pathlib import Path
import sys
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
import json


def session(path):
    options = ort.SessionOptions(); options.log_severity_level = 4
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    for name in ('session.intra_op.allow_spinning', 'session.inter_op.allow_spinning'):
        options.add_session_config_entry(name, '0')
    value = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    assert value.get_providers() == ['CPUExecutionProvider']
    return value


class Whisper:
    def __init__(self, root, manifest):
        paths = {name: root / info['path'] for name, info in manifest['models'].items()}
        self.sessions = {name: session(paths['onnx/' + file]) for name, file in (
            ('encoder', 'encoder_model.onnx'), ('first', 'decoder_model.onnx'),
            ('past', 'decoder_with_past_model.onnx'))}
        self.outputs = {name: [v.name for v in value.get_outputs()] for name, value in self.sessions.items()}
        self.cache_inputs = [v.name for v in self.sessions['past'].get_inputs() if v.name.startswith('past_key_values.')]
        self.tokenizer = Tokenizer.from_file(str(paths['tokenizer.json']))
        self.generation = json.loads(paths['generation_config.json'].read_text(encoding='utf-8'))
        self.extractor = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000,
            hop_length=160, chunk_length=30, n_fft=400, dither=0.0)
        self.last_features = None

    def graph(self, name, feeds):
        return dict(zip(self.outputs[name], self.sessions[name].run(None, feeds), strict=True))

    def features(self, pcm):
        padded = np.zeros((1, 480000), np.float32)
        padded[0, :len(pcm)] = pcm
        return self.extractor._np_extract_fbank_features(padded, 'cpu')

    @staticmethod
    def log_probability(values, token):
        values = values.astype(np.float64); maximum = values.max()
        return float(values[token] - maximum - np.log(np.exp(values - maximum).sum()))

    def __call__(self, pcm):
        assert pcm.dtype == np.float32 and pcm.ndim == 1 and len(pcm) <= 480000 and np.isfinite(pcm).all()
        if np.count_nonzero(pcm) == 0:
            self.last_features = None
            return dict(text='', token_ids=[], stop_reason='SilentInput', skipped_as_no_speech=True)
        features = self.features(pcm)
        self.last_features = features  # Inspect after the timer; never reused as another request's input.
        hidden = self.graph('encoder', {'input_features': features})['last_hidden_state']
        config = self.generation; eos = config['eos_token_id']
        prefix = [config['decoder_start_token_id'], config['lang_to_id']['<|en|>'],
                  config['task_to_id']['transcribe'], config['no_timestamps_token_id']]
        tokens = []; cross = {}; previous = {}; total = 0.
        for step in range(444):
            feeds = {'input_ids': np.array([prefix if step == 0 else [tokens[-1]]], np.int64)}
            if step == 0:
                feeds['encoder_hidden_states'] = hidden
            else:
                for name in self.cache_inputs:
                    feeds[name] = (cross if '.encoder.' in name else previous)['present.' + name[len('past_key_values.'):]]
            outputs = self.graph('first' if step == 0 else 'past', feeds)
            logits = outputs['logits']; assert np.isfinite(logits).all()
            if step == 0:
                no_speech = float(np.exp(self.log_probability(logits[0, 0], self.tokenizer.token_to_id('<|nospeech|>'))))
            scores = logits[0, -1].copy(); scores[config['suppress_tokens']] = -np.inf
            if step == 0:
                scores[config['begin_suppress_tokens']] = -np.inf
            scores[eos + 1:] = -np.inf
            token = int(scores.argmax()); total += self.log_probability(scores, token); tokens.append(token)
            if token == eos:
                break
            if step == 0:
                cross = outputs
            previous = outputs
        stopped = tokens[-1] == eos
        average = total / (len(tokens) - int(stopped) + 1)
        skipped = no_speech > .6 and average <= -1.
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return dict(text='' if skipped else text, token_ids=tokens,
                    stop_reason='EndToken' if stopped else 'TokenLimit', skipped_as_no_speech=skipped)

