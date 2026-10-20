"""Freeze complete CPU ORT outputs for the local, hash-pinned ResNet model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def generate(model):
    expected = '4f0558b775c85b568012f7174cadda33fb4b4e5a253ce17e5d839f123e0bc7e6'
    if ort.__version__ != '1.29.0' or hashlib.sha256(model.read_bytes()).hexdigest() != expected:
        raise ValueError('Use ORT 1.29.0 and the pinned local ResNet model')
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
    result = dict(schema=1, ort_version=ort.__version__, model_sha256=expected,
                  provider='CPUExecutionProvider', execution='sequential', intra_op=1, inter_op=1,
                  optimization='ORT_ENABLE_ALL', cases={})
    for name in ('constant', 'ramp'):
        values = np.full((1, 3, 224, 224), .5, np.float32) if name == 'constant' else (
            np.arange(3 * 224 * 224, dtype=np.int64) % 256).astype(np.float32).reshape(1, 3, 224, 224) / np.float32(256)
        outputs = [session.run(['output'], {session.get_inputs()[0].name: values})[0] for _ in range(3)]
        assert all(np.array_equal(outputs[0].view(np.uint32), output.view(np.uint32)) for output in outputs[1:])
        output = outputs[0].astype('<f4')
        result['cases'][name] = dict(shape=list(output.shape), sha256=hashlib.sha256(output.tobytes()).hexdigest(),
                                    values=output.reshape(-1).tolist())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('models/resnet50-onnx/model.onnx'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.model)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
