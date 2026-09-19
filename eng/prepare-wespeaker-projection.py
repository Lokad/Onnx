"""Prepare the pinned local WeSpeaker affine layer as ordinary ONNX (no download)."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def prepare(model_directory: Path, output: Path):
    if output.exists() or output.with_suffix(output.suffix + '.json').exists():
        raise FileExistsError('Choose a new projection output path.')
    pins = {
        'resnet_seg_1_weight.npy': ('ca91250bb69bea25bdc7c710e253a74450a415b3da587e53e04fd5a01abbe4da', (256, 5120)),
        'resnet_seg_1_bias.npy': ('51fcb6d0530993ad044a797310f4bfd6af266af0dbf364f6bc0008fdd63520cd', (256,)),
    }
    arrays = []
    for name, (digest, shape) in pins.items():
        path = model_directory / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned asset digest mismatch: ' + name)
        value = np.load(path, allow_pickle=False)
        if value.dtype != np.dtype('float32') or value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Invalid projection tensor: ' + name)
        arrays.append(value)
    graph = helper.make_graph(
        [helper.make_node('Gemm', ['pooled', 'weight', 'bias'], ['embedding'], transB=1)], 'WeSpeaker projection',
        [helper.make_tensor_value_info('pooled', TensorProto.FLOAT, ['batch', 5120])],
        [helper.make_tensor_value_info('embedding', TensorProto.FLOAT, ['batch', 256])],
        [numpy_helper.from_array(arrays[0], name='weight'), numpy_helper.from_array(arrays[1], name='bias')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=9)
    onnx.checker.check_model(model)
    data = model.SerializeToString(deterministic=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(data)
    record = dict(source_repository='welcomyou/pyannote-community-1-onnx-split',
        source_revision='cde44c2db938c8abb755853b9a87cb3179c47803', license='CC-BY-4.0',
        inputs={name: digest for name, (digest, _) in pins.items()},
        output_sha256=hashlib.sha256(data).hexdigest(), output_bytes=len(data), numpy=np.__version__, onnx=onnx.__version__)
    with output.with_suffix(output.suffix + '.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.model_directory, args.output), indent=2))
