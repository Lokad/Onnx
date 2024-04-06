import os
from typing import Dict
import urllib.request

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

from interop import backend

from torch import Tensor
from transformers import AutoTokenizer, AutoModel

file_dir = os.path.dirname(os.path.realpath(__file__))
onnx_model_file = os.path.join(file_dir, 'me5s.model.onnx')
if not os.path.isfile(onnx_model_file):
    print (f'Downloading multilingual embedded-small 5 ONNX model file to {onnx_model_file}...')
    dl,_ = urllib.request.urlretrieve("https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model.onnx?download=true", onnx_model_file)
    if dl != onnx_model_file: 
        raise RuntimeError(f'Could not download model file from https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model.onnx.')
    else:
        print('download complete.')

tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')

def test_model_file_prepare():
    rep = backend.prepare_file(onnx_model_file)
    assert rep.graph.Nodes.Count == 1243
    assert rep.graph.Inputs.Count == 3

def test_tokenizer():
    input_texts = ['Hello world']
    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=512, padding=False, truncation=True, return_tensors='pt')
    t = backend.get_input_ndarray_from_text('Hello world', 'me5s')
    np.testing.assert_equal(batch_dict['input_ids'].numpy(), t[0])
    np.testing.assert_equal(batch_dict['attention_mask'].numpy(), t[1])

def test_model_run():
    input_texts = ['Hello world']
    batch_dict = tokenizer(['Hello world'], max_length=512, padding=False, truncation=True, return_tensors='pt')
    outputs = model(**batch_dict)
    t = backend.get_input_ndarray_from_text('Hello world', 'me5s')
    rep = backend.prepare_file(onnx_model_file)
    r = rep.run(t)
    np.testing.assert_allclose(outputs.last_hidden_state.detach().numpy(), r[0], rtol=1e-6, atol=1e-6)

if __name__ == '__main__':
    test_model_file_prepare()
    test_tokenizer()
    test_model_run()