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

def test_model_file_prepare():
    rep = backend.prepare_file(onnx_model_file)
    assert rep.graph.Nodes.Count == 1243
    assert rep.graph.Inputs.Count == 3

def test_tokenizer():
    input_texts = ['Hello world']
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')

    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=512, padding=False, truncation=True, return_tensors='pt')

    t = backend.get_input_ndarray_from_text('Hello world', 'me5s')
    np.testing.assert_equal(batch_dict['input_ids'].numpy(), t[0])
    np.testing.assert_equal(batch_dict['attention_mask'].numpy(), t[1])
