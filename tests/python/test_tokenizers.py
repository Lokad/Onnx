import numpy as np

from interop import backend

from torch import Tensor
from transformers import AutoTokenizer

def test_me5s():
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
    input_texts = ['Hello world']
    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=512, padding=False, truncation=True, return_tensors='pt')
    t = backend.get_input_ndarray_from_text('Hello world', 'me5s')
    np.testing.assert_equal(batch_dict['input_ids'].numpy(), t[0])
    np.testing.assert_equal(batch_dict['attention_mask'].numpy(), t[1])
