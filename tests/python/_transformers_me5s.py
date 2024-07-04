# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModel

# Each input text should start with "query: " or "passage: ", even for non-English texts.
# For tasks other than retrieval, you can simply use the "query: " prefix.
input_texts = ["Hello world I'm some text"]
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
print(batch_dict)
outputs = model(**batch_dict)
print(outputs.last_hidden_state)

