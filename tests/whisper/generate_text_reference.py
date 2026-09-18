"""Freeze byte-level decoding examples from the pinned local Whisper tokenizer."""
from pathlib import Path
import argparse,hashlib,json
import tokenizers

parser=argparse.ArgumentParser()
parser.add_argument('--models',default='models/whisper-large-v3-turbo')
parser.add_argument('--output',required=True)
args=parser.parse_args()
root=Path(args.models);source=root/'tokenizer.json'
raw=json.loads(source.read_text(encoding='utf-8'))
tokenizer=tokenizers.Tokenizer.from_file(str(source))
phrases=[' Hello world.',' Bonjour, je suis prêt.',' L’été à Paris : où êtes-vous ?','  Spaces\nnewlines\tand café.',' 日本語と中文',' العربية',' 😀👍🏽',' naïve façade — 20 €','']
cases=[];used={32,33}
for phrase in phrases:
    ids=tokenizer.encode(phrase,add_special_tokens=False).ids
    cases.append(dict(ids=ids,text=tokenizer.decode(ids,skip_special_tokens=True)))
    used.update(ids)
# Individual UTF-8 bytes must be joined before decoding, including across BPE tokens.
for ids in ([127],[127,102],[32,50257,33]):
    # EOS is a generation terminator; text after it is not part of the request.
    prefix=ids[:ids.index(50257)] if 50257 in ids else ids
    cases.append(dict(ids=list(ids),text=tokenizer.decode(prefix,skip_special_tokens=True)));used.update(prefix)
generation=json.loads((root/'generation_config.json').read_text(encoding='utf-8'))
needed=set(generation['lang_to_id'])|{'<|nospeech|>','<|startoftranscript|>','<|endoftext|>','<|transcribe|>','<|notimestamps|>'}
fixture=dict(tokenizers=tokenizers.__version__,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    tokenizer=dict(model=dict(type='BPE',vocab={k:v for k,v in raw['model']['vocab'].items() if v in used}),decoder=raw['decoder'],
        added_tokens=[dict(id=t['id'],content=t['content']) for t in raw['added_tokens'] if t['content'] in needed]),
    generation=generation,cases=cases)
Path(args.output).write_text(json.dumps(fixture,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(len(cases),'cases',len(used),'text tokens',tokenizers.__version__)
