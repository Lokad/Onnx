"""Evidence helpers for the fixed natural-conversation recording comparison."""
from pathlib import Path
import hashlib
import importlib.util
import json

CORE='187de61ad8f034b9b7ad2fb3490358443fa84334204720e81bc3546a31f3c8d4'
DATA='809242b58725c6ae47514cc3908ef59ffafae6be36bb6e2fba20144d9a975af5'
NAMES=['ES2004a','IS1009a','ES2004a-recovery30']


def pin(path):
    with Path(path).open('rb') as stream:
        return dict(bytes=Path(path).stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def limits(host):
    if host=='amd':return dict(rss=14*1024**3,seconds=7200,available=1024**3,preflight=13*1024**3)
    if host=='windows':return dict(rss=20*1024**3,seconds=7200,available=1024**3,preflight=20*1024**3)
    raise ValueError('Unknown host profile')
