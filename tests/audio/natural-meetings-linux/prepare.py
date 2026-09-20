"""Prepare a separate Linux native reference, borrowing immutable inputs."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import shutil
import sys
from dependencies import canonical,CANONICAL_BODY,LINUX_DUMP_BODY


def execution_suffix(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    first=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple)
        and [v.id for v in n.targets[0].elts]==['root','base'])
    return canonical(ast.Module(body=main.body[first:],type_ignores=[]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','artifact','original']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();base=a.artifact.resolve();original=a.original.resolve();source=Path(__file__).parent
    assert not (base/'manifest.json').exists() and not (base/'runtime').exists()
    sys.path.insert(0,str(original/'runtime'))
    from common import pin,read,write
    old=read(original/'frozen.json')
    for name,wanted in old['files'].items():assert pin(original/name)==wanted,name
    dependency=read(base/'dependency-check.json');assert dependency['passed'] is True
    assert dependency['portable_function_body_sha256']==CANONICAL_BODY and dependency['borrowed_function_body_sha256']==LINUX_DUMP_BODY
    for name,wanted in dependency['native_files'].items():assert pin(Path(name))==wanted,name
    before=execution_suffix(original/'runtime/native.py');after=execution_suffix(source/'native.py');assert before==after
    runtime=base/'runtime';runtime.mkdir()
    for name in ['native.py','supervise.py']:shutil.copyfile(source/name,runtime/name)
    for name in ['common.py','whisper_recording.py','campaign_processes.py']:shutil.copyfile(original/'runtime'/name,runtime/name)
    for name in ['reference-source','upstream']:shutil.copytree(original/name,base/name)
    manifest=read(original/'manifest.json');manifest['protocol']='natural-meeting-native-linux-v1'
    manifest['families']={'whisper':manifest['families']['whisper']};manifest['schedule']=['whisper']
    manifest['native_files']=dependency['native_files'];manifest['limits']={'native':manifest['limits']['managed']}
    manifest['hosts']={'native':'Linux AMD EPYC 9V74 CPU2'}
    manifest['python_path']=[str(base/'python'),str(root/'artifacts/asr-multilingual-amd-20260920/python')]
    manifest['original_manifest']=dict(path=str(original/'manifest.json'),**pin(original/'manifest.json'))
    manifest['original_frozen']=pin(original/'frozen.json');manifest['dependency_check']=pin(base/'dependency-check.json')
    manifest['portable_execution_suffix_sha256']=hashlib.sha256(json.dumps(after,sort_keys=True).encode()).hexdigest()
    manifest['borrowed_function_body_sha256']=LINUX_DUMP_BODY;manifest['portable_function_body_sha256']=CANONICAL_BODY
    for item in manifest['families']['whisper']['models'].values():assert pin(root/item['path'])=={k:item[k] for k in ['bytes','sha256']}
    for case in manifest['cases']:
        item=case['audio'];assert pin(root/item['path'])=={k:item[k] for k in ['bytes','sha256']}
    write(base/'manifest.json',manifest)
    write(base/'preparation.json',dict(passed=True,manifest=pin(base/'manifest.json'),original_frozen=pin(original/'frozen.json'),
        dependency_check=pin(base/'dependency-check.json'),portable_execution_suffix_sha256=manifest['portable_execution_suffix_sha256'],
        scope='Native execution/serialization suffix identical; only platform assertion and supervisor command/dependency path differ'))
    print('Prepared one Linux Whisper reference; models, PCM and inference bodies unchanged.')


if __name__=='__main__':main()
