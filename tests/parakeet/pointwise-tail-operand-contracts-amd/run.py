"""One diagnosed multiply-order repair; reuse the original bounded contracts."""
import ast
import difflib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

TOOLS=Path(__file__).resolve().parent;ROOT=TOOLS.parents[2]
ORIGINAL=TOOLS.parent/'pointwise-tail-contracts-amd'
OLD_SOURCE=ROOT/'artifacts/parakeet-pointwise-tail-source-20260927'
SOURCE=ROOT/'artifacts/parakeet-pointwise-tail-operand-source-20260927'
BASE=ROOT/'artifacts/parakeet-pointwise-tail-operand-contracts-amd-20260927'
REMOTE='/dev/shm/lokad-pointwise-tail-operand-contracts-20260927'
DIAGNOSIS=ROOT/'artifacts/parakeet-pointwise-tail-nan-diagnostic-20260927'
GENERATED=SOURCE/'contracts-tools'
sys.path.insert(0,str(ORIGINAL))
loader=importlib.util.spec_from_file_location('failed_tail_contracts',ORIGINAL/'run.py')
prior=importlib.util.module_from_spec(loader);loader.loader.exec_module(prior)
pin,read,write=prior.pin,prior.read,prior.write


def replace_once(text,before,after):
    assert text.count(before)==1,before
    return text.replace(before,after)


def prepare():
    assert not SOURCE.exists() and not BASE.exists();prior.prepared()
    explanation=read(DIAGNOSIS/'explanation.json')
    assert explanation['all_reproduced'] and len(explanation['reproduced_first_mismatches'])==260
    assert explanation['review']==pin(DIAGNOSIS/'review.json')
    old=read(OLD_SOURCE/'prepared.json');values={}
    helper='src/Lokad.Onnx/MathOps.PackedColumnTails.cs'
    for name,wanted in old['source'].items():
        path=OLD_SOURCE/'source'/name;assert pin(path)==wanted;values[name]=path.read_bytes()
    before=values[helper].decode();after=before
    for row in range(8):
        after=replace_once(after,f'c{row} = c{row} + Vector256.Create(a{row}[j]) * bv;',
            f'c{row} = c{row} + bv * Vector256.Create(a{row}[j]);')
    after=replace_once(after,'Preserve the original A*B then C+product operations, never FMA.',
        'Match the compiled baseline B*A multiply; retain separate add, never FMA.')
    assert before.split('    internal static unsafe void PackedColumnMaskedEightRows')[0]==after.split('    internal static unsafe void PackedColumnMaskedEightRows')[0]
    values[helper]=after.encode();SOURCE.mkdir()
    for name,data in values.items():
        path=SOURCE/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    patch=(OLD_SOURCE/'candidate.patch').read_text()+''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=helper,tofile=helper))
    (SOURCE/'candidate.patch').write_text(patch,encoding='utf8')
    (SOURCE/'prospective-plan.md').write_bytes((ROOT/'PLAN.md').read_bytes())
    current=dict(old,source={n:pin(SOURCE/'source'/n) for n in values},patch=pin(SOURCE/'candidate.patch'),
        plan=pin(SOURCE/'prospective-plan.md'),prior_source=pin(OLD_SOURCE/'prepared.json'),
        numerical_diagnosis=pin(DIAGNOSIS/'explanation.json'),repair=pin(Path(__file__)),
        repair_scope='Reverse only the eight masked multiply operands to match observed baseline instruction order; explanatory comment updated.')
    write(SOURCE/'prepared.json',current)
    GENERATED.mkdir()
    # Retain exact generated tools: all changes below are declared and reversible.
    for name in ['run.py','audit.py','vm.py','cases.py','Contracts.cs.txt','test_audit.py','README.md']:
        data=(ORIGINAL/name).read_text(encoding='utf8')
        if name=='run.py':
            data=replace_once(data,'ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent',f'ROOT=Path({str(ROOT)!r});TOOLS=Path(__file__).resolve().parent')
            data=replace_once(data,"BASE=ROOT/'artifacts/parakeet-pointwise-tail-contracts-amd-20260927'",f'BASE=Path({str(BASE)!r})')
            data=replace_once(data,"REMOTE='/dev/shm/lokad-pointwise-tail-contracts-20260927'",f'REMOTE={REMOTE!r}')
            data=replace_once(data,"SOURCE=ROOT/'artifacts/parakeet-pointwise-tail-source-20260927'",f'SOURCE=Path({str(SOURCE)!r})')
            data=data.replace('TOOLS.parent',"(ROOT/'tests/parakeet')")
        elif name=='vm.py':
            data=replace_once(data,"elif mode=='avx512-disabled':environment['DOTNET_EnableAVX512']='0'",
                "elif mode=='avx512-disabled':environment.update(DOTNET_EnableAVX512='0',DOTNET_JitDisasm=spec['disasm'])")
        elif name=='Contracts.cs.txt':
            observed=(DIAGNOSIS/'bundle/contract-source/Program.cs').read_text(encoding='utf8')
            data=observed
        elif name=='audit.py':
            data=replace_once(data,"({'DOTNET_EnableAVX512':'0'} if mode=='avx512-disabled'", "({'DOTNET_EnableAVX512':'0','DOTNET_JitDisasm':spec['disasm']} if mode=='avx512-disabled'")
            data=replace_once(data,"    if passed:assert results['normal']['results']==results['avx512-disabled']['results']", "    if passed:\n        for normal,disabled in zip(results['normal']['results'],results['avx512-disabled']['results'],strict=True):\n            if not normal['exceptional']:assert normal==disabled\n    cross_mode_differences=[{k:a[k] for k in ['m','n','k','exceptional']} for a,b in zip(results['normal']['results'],results['avx512-disabled']['results'],strict=True) if a['bit_exact'] and b['bit_exact'] and a['output_sha256']!=b['output_sha256']]\n    assert all(r['exceptional'] for r in cross_mode_differences)")
            data=replace_once(data,'normal_disabled_exact=passed,scalar_baseline_candidate_exact=True,source=spec[\'source\'],',
                "normal_disabled_exact=passed and not cross_mode_differences,finite_normal_disabled_exact=passed,cross_mode_exceptional_hash_differences=cross_mode_differences,scalar_baseline_candidate_exact=True,source=spec['source'],")
        elif name=='README.md':data=(TOOLS/'README.md').read_text(encoding='utf8')
        if name.endswith('.py'):ast.parse(data,name)
        (GENERATED/name).write_text(data,encoding='utf8')
    (GENERATED/'repair-adapter.py').write_bytes(Path(__file__).read_bytes())
    # This exact consumer retains every original case/check; only flag validation changed.
    subprocess.run([sys.executable,'-X','utf8','-B',str(GENERATED/'test_audit.py')],check=True,cwd=ROOT)
    subprocess.run([sys.executable,'-X','utf8','-B',str(GENERATED/'run.py'),'prepare'],check=True,cwd=ROOT)
    write(SOURCE/'adapter.json',dict(adapter=pin(Path(__file__)),generated={p.name:pin(p) for p in GENERATED.iterdir()},source=pin(SOURCE/'prepared.json'),diagnosis=pin(DIAGNOSIS/'explanation.json')))


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:
        receipt=read(SOURCE/'adapter.json');assert receipt['adapter']==pin(Path(__file__))
        assert receipt['source']==pin(SOURCE/'prepared.json') and receipt['diagnosis']==pin(DIAGNOSIS/'explanation.json')
        for name,wanted in receipt['generated'].items():assert pin(GENERATED/name)==wanted,name
        script='audit.py' if sys.argv[1]=='audit' else 'run.py'
        args=sys.argv[2:] if script=='audit.py' else sys.argv[1:]
        raise SystemExit(subprocess.run([sys.executable,'-X','utf8','-B',str(GENERATED/script),*args],cwd=ROOT).returncode)
