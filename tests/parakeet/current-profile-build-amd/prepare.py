"""Freeze the qualified product and diagnostic source with two identity changes."""
import ast
import difflib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save


ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-current-profile-build-amd-20260923'
PRIOR = ROOT/'artifacts/pyannote-winograd-profile-build-amd-20260923'
CURRENT = ROOT/'artifacts/pyannote-winograd-product-app-amd-20260923'
ROOT_BUILD = ROOT/'artifacts/pyannote-winograd-product-root-amd-20260923'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('winograd_profile_build_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor)


def previous_closed():
    for folder, digest, root_relative in [
        (PRIOR, '96143e786b96223d70f6b3bf6d9be549b00a064c3dd9444349a976551c1e3cb6', False),
        (CURRENT, 'dc2c7b9f5086ab9b4ee615b9dad7643eaf4c1ed65c1cc71b3e76ee794237d88e', False),
        (ROOT_BUILD, '62141a2a722548697c106e42b2c0d9425b4f0c6ce166611a5bc3ca26a4fccdd0', False)]:
        assert pin(folder/'closed.json')['sha256'] == digest
        proof = read(folder/'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items():
            assert pin((ROOT if root_relative else folder)/name) == wanted, name


def prepare():
    assert not BASE.exists(); previous_closed()
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir(); originals = {}
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)
    for name in ['Program.cs','Diagnostic.cs','NpySupport.cs','SampledAudio.csproj']:
        copy(PRIOR/'bundle/source'/name, bundle/'source'/name)
    copy(ROOT/'global.json', bundle/'source/global.json')
    for source in (CURRENT/'collected/runtimes/candidate').iterdir():
        if source.is_file(): copy(source, bundle/'runtime'/source.name)
    for source in (PRIOR/'collected/runtime').iterdir():
        if source.is_file(): copy(source, bundle/'previous'/source.name)
    product = {name:pin(bundle/'runtime'/name) for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']}
    assert product['Lokad.Onnx.dll']['sha256'] == '521bae1702849ca23dda586515e7cbabaac2d1eabdff04dc90a7ba76059e93fb'
    assert product['Lokad.Onnx.Data.dll']['sha256'] == 'f3b9aa81ee9766797e95216714dec559c5d8b020f8df510cf5f7ee0dda82693a'
    assert not list((bundle/'runtime').glob('SampledAudio.*'))
    path = bundle/'source/Program.cs'; before = path.read_text(encoding='utf8')
    old_guard='family=="pyannote" && !conformance,"Only the original pyannote timing workload"'
    new_guard='family=="parakeet" && !conformance,"Only the current Parakeet timing workload"'
    old_call='actual=parakeet.Transcribe(c.Pcm,16000,ParakeetTranscriptionOptions.Default,CancellationToken.None);'
    new_call='actual=pass<warmup?SampledRequests.WarmupParakeet(parakeet,c.Pcm):SampledRequests.FullParakeet(parakeet,c.Pcm);'
    assert before.count(old_guard)==before.count(old_call)==1
    after=before.replace(old_guard,new_guard).replace(old_call,new_call)
    path.write_text(after,encoding='utf8',newline='\n')
    patch=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='qualified/Program.cs',tofile='parakeet/Program.cs'))
    path=bundle/'source/Diagnostic.cs';before=path.read_text(encoding='utf8')
    needle='static class SampledRequests\n{'
    addition=needle+"""
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static ParakeetTranscription WarmupParakeet(ParakeetTranscriber d, float[] pcm)
        => d.Transcribe(pcm,16000,ParakeetTranscriptionOptions.Default,CancellationToken.None);

    [MethodImpl(MethodImplOptions.NoInlining)]
    public static ParakeetTranscription FullParakeet(ParakeetTranscriber d, float[] pcm)
    { var value=d.Transcribe(pcm,16000,ParakeetTranscriptionOptions.Default,CancellationToken.None); After(value); return value; }
"""
    assert before.count(needle)==1;after=before.replace(needle,addition)
    path.write_text(after,encoding='utf8',newline='\n')
    patch+=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='qualified/Diagnostic.cs',tofile='parakeet/Diagnostic.cs'))
    (bundle/'consumer.patch').write_text(patch,encoding='utf8')
    for suffix in ['dll','deps.json','runtimeconfig.json']:
        copy(PRIOR/'bundle/bridge'/('Bridge.'+suffix),bundle/'bridge'/('Bridge.'+suffix))
    for p in TOOLS.glob('*.py'):
        ast.parse(p.read_text(),str(p))
        if p.name in ['protocol.py','remote.py','remote_prepare.py','checks.py']: copy(p,bundle/'tools'/p.name)
    for folder,label in [(PRIOR,'prior'),(CURRENT,'current'),(ROOT_BUILD,'root')]:
        copy(folder/'closed.json',bundle/'evidence'/(label+'-closed.json'))
    copy(ROOT/'.agent/m38-parakeet-current-profile-20260923.md',bundle/'prospective-plan.md')
    originals.pop('.agent/m38-parakeet-current-profile-20260923.md')
    stage = dict(passed=True, product=product, previous_consumer=pin(bundle/'previous/SampledAudio.dll'),
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json',stage)
    for p in [*TOOLS.iterdir(),MONITOR]:
        if p.is_file(): originals[p.relative_to(ROOT).as_posix()] = pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),product=product)))


if __name__ == '__main__': prepare()
