"""Preserve the failed archive and apply only the explicit source-policy corrections."""
import difflib,shutil,subprocess,xml.etree.ElementTree as ET
from common import *

BASE=ROOT/'artifacts/whisper-memory-product-v2-20260921'
FAILED=ROOT/'artifacts/whisper-memory-product-20260921'
CORRECTIONS={'WhisperDecoderWeights.cs':'src/Lokad.Onnx.Data/WhisperDecoderWeights.cs',
    'WhisperDecoderWeightsTests.cs':'tests/Lokad.Onnx.Backend.Tests/WhisperDecoderWeightsTests.cs'}


def main():
    closed=read(FAILED/'failure-closed.json');assert closed['closure_passed'] and not closed['campaign_passed']
    assert all(absent(b) for b in closed['births'])
    for name,wanted in closed['files'].items():assert pin(FAILED/name)==wanted,name
    original=read(FAILED/'prepared.json');inventory=read(FAILED/'source-manifest.json')
    BASE.mkdir();source=BASE/'source';source.mkdir();folder=Path(__file__).parent
    for name,wanted in inventory.items():
        assert pin(FAILED/'source'/name)==wanted,name
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(FAILED/'source'/name,target)
    patches=[];corrected={}
    for name,relative in CORRECTIONS.items():
        target=source/relative;before=target.read_text(encoding='utf-8');shutil.copyfile(folder/name,target)
        corrected[relative]=pin(target)
        patches.extend(difflib.unified_diff(before.splitlines(True),target.read_text(encoding='utf-8').splitlines(True),fromfile='first-candidate/'+relative,tofile='corrected-candidate/'+relative))
    (BASE/'source-policy.patch').write_text(''.join(patches),encoding='utf-8')
    for name in ['source.tar','original-source-manifest.json','candidate.patch']:shutil.copyfile(FAILED/name,BASE/name)
    final={p.relative_to(source).as_posix():pin(p) for p in sorted(source.rglob('*')) if p.is_file()}
    assert {n for n in inventory if inventory[n]!=final[n]}==set(CORRECTIONS.values())
    write(BASE/'source-manifest.json',final)
    app=BASE/'consumer';app.mkdir();(BASE/'feed').mkdir()
    for name in ['Consumer.csproj','Program.cs']:shutil.copyfile(folder/name,app/name)
    config=ET.Element('configuration');sources=ET.SubElement(config,'packageSources');ET.SubElement(sources,'clear')
    ET.SubElement(sources,'add',dict(key='local',value=str(BASE/'feed')));ET.SubElement(sources,'add',dict(key='nuget.org',value='https://api.nuget.org/v3/index.json'))
    ET.ElementTree(config).write(app/'nuget.config',encoding='utf-8',xml_declaration=True)
    shutil.copyfile(ROOT/'.agent/m5-whisper-memory-product-20260921.md',BASE/'prospective-plan.md')
    proof=dict(original['product_evidence']);proof[(FAILED/'failure-closed.json').relative_to(ROOT).as_posix()]=pin(FAILED/'failure-closed.json')
    for path in folder.glob('*.py'):compile(path.read_text(encoding='utf-8'),str(path),'exec')
    write(BASE/'prepared.json',dict(source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        qualified_source='private candidate plus nullable-flow and explicit-test-overload corrections',archive=pin(BASE/'source.tar'),
        source_manifest=pin(BASE/'source-manifest.json'),original_manifest=pin(BASE/'original-source-manifest.json'),corrected_sources=corrected,
        original_prepared=pin(FAILED/'prepared.json'),product_evidence=proof,settings=SETTINGS,
        consumer_files={n:pin(app/n) for n in ['Consumer.csproj','Program.cs','nuget.config']},
        tools={p.relative_to(ROOT).as_posix():pin(p) for p in folder.iterdir() if p.is_file()}))
    print(json.dumps(dict(prepared=True,files=len(final),corrections=corrected,receipt=pin(BASE/'prepared.json'))))


if __name__=='__main__':main()
