"""Prepare portable test guards while preserving every qualified assertion."""
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
SOURCE=ROOT/'artifacts/parakeet-packed-final-row-source-20260925'
BUILD=ROOT/'artifacts/parakeet-packed-final-row-build-amd-20260925'
OUT=ROOT/'artifacts/parakeet-packed-final-row-public-tests-20260925'
NAME='tests/Lokad.Onnx.Backend.Tests/OwnedPackedWeightTests.cs'
TARGET=TOOLS/'OwnedPackedWeightTests.cs.txt'
FMA='Skip.If(!Fma.IsSupported, "Owned packed execution requires FMA.");'
GUARDS={
    'PreparationIsOptIn_Idempotent_AndSurvivesInvalidation':FMA,
    'ExistingPackedRecordAndBudgetRemainUntouched':FMA,
    'SharedOrVisibleWeightIsNotConsumed':FMA,
    'ActualShapesUseExistingArithmetic_WithoutRemainderReconstruction':FMA,
    'LogicalFallbackRemainsBitExact':'Skip.If(mode == "intrinsics" && !Fma.IsSupported, "Explicit intrinsic mode requires FMA.");',
    'PackedStorageDestinationAliasIsRejectedBeforeAnyWrite':FMA,
    'ContextsProtectPackedRoots_AndHeldOutputsSurvivePoolReuse':FMA,
    'PreparedGraphDoesNotRetainTheReplacedOriginalArray':FMA,
    'HardwareDisabledPreparationLeavesOriginals_AndLogicalAutoFallbackWorks':
        'Skip.If(Fma.IsSupported || Avx2.IsSupported || Avx512F.IsSupported, "This contract requires disabled x86 intrinsics.");'
}


def read(path):return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def header(text,name):
    pattern=r'(?P<attributes>(?:    \[[^\n]+\]\n)+)    public void '+re.escape(name)+r'\([^\n]*\)\n    \{\n'
    matches=list(re.finditer(pattern,text));assert len(matches)==1,name
    return matches[0]


def guarded(original):
    result=original
    for name,guard in GUARDS.items():
        match=header(result,name);old=match.group()
        assert old.count('[Fact]')+old.count('[Theory]')==1
        new=old.replace('[Fact]','[SkippableFact]').replace('[Theory]','[SkippableTheory]')+'        '+guard+'\n'
        result=result[:match.start()]+new+result[match.end():]
    verify(original,result)
    return result


def verify(original,actual):
    restored=actual
    for name,guard in GUARDS.items():
        match=header(restored,name);old=match.group();line='        '+guard+'\n'
        assert restored[match.end():].startswith(line),(name,'guard not first')
        assert old.count('[SkippableFact]')+old.count('[SkippableTheory]')==1
        new=old.replace('[SkippableFact]','[Fact]').replace('[SkippableTheory]','[Theory]')
        restored=restored[:match.start()]+new+restored[match.end()+len(line):]
    assert restored==original,'Qualified test assertions/helpers or data changed'
    return dict(passed=True,guards=len(GUARDS),original_assertions_helpers_and_data_exact=True)


def qualified_census():
    closure=read(BUILD/'closed.json');assert closure['passed']
    cases={}
    for mode in ['512','256','scalar']:
        name='capture-collected/logs/owned-'+mode+'.trx';path=BUILD/name
        assert pin(path)==closure['files'][name]
        rows=ET.parse(path).findall('.//{*}UnitTestResult')
        assert all(r.attrib['outcome']=='Passed' for r in rows)
        selected=[r.attrib['testName'] for r in rows if r.attrib['testName'].startswith((
            'Lokad.Onnx.Backend.Tests.OwnedPackedWeightTests.',
            'Lokad.Onnx.Backend.Tests.OwnedPackedUnavailableTests.'))]
        assert len(selected)==len(set(selected));cases[mode]=sorted(selected)
    assert cases['512']==cases['256'] and len(cases['512'])==40 and len(cases['scalar'])==1
    assert 'OwnedPackedUnavailableTests.' in cases['scalar'][0]
    return dict(passed=cases['512'],skipped=cases['scalar'])


def main():
    assert not OUT.exists() and not TARGET.exists()
    prepared=read(SOURCE/'prepared.json');assert prepared['passed']
    original=SOURCE/'source'/NAME;assert pin(original)==prepared['source'][NAME]
    text=original.read_text(encoding='utf8');actual=guarded(text);census=qualified_census()
    OUT.mkdir();TARGET.write_text(actual,encoding='utf8')
    receipt=dict(passed=True,source=pin(original),prepared=pin(SOURCE/'prepared.json'),qualified_contracts=pin(BUILD/'closed.json'),
        output=pin(TARGET),review=verify(text,actual),new_backend_cases=census,
        normal_expected=dict(passed=3539,skipped=42),avx512_disabled_expected=dict(passed=3449,skipped=132),
        root_product_changed=False,compiled=False,release_admitted=False,generator=pin(Path(__file__)))
    (OUT/'prepared.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='new_backend_cases'}))


if __name__=='__main__':main()
