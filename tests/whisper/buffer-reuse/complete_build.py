"""Supply the omitted CLI test prerequisite without overwriting the first test evidence."""
from pathlib import Path
import json,shutil,subprocess,tarfile,xml.etree.ElementTree as ET
from prepare import ROOT,BASE,pin,run


def main():
    source=BASE/'source';receipt=json.loads((BASE/'source.json').read_text())
    assert not (BASE/'built.json').exists()
    for name,wanted in receipt['files'].items():assert pin(source/name)==wanted,name
    original=ET.parse(BASE/'test-results/backend.trx').getroot()
    ns={'t':'http://microsoft.com/schemas/VisualStudio/TeamTest/2010'}
    failed=original.findall('.//t:UnitTestResult[@outcome="Failed"]',ns)
    assert len(failed)==4 and all('CliInputTests.Cli_' in x.attrib['testName'] for x in failed)
    archive=BASE/'cli-source.tar'
    if not (BASE/'cli-source.json').exists():
        assert not (source/'src/Lokad.Onnx.CLI').exists() and not archive.exists()
        subprocess.run(['git','archive','--format=tar','--output',str(archive),receipt['revision'],'src/Lokad.Onnx.CLI'],cwd=ROOT,check=True)
        with tarfile.open(archive) as tar:tar.extractall(source,filter='data')
        supplement={p.relative_to(source).as_posix():pin(p) for p in sorted((source/'src/Lokad.Onnx.CLI').rglob('*')) if p.is_file()}
        with (BASE/'cli-source.json').open('x') as f:json.dump(dict(revision=receipt['revision'],archive=pin(archive),files=supplement),f,indent=2)
        run(['dotnet','build',str(source/'src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj'),'--tl:off','--nologo','-v','minimal','-c','Release'],BASE/'cli-build.log')
        project=source/'tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'
        run(['dotnet','test',str(project),'--tl:off','--nologo','-v','minimal','-c','Release','--no-build','--logger','trx;LogFileName=backend.trx','--results-directory',str(BASE/'complete-test-results')],BASE/'complete-tests.log')
    cli=json.loads((BASE/'cli-source.json').read_text());assert pin(archive)==cli['archive']
    for name,wanted in cli['files'].items():assert pin(source/name)==wanted,name
    final=ET.parse(BASE/'complete-test-results/backend.trx').getroot()
    counters=final.find('.//t:Counters',ns).attrib
    assert counters['failed']=='0' and counters['total']=='3182' and counters['passed']=='3089'
    for name,wanted in receipt['files'].items():assert pin(source/name)==wanted,name
    product=BASE/'product-bin';product.mkdir(exist_ok=True);built=source/'src/Lokad.Onnx.CLI/bin/Release/net10.0'
    for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll','Google.Protobuf.dll','FastBertTokenizer.dll','Lokad.Tokenizers.dll','SixLabors.ImageSharp.dll']:
        if (product/name).exists():assert pin(built/name)==pin(product/name)
        else:shutil.copyfile(built/name,product/name)
    paths=[BASE/n for n in ['source.json','cli-source.json','original-source.tar','cli-source.tar','build.log','tests.log','cli-build.log','complete-tests.log','test-results/backend.trx','complete-test-results/backend.trx']]+list(product.iterdir())
    value=dict(built=True,tests_passed=True,source=pin(BASE/'source.json'),cli_supplement=pin(BASE/'cli-source.json'),initial_failure='Four CLI subprocess tests lacked the CLI build prerequisite; initial evidence retained.',counters=counters,files={p.relative_to(BASE).as_posix():pin(p) for p in paths})
    with (BASE/'built.json').open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps(dict(built=True,tests_passed=True,counters=counters,receipt=pin(BASE/'built.json'))))


if __name__=='__main__':main()
