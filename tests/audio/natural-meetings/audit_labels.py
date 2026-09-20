"""Independently reconstruct the prepared transcript from original XML DOM nodes."""
from pathlib import Path
from decimal import Decimal
from xml.dom import minidom
import argparse
import hashlib
import importlib.util
import json
import zipfile


def digest(path):
    with Path(path).open('rb') as stream:
        return dict(bytes=Path(path).stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();base=args.artifact.resolve()
    assert not args.output.exists()
    value=json.loads((base/'labels.json').read_text(encoding='utf-8'));root=Path(__file__).resolve().parents[3]
    normalizer=root/value['normalizer']['path'];assert digest(normalizer)=={k:value['normalizer'][k] for k in ['bytes','sha256']}
    spec=importlib.util.spec_from_file_location('retained_asr_normalization',normalizer)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert digest(base/'ami_public_manual_1.6.2.zip')==value['archive']
    all_rows={};observations=[]
    with zipfile.ZipFile(base/'ami_public_manual_1.6.2.zip') as archive:
        for name,wanted in value['annotation_files'].items():
            path=base/'annotation-files'/name;assert digest(path)==wanted
            assert archive.read(name)==path.read_bytes()
        for meeting in ['ES2004a','IS1009a']:
            rows=[]
            for speaker in 'ABCD':
                document=minidom.parseString(archive.read(f'words/{meeting}.{speaker}.words.xml'))
                elements=[n for n in document.documentElement.childNodes if n.nodeType==n.ELEMENT_NODE]
                for index,node in enumerate(elements):
                    if node.tagName!='w' or node.getAttribute('punc')=='true':continue
                    attrs={('{http://nite.sourceforge.net/}id' if name=='nite:id' else name):node.getAttribute(name) for name in node.attributes.keys()}
                    text=''.join(n.data for n in node.childNodes if n.nodeType in (n.TEXT_NODE,n.CDATA_SECTION_NODE)).strip()
                    rows.append(dict(id=node.getAttribute('nite:id'),speaker=speaker,index=index,
                        start=str(Decimal(attrs['starttime'])),end=str(Decimal(attrs['endtime'])),text=text,attributes=attrs))
            all_rows[meeting]=rows
    for case in value['cases']+[value['recovery']]:
        meeting=case['name'].split('-')[0];duration=Decimal(case['seconds']);rows=all_rows[meeting]
        included=sorted([r for r in rows if Decimal(r['end'])<=duration],key=lambda r:(Decimal(r['start']),Decimal(r['end']),r['speaker'],r['index']))
        crossing=sorted([r for r in rows if Decimal(r['start'])<duration<Decimal(r['end'])],key=lambda r:(Decimal(r['start']),Decimal(r['end']),r['speaker'],r['index']))
        assert included==case['words'] and crossing==case['excluded_boundary_words']
        text=' '.join(r['text'] for r in included);assert text==case['text']
        assert module.normalize(text)==case['normalized_reference'] and digest(case['audio'])==case['wav']
        observations.append(dict(name=case['name'],lexical_records=len(included),normalized_words=len(case['normalized_reference'].split()),
            crossing_records=len(crossing),truncated_records=sum(r['attributes'].get('trunc')=='true' for r in included)))
    result=dict(passed=True,labels=digest(base/'labels.json'),auditor=digest(Path(__file__)),cases=observations,
        scope='Independent XML DOM reconstruction of all lexical records and exact original-byte checks; no recognition or ASR score')
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(observations,indent=2))


if __name__=='__main__':main()
