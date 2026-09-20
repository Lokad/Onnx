"""Prepare timed human ASR references before recognizing the fixed AMI excerpts."""
from pathlib import Path
from decimal import Decimal, InvalidOperation
from collections import Counter, defaultdict
import argparse
import hashlib
import importlib.util
import json
import xml.etree.ElementTree as ET
import zipfile

MEETINGS = ['ES2004a','IS1009a']
ARCHIVE_SHA = 'b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d'
NITE = '{http://nite.sourceforge.net/}'


def pin(path):
    with Path(path).open('rb') as stream:
        return dict(bytes=Path(path).stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def timestamp(value):
    try:number=Decimal(value)
    except (InvalidOperation,TypeError):raise ValueError('Invalid annotation timestamp') from None
    if not number.is_finite() or number<0:raise ValueError('Invalid annotation timestamp')
    return number


def parse_words(data,meeting,speaker):
    root=ET.fromstring(data);prefix=meeting+'.'+speaker+'.words'
    if root.get(NITE+'id')!=prefix:raise ValueError('Wrong annotation identity')
    seen=set();words=[];counts=Counter()
    for index,node in enumerate(root):
        identity=node.get(NITE+'id')
        if not identity or not identity.startswith(prefix) or identity in seen:
            raise ValueError('Missing, duplicate or wrong word identity')
        seen.add(identity);counts[node.tag]+=1
        if node.tag!='w' or node.get('punc')=='true':continue
        start,end=timestamp(node.get('starttime')),timestamp(node.get('endtime'))
        if start>=end:raise ValueError('Nonpositive lexical interval')
        text=''.join(node.itertext()).strip()
        if not text:raise ValueError('Empty lexical word')
        words.append(dict(id=identity,speaker=speaker,index=index,start=str(start),end=str(end),
                          text=text,attributes=dict(node.attrib)))
    return words,dict(counts)


def reference(words,duration):
    duration=timestamp(str(duration))
    if duration<=0:raise ValueError('Nonpositive crop duration')
    included=[];crossing=[]
    for row in words:
        start,end=timestamp(row['start']),timestamp(row['end'])
        if start<duration:
            (included if end<=duration else crossing).append(row)
    key=lambda r:(timestamp(r['start']),timestamp(r['end']),r['speaker'],r['index'])
    included.sort(key=key);crossing.sort(key=key)
    return dict(words=included,excluded_boundary_words=crossing,text=' '.join(r['text'] for r in included))


def union(words,mapping,duration):
    groups=defaultdict(list)
    for row in words:
        start,end=timestamp(row['start']),min(timestamp(row['end']),Decimal(duration))
        if start<end:groups[mapping[row['speaker']]].append((start,end))
    result=[]
    for speaker,rows in groups.items():
        merged=[]
        for start,end in sorted(rows):
            if merged and start<=merged[-1][1]:merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
            else:merged.append((start,end))
        result.extend([float(start),float(end),speaker] for start,end in merged)
    return sorted(result)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--diarization',type=Path,required=True)
    args=parser.parse_args();base=args.artifact.resolve();diarization=args.diarization.resolve();repo=Path(__file__).resolve().parents[3]
    assert not (base/'labels.json').exists() and not (base/'annotation-files').exists()
    archive=base/'ami_public_manual_1.6.2.zip';assert pin(archive)['sha256']==ARCHIVE_SHA
    dataset=json.loads((diarization/'inputs/dataset.json').read_text())
    assert [r['name'] for r in dataset['cases']]==MEETINGS
    normalizer=repo/'tests/audio/multilingual/common.py'
    spec=importlib.util.spec_from_file_location('fixed_asr_normalization',normalizer)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    names=['00README_MANUAL.txt','corpusResources/meetings.xml']+[f'words/{m}.{s}.words.xml' for m in MEETINGS for s in 'ABCD']
    with zipfile.ZipFile(archive) as source:
        assert len(source.namelist())==len(set(source.namelist()))
        raw={name:source.read(name) for name in names}
    metadata=ET.fromstring(raw['corpusResources/meetings.xml']);cases=[];all_words={};details=[]
    for case in dataset['cases']:
        meeting=case['name'];members=[r for r in metadata if r.get('observation')==meeting];assert len(members)==1
        mapping={r.get('nxt_agent'):r.get('global_name') for r in members[0]};assert set(mapping)==set('ABCD') and len(set(mapping.values()))==4
        words=[];counts={}
        for speaker in 'ABCD':
            rows,kinds=parse_words(raw[f'words/{meeting}.{speaker}.words.xml'],meeting,speaker)
            words.extend(rows);counts[speaker]=kinds
        assert len({r['id'] for r in words})==len(words)
        assert union(words,mapping,600)==case['intervals'], 'Original lexical intervals differ from pinned words-only RTTM'
        wav=diarization/'inputs'/case['path'];assert pin(wav)==case['wav']
        selected=reference(words,600);selected['normalized_reference']=module.normalize(selected['text'])
        selected.update(name=meeting,seconds=600,samples=9600000,language='en',audio=str(wav),
            wav=case['wav'],pcm_sha256=case['pcm_sha256'],speakers=mapping,coverage=case['coverage'])
        cases.append(selected);all_words[meeting]=words;details.append(dict(meeting=meeting,tags=counts,raw_lexical_words=len(words),rttm_agreement=True))
    recovery=reference(all_words['ES2004a'],30);recovery['normalized_reference']=module.normalize(recovery['text'])
    manifest=json.loads((diarization/'manifest.json').read_text());recovery.update(name='ES2004a-recovery30',seconds=30,samples=480000,
        language='en',audio=cases[0]['audio'],wav=cases[0]['wav'],pcm_sha256=manifest['cases'][2]['pcm_sha256'])
    destination=base/'annotation-files';destination.mkdir()
    for name,data in raw.items():
        path=destination/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(data)
    value=dict(schema=1,archive=pin(archive),diarization_dataset=pin(diarization/'inputs/dataset.json'),
        normalizer=dict(path=normalizer.relative_to(repo).as_posix(),**pin(normalizer)),
        policy='Complete lexical words inside fixed crop; keep fillers/truncations; sort by (start,end,speaker,source index); existing multilingual normalization; recovery excluded from aggregate',
        scope='Chronological mixed-speaker WER observation, not an official AMI ASR benchmark; overlap has ambiguous word order',
        cases=cases,recovery=recovery,annotation_files={n:pin(destination/n) for n in names},preparation=details,
        preparer=pin(Path(__file__)))
    write(base/'labels.json',value)
    print(json.dumps([dict(name=r['name'],lexical_records=len(r['words']),normalized_words=len(r['normalized_reference'].split()),
        excluded_boundary_words=r['excluded_boundary_words']) for r in cases],indent=2))


if __name__=='__main__':main()
