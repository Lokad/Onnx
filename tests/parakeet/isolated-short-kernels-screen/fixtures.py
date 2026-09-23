"""Preserve twelve captured pairs and derive nine explicit row prefixes."""
import copy
import hashlib
import math
from pathlib import Path
from protocol import pin,read,save

PREFIXES=[(rows,index) for rows,offset in [(48,0),(61,3),(63,3)] for index in range(offset,offset+3)]


def append_prefixes(capture,folder):
    assert len(capture['entries'])==12
    result=copy.deepcopy(capture)
    for rows,index in PREFIXES:
        source=capture['entries'][index];entry=copy.deepcopy(source)
        assert rows<source['m']
        entry.update(name=f"{source['name']}-prefix-m{rows}",m=rows,derived_prefix=dict(source_index=index,rows=rows))
        for key,columns in [('a',source['k']),('y',source['n'])]:
            original=source[key];blob=(folder/original['file']).read_bytes()
            assert pin(folder/original['file'])=={k:original[k] for k in ['bytes','sha256']}
            assert original['shape']==[1,source['m'],columns]
            data=blob[:4*rows*columns];name=f'prefix-{rows}-{index}-{key}.bin'
            target=folder/name;assert not target.exists();target.write_bytes(data)
            entry[key].update(file=name,shape=[1,rows,columns],strides=[rows*columns,columns,1],**pin(target))
        result['entries'].append(entry)
    assert len(result['entries'])==21
    result['fixture_scope']='Twelve captured matrix pairs followed by nine derived row prefixes; prefixes are not recorded clips.'
    save(folder/'result.json',result)
    verify_prefixes(capture,result,folder)
    return result


def verify_prefixes(original,derived,folder):
    assert len(original['entries'])==12 and len(derived['entries'])==21
    assert derived['entries'][:12]==original['entries'] and derived['weights']==original['weights']
    for entry,(rows,index) in zip(derived['entries'][12:],PREFIXES,strict=True):
        parent=original['entries'][index]
        assert entry['derived_prefix']==dict(source_index=index,rows=rows)
        assert entry['name']==f"{parent['name']}-prefix-m{rows}" and entry['m']==rows
        assert all(entry[k]==parent[k] for k in ['node','node_name','k','n','weight'])
        for key,columns in [('a',parent['k']),('y',parent['n'])]:
            descriptor=entry[key];source=parent[key]
            assert descriptor['shape']==[1,rows,columns] and descriptor['strides']==[rows*columns,columns,1]
            assert descriptor['dtype']=='<f4' and not descriptor['reverse']
            assert pin(folder/descriptor['file'])=={k:descriptor[k] for k in ['bytes','sha256']}
            assert pin(folder/source['file'])=={k:source[k] for k in ['bytes','sha256']}
            with (folder/source['file']).open('rb') as f:expected=f.read(4*rows*columns)
            assert (folder/descriptor['file']).read_bytes()==expected
            assert descriptor['bytes']==4*math.prod(descriptor['shape'])
    return True
