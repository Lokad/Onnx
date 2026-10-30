"""Bind existing local fixtures/assets to a prospectively fixed short benchmark."""
from pathlib import Path
import argparse,copy,json
import audio_worker as audio
import benchmark_worker as common
import campaign_evidence as evidence
import gemma_worker as gemma
import graph_worker as graph


def prepare(args):
    output=args.output.resolve()
    evidence.require(not output.exists(),'choose fresh full fixture outputs')
    build=common.read_build(args.prepared.resolve(),'L1')
    original=json.loads(args.graphs.read_text(encoding='utf-8-sig'))
    graphs=copy.deepcopy(original);graphs['core']=build['core_sha256']
    inventory={}
    def include(path):
        path=Path(path).resolve()
        inventory[str(path)]=dict(path=str(path),bytes=path.stat().st_size,sha256=evidence.sha256(path))
    for case in graphs['cases']:
        case['model']=str((args.graphs.resolve().parent/case['model']).resolve());include(case['model'])
        sidecar=Path(case['model']).with_name('model.onnx_data')
        if sidecar.exists():include(sidecar)
        for item in (*case['inputs'],*case['outputs']):
            if 'file' in item:
                item['file']=str((args.graphs.resolve().parent/item['file']).resolve());include(item['file'])
    evidence.require({c['key'] for c in graphs['cases']}==set(graph.CASES) and len(graphs['cases'])==4,'all four graph cases are required')
    if args.inventory:
        old=json.loads(args.inventory.read_text(encoding='utf-8-sig'))
        for item in old['assets']:
            path=(args.inventory.resolve().parent/item.get('path',item.get('remote',''))).resolve()
            evidence.require(path.is_file() and path.stat().st_size==item['bytes'] and evidence.sha256(path)==item['sha256'],'source asset inventory changed')
            include(path)
    gemma.fixture(args.gemma.resolve().parent)
    for path in args.gemma.resolve().parent.rglob('*'):
        if path.is_file():include(path)
    output.mkdir(parents=True)
    common.save_json(output/'graphs.json',graphs)
    config=dict(schema=1,audio_root=str(args.audio_root.resolve()),gemma_root=str(args.gemma_root.resolve()),
        native_versions={},audio={},graphs=dict(path='graphs.json',sha256=evidence.sha256(output/'graphs.json')),
        gemma=dict(path=str(args.gemma.resolve()),sha256=evidence.sha256(args.gemma.resolve())))
    for family in ('pyannote','parakeet'):
        source=getattr(args,family).resolve()
        fixture,_=audio.manifest(source,family)
        fixture=copy.deepcopy(fixture)
        fixture.pop('warmup_passes',None)
        fixture.update(schema=2,protocol='audio-sustained-v2',conditioning_passes=2,measured_passes=3,process_timeout_seconds=600)
        common.save_json(output/(family+'.json'),fixture)
        config['audio'][family]=dict(path=family+'.json',sha256=evidence.sha256(output/(family+'.json')))
        versions={k:fixture['native_versions'][k] for k in ('numpy','onnxruntime')}
        if config['native_versions']:evidence.require(config['native_versions']==versions,'native version fixtures differ')
        config['native_versions']=versions
    config['assets']=sorted(inventory.values(),key=lambda item:item['path'])
    common.save_json(output/'full.json',config)
    print('PASS prospectively fixed full fixtures: '+str(output/'full.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','graphs','gemma','parakeet','pyannote','audio-root','gemma-root','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--inventory',type=Path,help='existing protected asset inventory, validated without mutation')
    args=p.parse_args()
    try:prepare(args)
    except (ValueError,OSError,KeyError) as error:p.exit(1,str(error)+'\n')


if __name__=='__main__':main()
