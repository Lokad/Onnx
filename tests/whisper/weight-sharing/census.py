"""Reproduce the exact serialized-weight census in an explicitly new output directory."""
from pathlib import Path
import argparse,gc,hashlib,json,time
import onnx

ROOT=Path(__file__).resolve().parents[3]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    base=args.output.resolve();assert not base.exists();base.mkdir(parents=True)
    manifest=json.loads((ROOT/'artifacts/whisper-buffer-reuse-20260920/prototype-manifest.json').read_text())
    start=time.monotonic();groups={};reports=[];pins={}
    for model_index,name in enumerate(['decoder_model.onnx','decoder_with_past_model.onnx']):
        path=ROOT/'models/whisper-large-v3-turbo/onnx'/name
        with path.open('rb') as stream:actual=dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())
        expected=manifest['models']['onnx/'+name];assert actual=={k:expected[k] for k in ['bytes','sha256']};pins[name]=actual
        model=onnx.load(path,load_external_data=False);rows=[];matched=0;raw_total=0
        for tensor in model.graph.initializer:
            assert tensor.raw_data,'Census requires raw payloads for every initializer'
            content=tensor.raw_data;sha=hashlib.sha256(content).hexdigest();key=(tensor.data_type,tuple(tensor.dims),sha,len(content))
            match=groups.get(key);row=dict(name=tensor.name,type=tensor.data_type,shape=list(tensor.dims),bytes=len(content),sha256=sha)
            if model_index==0:
                if match is None:groups[key]=(tensor.name,content)
                else:assert match[1]==content
            elif match is not None:
                assert match[1]==content;row['exact_match_in_first']=match[0];matched+=len(content)
            rows.append(row);raw_total+=len(content)
        reports.append(dict(model=name,initializers=len(model.graph.initializer),raw_initializers=len(rows),raw_payload_bytes=raw_total,matching_payload_bytes=matched,rows=rows))
        del model;gc.collect()  # Local Python parsing only; no .NET worker or inference.
    value=dict(passed=True,inference=False,models=pins,graphs=reports,seconds=time.monotonic()-start,scope='Exact serialized initializer content/shape/type census; no product sharing or process-memory saving claimed')
    with (base/'census.json').open('x') as stream:json.dump(value,stream,indent=2)
    print(json.dumps(dict(seconds=value['seconds'],graphs=[{k:v for k,v in r.items() if k!='rows'} for r in reports])))


if __name__=='__main__':main()
