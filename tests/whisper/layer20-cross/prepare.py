"""Generate controlled layer-20 workers, then bind their complete saved-input square."""
from pathlib import Path
import argparse,ast,copy,difflib,json,shutil,subprocess,sys
import onnx
from common import ROOT,CORE,ORIGIN,PROTOCOL,KINDS,pin,read,write,verify,schedule

HERE=Path(__file__).resolve().parent
OLD=ROOT/'artifacts/whisper-trace-selected-20260920'

def replace(text,old,new,count=1):
    assert text.count(old)==count,(old,text.count(old),count)
    return text.replace(old,new)

def transform(name,text):
    if name=='Program.cs':
        for old,new,count in [('whisper-selected-natural-trace-v1',PROTOCOL,1),('selected is >= 0 and < 4','selected is >= 0 and < 8',1),
            ('requests.Length == 4','requests.Length == 8',1),('Four fixed requests','Eight fixed case/feature requests',1),('outputSpecs.Length == 41','outputSpecs.Length == 12',1),
            ('"input_features"','"/layers.19/Add_1_output_0"',2),('"managed_features"','"managed_input"',1),('"native_features"','"native_input"',1),
            ('new[] { 1, 128, 3000 }','new[] { 1, 1500, 1280 }',1),('context.Outputs.Count == 41','context.Outputs.Count == 12',1),
            ('kind + ":40"','kind + ":11"',1),('all 41 outputs retained','all 12 outputs retained',1),
            ('Selected natural-case trace; original-output instrumentation differences retained; no product qualification','Natural layer20 input cross; extraction effects measured separately; no full-model qualification',1)]:text=replace(text,old,new,count)
    elif name=='Probe.csproj':
        text=replace(text,'WhisperSelectedTrace','WhisperLayer20Cross')
        text=replace(text,'    <Compile Include="../../Shared/NpySupport.cs" Link="NpySupport.cs" />\n','')
    elif name=='native.py':
        for old,new,count in [('args.request_index in range(4)','args.request_index in range(8)',1),("['input_features']","['/layers.19/Add_1_output_0']",1),
            ("'native_features' if kind=='NN' else 'managed_features'","'native_input' if kind=='NN' else 'managed_input'",1),
            ("{'input_features':features}","{'/layers.19/Add_1_output_0':features}",1),('len(values)==41','len(values)==12',1),('all 41 outputs','all 12 outputs',1)]:text=replace(text,old,new,count)
    elif name=='run.py':
        text=replace(text,'len(jobs)==8','len(jobs)==16');text=replace(text,'WhisperSelectedTrace.dll','WhisperLayer20Cross.dll')
        text=replace(text,'eight finite sequential workers','sixteen finite sequential workers');text=replace(text,'all eight trace workers','all sixteen layer20 workers')
    return text

def generate(base):
    assert not base.exists() and pin(OLD/'closed.json')['sha256']==ORIGIN
    closed=read(OLD/'closed.json');assert closed['passed'] is True
    (base/'host').mkdir(parents=True);(base/'tools').mkdir();sources={};generated={};diff=[]
    for name in ['Program.cs','Probe.csproj','native.py','run.py']:
        key='closed-source/'+name;source=OLD/key;assert pin(source)==closed['files'][key];sources[key]=pin(source)
        before=source.read_text();after=transform(name,before);target=base/('host' if name.endswith(('.cs','.csproj')) else 'tools')/name
        with target.open('x',encoding='utf-8',newline='\n') as stream:stream.write(after)
        generated[target.relative_to(base).as_posix()]=pin(target)
        diff.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='selected-trace/'+name,tofile='layer20-cross/'+name))
    previous=read(OLD/'manifest.json');support=ROOT/'tests/Shared/NpySupport.cs';assert pin(support)==previous['files']['tests/Shared/NpySupport.cs']
    shutil.copyfile(support,base/'host/NpySupport.cs');generated['host/NpySupport.cs']=pin(base/'host/NpySupport.cs')
    for name in ['common.py','audit.py','test_audit.py','close.py']:
        shutil.copyfile(HERE/name,base/'tools'/name);generated['tools/'+name]=pin(base/'tools'/name)
    for path in (base/'tools').glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    with (base/'generated.diff').open('x',encoding='utf-8') as stream:stream.write(''.join(diff))
    write(base/'generation.json',dict(prior=pin(OLD/'closed.json'),sources=sources,generated=generated,diff=pin(base/'generated.diff'),generator={p.name:pin(p) for p in HERE.iterdir() if p.is_file()}))
    print('Generated',len(generated),'files; original host/core/options retained with declared input/output changes.')

def freeze(base):
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Commit tools first'
    assert not (base/'manifest.json').exists() and shutil.disk_usage(base).free>=25*1024**3
    assert pin(OLD/'closed.json')['sha256']==ORIGIN;closed=read(OLD/'closed.json');previous=read(OLD/'manifest.json')
    assert pin(OLD/'manifest.json')==closed['files']['manifest.json'];generation=read(base/'generation.json')
    for name,want in generation['generator'].items():assert pin(HERE/name)==want,name
    for name,want in generation['generated'].items():assert pin(base/name)==want,name
    assert pin(base/'generated.diff')==generation['diff']
    files={}
    def bind(path,expected=None):
        value=pin(path)
        if expected is not None:assert value==expected,path
        files[path.relative_to(ROOT).as_posix()]=value
    for p in [OLD/'closed.json',OLD/'manifest.json']:bind(p)
    review=ROOT/'artifacts/whisper-layer20-structural-review-20260920/review.json'
    assert pin(review)['sha256']=='580c41695d6f3bbc881d31ab2bf6447c6081ee5d90383de3727bc65b572ac858';structural=read(review);assert structural['passed']
    bind(review);bind(review.with_name('review.py'),structural['script'])
    for name,want in structural['files'].items():bind(ROOT/name,want)
    model_name='models/whisper-large-v3-turbo/onnx/encoder_layer20_20260918.onnx';model=onnx.load(ROOT/model_name,load_external_data=False)
    outputs=[]
    for v in model.graph.output:
        assert v.type.tensor_type.elem_type==onnx.TensorProto.FLOAT
        dims=[d.dim_value if d.HasField('dim_value') else 1 for d in v.type.tensor_type.shape.dim]
        assert all(d.HasField('dim_value') or d.dim_param=='batch_size' for d in v.type.tensor_type.shape.dim)
        outputs.append(dict(name=v.name,shape=dims))
    assert len(outputs)==12 and outputs[-1]['name']=='/layers.20/Add_1_output_0'
    requests=[]
    for selected,item in enumerate(previous['requests']):
        for features in ['managed','native']:
            request=dict(request=len(requests),selected_request=selected,original_request=item['original_request'],features=features,name=item['name'],baselines={})
            for engine in KINDS:
                kind=('M' if engine=='managed' else 'N')+('M' if features=='managed' else 'N')
                folder=OLD/'outputs'/f"{engine}-{selected:02}-{item['name']}";result=read(folder/'result.json')
                bind(folder/'result.json',closed['files'][(folder/'result.json').relative_to(OLD).as_posix()])
                row=next(r for r in result['records'] if r['kind']==kind)
                def saved(index):
                    value=row['outputs'][index];assert value['shape']==[1,1500,1280] and value['name']==previous['outputs'][index]['name']
                    path=folder/value['file'];bind(path,closed['files'][path.relative_to(OLD).as_posix()]);assert pin(path)['sha256']==value['sha256']
                    return dict(file=path.relative_to(ROOT).as_posix(),format='f32',shape=value['shape'],raw_sha256=value['sha256'])
                request[engine+'_input']=saved(27)
                baseline=saved(28)
                for cell in KINDS[engine]:request['baselines'][cell]=baseline
            requests.append(request)
    assert pin(base/'bin/Lokad.Onnx.dll')['sha256']==CORE
    assert '0 Warning(s)' in (base/'build.log').read_text() and '0 Error(s)' in (base/'build.log').read_text() and 'OK' in (base/'unit-tests.log').read_text()
    for name in ['build.log','unit-tests.log','generation.json','generated.diff']:bind(base/name)
    for directory in [base/'bin',base/'tools',base/'host',HERE]:
        for p in directory.iterdir():
            if p.is_file():bind(p)
    for name in ['audit.py','common.py']:
        path=ROOT/'tests/whisper/input-cross'/name;bind(path,previous['files'][path.relative_to(ROOT).as_posix()])
    for name,want in previous['native_runtime']['files'].items():assert pin(Path(name))==want
    spec=dict(schema=1,protocol=PROTOCOL,source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        model=model_name,original_model=previous['original_model'],requests=requests,outputs=outputs,core_sha256=CORE,native_runtime=previous['native_runtime'],
        native_serialization=previous['native_serialization'],files=files,limits=dict(seconds=180,rss=8*1024**3,available=1024**3,preflight_available=10*1024**3),
        preflight_wait_seconds=900,workers=16,calls=32,arrays=384,output_payload_bytes=4423680000,scaled_error_limit=1e-4)
    spec['schedule']=schedule(spec);write(base/'manifest.json',spec);verify(spec)
    print('Frozen',len(files),'files;',pin(base/'manifest.json'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['generate','freeze']);parser.add_argument('--artifact',type=Path,required=True)
    args=parser.parse_args();globals()[args.action](args.artifact.resolve())
