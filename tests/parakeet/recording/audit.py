"""Independent recording/result/process audit; no inference and no tolerance fitting."""
from pathlib import Path
import argparse,hashlib,importlib.util,importlib.metadata,json,re,struct
import numpy as np
import psutil

NAMES=['connected','shifted','hard-boundary','token-limit','window-limit','maximum-speech','tiny-tail','maximum-silence','empty']

def require(condition,message):
    if not condition:raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def canonical_refusals(names):
    # Labels used float.ToString(), whose infinity symbol follows the runtime culture.
    aliases={'nonfinite-∞':'nonfinite-Infinity','nonfinite--∞':'nonfinite--Infinity'}
    return [aliases.get(name,name) for name in names]

def boundary(pcm,start):
    count=min(480000,len(pcm)-start)
    if count==len(pcm)-start:return count,'EndOfRecording'
    flags=[sum(float(v)*float(v) for v in pcm[start+i:start+i+160])<=160*.000009 for i in range(400000,480000,160)]
    runs=[];i=0
    while i<len(flags):
        if not flags[i]:i+=1;continue
        begin=i
        while i<len(flags) and flags[i]:i+=1
        if i-begin>=20:runs.append((i-begin,begin))
    if not runs:return count,'HardLimit'
    size,begin=sorted(runs)[-1];return 400000+160*begin+80*size,'Quiet'

def validate(result,case,pcm,vocab):
    require(result['duration_seconds']==len(pcm)/16000,'Duration differs')
    windows=result['windows'];require(len(windows)<=case['max_windows'],'Too many windows')
    position=0;text=[];limited=False
    for index,window in enumerate(windows):
        require(position<len(pcm),'Window after end')
        length,kind=boundary(pcm,position)
        require((window['start_seconds'],window['audio_seconds'],window['boundary'])==(position/16000,length/16000,kind),'Window boundary differs')
        decoded=window['decoding'];tokens=decoded['token_ids'];frames=decoded['frame_indices'];durations=decoded['duration_frames']
        require(len(tokens)==len(frames)==len(durations) and len(tokens)<=case['max_tokens'],'Token cardinality differs')
        require(all(type(v) is int and 0<=v<8192 for v in tokens),'Token ID differs')
        expected_text=re.sub(r'\A\s|\s\B|(\s)\b',lambda match:' ' if match.group(1) else '', ''.join(vocab[v] for v in tokens))
        require(decoded['text']==expected_text,'Decoded text differs')
        audio=pcm[position:position+length]
        if not np.any(audio):
            require(decoded==dict(text='',token_ids=[],frame_indices=[],duration_frames=[],stop_reason='SilentInput',encoded_frames=0,decoder_calls=0),'Silent decision differs')
        else:
            encoded=(max(257,length)//160+8)//8
            require(decoded['encoded_frames']==encoded and decoded['stop_reason'] in ('EndOfAudio','TokenLimit'),'Encoded frame/stop differs')
            require(all(type(v) is int and 0<=v<encoded for v in frames) and frames==sorted(frames),'Token frame differs')
            require(all(type(v) is int and 0<=v<=4 for v in durations),'Predicted duration differs')
            require(type(decoded['decoder_calls']) is int and len(tokens)<=decoded['decoder_calls']<=encoded*(case['max_tokens_per_frame']+1),'Decoder work differs')
            if decoded['stop_reason']=='TokenLimit':require(len(tokens)==case['max_tokens'],'False token limit')
        limited=decoded['stop_reason']=='TokenLimit'
        if limited:
            require(index==len(windows)-1,'Continued after token limit');break
        position+=length
        if decoded['text']:text.append(decoded['text'])
    expected_stop='TokenLimit' if limited else 'Completed' if position==len(pcm) else 'WindowLimit'
    if expected_stop=='WindowLimit':require(len(windows)==case['max_windows'],'Premature window limit')
    require((result['stop_reason'],result['processed_seconds'],result['text'])==(expected_stop,position/16000,' '.join(text)),'Committed recording differs')

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'Existing audit output');base=a.artifact.resolve();root=Path(__file__).resolve().parents[3]
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    frozen=read(base/'frozen.json');inputs=read(base/'inputs/inputs.json');native=read(base/'native/result.json');managed=read(base/'managed/result.json');processes=read(base/'processes.json')
    require(inputs['schema']==native['schema']==managed['schema']==1 and inputs['sample_rate']==16000,'Schema differs')
    require([c['name'] for c in inputs['cases']]==NAMES,'Input coverage differs')
    require(sha(base/'inputs/inputs.json')==frozen['inputs_sha256']==native['inputs_sha256']==managed['inputs_sha256'],'Input manifest hash differs')
    for rel,value in frozen['source'].items():require(sha(base/'source'/rel)==value,'Preserved source differs: '+rel)
    for rel,value in frozen['binaries'].items():require(sha(base/rel)==value,'Binary differs: '+rel)
    require(inputs['generator_sha256']==frozen['source']['tests/parakeet/recording/prepare.py'],'Input generator differs')
    source_path=Path(inputs['source']);require(sha(source_path)==inputs['source_sha256'],'Source fixture manifest differs');source=read(source_path)
    require(inputs['selected']==source['selected'],'Source labels differ')
    source_arrays={}
    for name in ('connected','shifted'):
        original=next(c for c in source['cases'] if c['name']==name);case=next(c for c in inputs['cases'] if c['name']==name)
        require((case['pcm_sha256'],case['reference_text'],case['samples'])==(original['pcm_sha256'],original['reference_text'],original['samples']),'Constructed source differs')
        source_arrays[name]=np.load(base/'inputs'/case['pcm'],allow_pickle=False)
    require(managed['core_sha256']==frozen['binaries']['bin/Lokad.Onnx.dll']==frozen['binaries']['cli-bin/Lokad.Onnx.dll'],'Core differs')
    require(managed['data_sha256']==frozen['binaries']['bin/Lokad.Onnx.Data.dll']==frozen['binaries']['cli-bin/Lokad.Onnx.Data.dll'],'Data differs')
    require(managed['runner_sha256']==frozen['binaries']['bin/ParakeetRecordingReplay.dll'],'Runner differs')
    require(native['generator_sha256']==frozen['source']['tests/parakeet/recording/native.py'] and native['source_sha256']==frozen['source']['external/onnx-asr/src/onnx_asr/asr.py'],'Native source binding differs')
    pins=read(root/'tests/parakeet/transcribe/assets.json');require(native['assets']==pins,'Model pins differ')
    for name,pin in pins['files'].items():require(sha(root/'models/parakeet-tdt-0.6b-v3'/name)==pin['sha256'],'Model differs: '+name)
    vocab={}
    for line in (root/'models/parakeet-tdt-0.6b-v3/vocab.txt').read_text(encoding='utf-8').splitlines():
        piece,index=line.rsplit(' ',1);vocab[int(index)]=piece.replace('\u2581',' ')
    score_path=root/'tests/audio/accuracy/score.py';spec=importlib.util.spec_from_file_location('fixed_accuracy',score_path);scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    metrics=[];checks=0
    for group in (native,managed):
        require([c['name'] for c in group['cases']]==NAMES+NAMES[:1],'Result coverage differs')
        require([c['repeat'] for c in group['cases']]==[False]*len(NAMES)+[True],'Repeat coverage differs')
    for index,(expected,actual) in enumerate(zip(native['cases'],managed['cases'],strict=True)):
        case=inputs['cases'][index%len(NAMES)];path=base/'inputs'/case['pcm'];require(sha(path)==case['pcm_sha256'],'PCM differs')
        pcm=np.load(path,allow_pickle=False);require(pcm.dtype==np.float32 and pcm.shape==(case['samples'],) and np.isfinite(pcm).all(),'PCM contract differs')
        wave=base/'inputs'/case['wave'];require(sha(wave)==case['wave_sha256'],'WAV hash differs');raw=wave.read_bytes()
        header=struct.pack('<4sI4s4sIHHIIHH4sI',b'RIFF',36+len(pcm)*4,b'WAVE',b'fmt ',16,3,1,16000,64000,4,32,b'data',len(pcm)*4)
        require(raw[:44]==header and raw[44:]==pcm.astype('<f4',copy=False).tobytes(),'WAV/PCM differs')
        if case['name']=='hard-boundary':
            require(np.array_equal(pcm,source_arrays['connected']+np.float32(.02)) and case['reference_text']==inputs['cases'][0]['reference_text'],'DC stress differs')
        elif case['name']=='maximum-speech':require(np.array_equal(pcm,np.resize(source_arrays['connected'],9600000)),'Maximum stress differs')
        elif case['name'] in ('token-limit','window-limit'):require(np.array_equal(pcm,source_arrays['connected']),'Limit source differs')
        require(expected['result']==actual['result'],'Native/managed mismatch: '+case['name'])
        validate(actual['result'],case,pcm,vocab)
        checks+=sum(w['decoding']['stop_reason']!='SilentInput' for w in expected['result']['windows'])
        if index<3:metrics.append(dict(name=case['name'],**scorer.metrics(case['reference_text'],actual['result']['text'])))
    require(native['upstream_crosschecks']==checks and native['numpy']=='2.2.4' and native['ort']=='1.29.0','Original upstream/dependency coverage differs')
    require(managed['cases'][0]['result']==managed['cases'][-1]['result'] and managed['ownership'],'Ownership/repeat differs')
    expected_refusals=['sample-rate','too-long','windows-0','windows-513','tokens-0','tokens-4097','per-frame-0','per-frame-11','nonfinite-NaN','nonfinite-Infinity','nonfinite--Infinity','null-options','null-decoding','short-api-bound','pre-canceled','during-inference']
    require(canonical_refusals(managed['refusals'])==expected_refusals,'Refusal coverage differs')
    require(len(managed['concurrent'])==2 and managed['concurrent'][0]==managed['concurrent'][1],'Concurrent silent results differ')
    validate(managed['concurrent'][0],dict(max_tokens=4096,max_tokens_per_frame=10,max_windows=256),np.zeros(496000,np.float32),vocab)
    for name in ('connected','token-limit'):
        row=next(c for c in native['cases'] if c['name']==name);require(read(base/('cli-'+name+'.stdout'))==row['result'],'CLI differs: '+name)
        error=(base/('cli-'+name+'.stderr')).read_text(encoding='utf-8')
        require(('stopped before completion' in error) if name=='token-limit' else error=='','CLI diagnostic differs')
    require(processes['complete'] and [r['name'] for r in processes['runs']]==['native','managed','cli-connected','cli-token-limit'],'Process coverage differs')
    for row in processes['runs']:
        require(row['code']==0 and row['seconds']<1800,'Worker termination/bound differs')
        samples=[json.loads(line) for line in (base/(row['name']+'-samples.jsonl')).read_text(encoding='utf-8').splitlines()]
        require(len(samples)==row['samples'] and len(samples)>0,'Sample coverage differs')
        peak=0;members={};previous_seconds=-1
        for sample in samples:
            require(previous_seconds<sample['seconds']<=row['seconds'],'Sample time differs');previous_seconds=sample['seconds']
            for member in sample['members']:
                require(member['affinity']==[2] and member['rss']>=0 and member['cpu_seconds']>=0,'Worker affinity/resource differs')
                require(member['create_time']>=row['create_time'],'Unrelated older process')
                if member['pid']==row['pid']:require(member['create_time']==row['create_time'],'Worker identity differs')
                key=(member['pid'],member['create_time'])
                require(member['cpu_seconds']>=members.get(key,0),'Process CPU moved backwards');members[key]=member['cpu_seconds']
            peak=max(peak,sum(m['rss'] for m in sample['members']))
        require(peak==row['peak_rss'] and peak<20*1024**3,'RSS bound differs')
        require((row['pid'],row['create_time']) in members,'No worker samples')
        # CREATE_NO_WINDOW may own a small conhost child; include it in RSS and termination checks.
        for pid,created in members:
            try:require(psutil.Process(pid).create_time()!=created,'Worker/child is still alive')
            except psutil.NoSuchProcess:pass
    output=dict(schema=1,complete=True,recording_requests=len(managed['cases']),concurrent_silent_requests=2,cli_requests=2,upstream_window_crosschecks=checks,
        refusals=len(managed['refusals']),metrics=metrics,core_sha256=managed['core_sha256'],data_sha256=managed['data_sha256'],runner_sha256=managed['runner_sha256'],
        observed_managed_peak=next(r['peak_rss'] for r in processes['runs'] if r['name']=='managed'),managed_process_peak=managed['peak_working_set'],
        audit_sha256=sha(__file__),scorer_sha256=sha(score_path),audit_dependencies={n:importlib.metadata.version(n) for n in ('numpy','psutil','jiwer')},bindings={str(p.relative_to(base)):sha(p) for p in [base/'frozen.json',base/'inputs/inputs.json',base/'native/result.json',base/'managed/result.json',base/'processes.json']})
    with a.output.open('x',encoding='utf-8') as f:json.dump(output,f,indent=2);f.write('\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
