"""Independently audit recording coverage, provenance, timelines and native application decisions."""
from pathlib import Path
import argparse,hashlib,json,math,sys
import numpy as np
from tokenizers import Tokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'audio/accuracy'))
from score import metrics

BEGIN,END=50365,50257
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def decisions(value):
    if isinstance(value,dict):return {k:decisions(v) for k,v in value.items() if k not in ('no_speech_probability','average_log_probability')}
    if isinstance(value,list):return [decisions(v) for v in value]
    return value

def window_commit(decoding,length,offset,tokenizer):
    if decoding['skipped_as_no_speech'] or decoding['stop_reason']=='SilentInput':
        return [],0 if decoding['stop_reason']=='TokenLimit' else length
    tokens=[t for t in decoding['token_ids'] if t!=END]
    breaks=[i for i in range(1,len(tokens)) if tokens[i-1]>=BEGIN and tokens[i]>=BEGIN]
    closes=bool(len(tokens)>1 and tokens[-1]>=BEGIN and tokens[-2]<BEGIN)
    limited=decoding['stop_reason']=='TokenLimit';slices=[]
    if breaks or limited:
        if closes:breaks.append(len(tokens))
        cursor=0;advance=0
        for boundary in breaks:
            part=tokens[cursor:boundary]
            require(part[0]>=BEGIN and part[-1]>=BEGIN,'Missing segment timestamp')
            start=(part[0]-BEGIN)*320;end=(part[-1]-BEGIN)*320
            slices.append((start,end,part));advance=end;cursor=boundary
        if closes and not limited:advance=length
    else:
        times=[t-BEGIN for t in tokens if t>=BEGIN]
        slices=[(0,times[-1]*320 if times and times[-1]>0 else length,tokens)];advance=length
    expected=[]
    for start,end,part in slices:
        ids=[t for t in part if t<END];text=tokenizer.decode(ids,skip_special_tokens=True)
        start=min(length,start);end=min(length,end)
        if end>start and text.strip():expected.append(dict(start_seconds=(offset+start)/16000,end_seconds=(offset+end)/16000,text=text,token_ids=ids))
    return expected,min(length,advance)

def validate_recording(result,case,tokenizer):
    count=case['samples'];position=0;windows=result['windows'];segments=result['segments']
    reconstructed=[]
    require(result['duration_seconds']==count/16000,'Recording duration differs')
    require(0<len(windows)<=case['max_windows'],'Window count differs')
    for w in windows:
        length=min(480000,count-position)
        require(w['start_seconds']==position/16000 and w['audio_seconds']==length/16000,'Window coverage differs')
        advance=round(w['advanced_seconds']*16000)
        require(w['advanced_seconds']==advance/16000 and 0<=advance<=length,'Invalid advance')
        d=w['decoding'];tokens=d['token_ids']
        require(len(tokens)<=case['max_new_tokens'] and all(type(t) is int and (0<=t<=END or BEGIN<=t<51866) for t in tokens),'Token contract differs')
        if d['stop_reason']=='EndToken':require(tokens and tokens[-1]==END and END not in tokens[:-1],'EOS contract differs')
        elif d['stop_reason']=='TokenLimit':require(len(tokens)==case['max_new_tokens'] and END not in tokens,'Token limit differs')
        else:require(d['stop_reason']=='SilentInput' and not tokens and d['skipped_as_no_speech'],'Silent contract differs')
        require(d['text']==('' if d['skipped_as_no_speech'] else tokenizer.decode([t for t in tokens if t<BEGIN],skip_special_tokens=True)),'Window text differs')
        if d['stop_reason']!='SilentInput':
            ns,lp=d['no_speech_probability'],d['average_log_probability']
            require(math.isfinite(ns) and 0<=ns<=1 and math.isfinite(lp) and lp<=0,'Nonfinite/invalid confidence')
            require(d['skipped_as_no_speech']==(ns>.6 and lp<=-1),'No-speech policy differs')
        committed,expected_advance=window_commit(d,length,position,tokenizer)
        require(advance==expected_advance,'Seek does not match actual timestamp decisions')
        reconstructed.extend(committed)
        position+=advance
    require(segments==reconstructed,'Segments do not match actual window tokens')
    require(result['processed_seconds']==position/16000,'Processed position differs')
    stop=result['stop_reason']
    if stop=='Completed':require(position==count,'Completed recording has unprocessed audio')
    elif stop=='WindowLimit':require(position<count and len(windows)==case['max_windows'],'Window-limit status differs')
    elif stop=='TokenLimit':require(windows[-1]['decoding']['stop_reason']=='TokenLimit','Token-limit status differs')
    else:require(stop=='NoProgress' and windows[-1]['advanced_seconds']==0,'Unexpected recording stop')
    previous=0
    for segment in segments:
        require(previous<=segment['start_seconds']<segment['end_seconds']<=position/16000,'Invalid or uncommitted segment interval')
        require(segment['text'].strip() and segment['text']==tokenizer.decode(segment['token_ids'],skip_special_tokens=True),'Segment text differs')
        require(all(type(t) is int and 0<=t<END for t in segment['token_ids']),'Nontext segment token')
        previous=segment['end_seconds']
    require(result['text']==tokenizer.decode([t for s in segments for t in s['token_ids']],skip_special_tokens=True),'Aggregate text differs')

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--native',type=Path,required=True)
    p.add_argument('--managed',type=Path,required=True);p.add_argument('--frozen',type=Path,required=True);p.add_argument('--models',type=Path,required=True)
    p.add_argument('--cli',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    data=json.loads(a.inputs.read_text());native=json.loads(a.native.read_text());managed=json.loads(a.managed.read_text());frozen=json.loads(a.frozen.read_text())
    names=['connected','shifted','token-limit','window-limit']
    require(data['schema']==native['schema']==managed['schema']==1,'Schema differs')
    require(native['inputs_sha256']==managed['inputs_sha256']==frozen['inputs_sha256']==sha(a.inputs),'Input binding differs')
    require([c['name'] for c in data['cases']]==[c['name'] for c in native['cases']]==names,'Reference coverage differs')
    require([c['name'] for c in managed['cases']]==names+names[:1] and [c['repeat'] for c in managed['cases']]==[False]*4+[True],'Managed coverage differs')
    require(managed['core_sha256']==frozen['binaries']['recording']['Lokad.Onnx.dll']
        and managed['data_sha256']==frozen['binaries']['recording']['Lokad.Onnx.Data.dll']
        and managed['runner_sha256']==frozen['binaries']['recording']['RecordingReplay.dll'],'Build identity differs')
    require(managed['refusals']==10 and managed['ownership'] and all(c['ownership'] for c in managed['cases']),'Refusal/ownership proof differs')
    require(managed['cases'][0]['result']==managed['cases'][4]['result'],'Repeated recording differs')
    require(not managed['flags'],'Unexpected environment overrides')
    for filename,item in native['assets']['files'].items():require(sha(a.models/filename)==item['sha256'] and (a.models/filename).stat().st_size==item['bytes'],'Model identity differs')
    tokenizer=Tokenizer.from_file(str(a.models/'tokenizer.json'));rows=[];values=0
    expected_arrays=set()
    for case,n,m in zip(data['cases'],native['cases'],managed['cases'],strict=False):
        require(sha(a.inputs.parent/case['pcm'])==case['pcm_sha256']==n['pcm_sha256']==m['pcm_sha256'],'PCM identity differs')
        require(sha(a.inputs.parent/case['wave'])==case['wave_sha256'],'WAV identity differs')
        for result in (n['result'],m['result']):validate_recording(result,case,tokenizer)
        confidence=[]
        for i,window in enumerate(n['result']['windows']):
            if window['decoding']['stop_reason']=='SilentInput':continue
            prefix=case['name']+f'-w{i:03d}'
            expected_arrays|={prefix+'-features.npy',prefix+'-hidden.npy'}
            expected_arrays|={prefix+f'-{step:03d}-logits.npy' for step in range(len(window['decoding']['token_ids']))}
        for x,y in zip(n['result']['windows'],m['result']['windows']):
            for key in ('no_speech_probability','average_log_probability'):
                if x['decoding'][key] is not None and y['decoding'][key] is not None:confidence.append(abs(x['decoding'][key]-y['decoding'][key]))
        row=dict(name=case['name'],application_equal=decisions(n['result'])==decisions(m['result']),
            native_windows=len(n['result']['windows']),managed_windows=len(m['result']['windows']),
            native_segments=len(n['result']['segments']),managed_segments=len(m['result']['segments']),
            maximum_observed_confidence_difference=max(confidence,default=0),stop_reason=m['result']['stop_reason'])
        if case['name'] in ('connected','shifted'):
            row['native_accuracy']=metrics(case['reference_text'],n['result']['text']);row['managed_accuracy']=metrics(case['reference_text'],m['result']['text'])
        rows.append(row)
    require(set(native['files'])==expected_arrays,'Native array coverage differs')
    require({p.name for p in a.native.parent.glob('*.npy')}==expected_arrays,'Unlisted native array')
    for name,item in native['files'].items():
        require(Path(name).name==name and sha(a.native.parent/name)==item['sha256'] and (a.native.parent/name).stat().st_size==item['bytes'],'Native array identity differs')
        array=np.load(a.native.parent/name,allow_pickle=False,mmap_mode='r')
        require(array.dtype==np.float32 and list(array.shape)==item['shape'] and np.isfinite(array).all(),'Native array contract differs');values+=array.size
    cli=None
    if a.cli:
        cli=json.loads(a.cli.read_text());require(cli==managed['cases'][0]['result'],'CLI and API differ')
    summary=dict(schema=1,inputs_sha256=sha(a.inputs),native_sha256=sha(a.native),managed_sha256=sha(a.managed),frozen_sha256=sha(a.frozen),
        auditor_sha256=sha(Path(__file__)),native_array_count=len(expected_arrays),native_values=values,cases=rows,
        application_passed=all(r['application_equal'] for r in rows),cli_matches=cli is not None,refusals=managed['refusals'],
        short_regression=managed['short_regression'],peak_working_set=managed['peak_working_set'],
        scope='Application decisions and ownership. Full managed tensor qualification and independent long-conversation accuracy remain open.')
    a.output.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('native_array_count','native_values','application_passed','cli_matches')}))
    for row in rows:print(row['name'],row['application_equal'],row['native_windows'],row['managed_windows'])
    return 0 if summary['application_passed'] else 2

if __name__=='__main__':raise SystemExit(main())
