"""Pinned FP32 Parakeet native pipeline, with independent upstream loop crosscheck."""
from pathlib import Path
import argparse
import ast
import collections
import hashlib
import json
import re
import types
import numpy as np
import onnxruntime as ort


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def source_sha(path):return hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def upstream_method(source, cls, method, namespace):
    tree=ast.parse(source)
    definition=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name==cls
                    for n in c.body if isinstance(n,ast.FunctionDef) and n.name==method)
    # Execute the actual pinned method unmodified, with postponed annotations;
    # this does not import the package's downloader or create a native dependency
    # in the managed replay. Only its greedy loop/text cleanup are needed here.
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),definition],type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module,'pinned-onnx-asr/asr.py','exec'),namespace)
    return namespace[method]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--speech',type=Path,required=True)
    parser.add_argument('--reference-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert (np.__version__,ort.__version__)==('2.2.4','1.29.0')
    args.output.mkdir(parents=True,exist_ok=False)
    assets_path=Path(__file__).with_name('assets.json')
    assets=json.loads(assets_path.read_text(encoding='utf-8'))
    for name,entry in assets['files'].items():
        assert (args.models/name).stat().st_size==entry['bytes'] and sha(args.models/name)==entry['sha256'],name
    upstream=args.reference_source/'src/onnx_asr/asr.py'
    upstream_bytes=upstream.read_bytes()
    assert source_sha(upstream)==assets['reference']['asr_lf_sha256'],'Reference source differs'
    namespace=dict(np=np,re=re,TimestampedResult=collections.namedtuple('TimestampedResult','text timestamps tokens logprobs'))
    upstream_loop=upstream_method(upstream_bytes.decode('utf-8'),'_AsrWithTransducerDecoding','_decoding',namespace)
    upstream_text=upstream_method(upstream_bytes.decode('utf-8'),'_AsrWithDecoding','_decode_tokens',namespace)
    vocab={}
    for line in (args.models/'vocab.txt').read_text(encoding='utf-8').splitlines():
        piece,index=line.rsplit(' ',1);vocab[int(index)]=piece.replace('\u2581',' ')
    assert len(vocab)==8193 and vocab[8192]=='<blk>'
    options=ort.SessionOptions();options.log_severity_level=4
    options.intra_op_num_threads=options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    for setting in ('session.intra_op.allow_spinning','session.inter_op.allow_spinning'):options.add_session_config_entry(setting,'0')
    sessions={name:ort.InferenceSession(str(args.models/file),options,providers=['CPUExecutionProvider'])
              for name,file in assets['graphs'].items()}
    files={};cases=[]
    def save(name,value):
        value=np.ascontiguousarray(value)
        assert value.dtype in (np.dtype('float32'),np.dtype('int32'),np.dtype('int64')) and np.isfinite(value).all()
        path=args.output/(name+'.npy');assert not path.exists()
        np.save(path,value,allow_pickle=False)
        files[path.name]=dict(dtype=str(value.dtype),shape=list(value.shape),bytes=path.stat().st_size,sha256=sha(path))
        return path.name
    def run(model,inputs):
        before={k:v.tobytes() for k,v in inputs.items()}
        values=sessions[model].run(None,inputs)
        assert all(v.tobytes()==before[k] for k,v in inputs.items()),'Native input changed'
        return dict(zip([v.name for v in sessions[model].get_outputs()],values))
    speech=json.loads(args.speech.read_text(encoding='utf-8'))
    specs=[(name,name,4096,10) for name in ('english-16k','french-44k-stereo','jfk-48k-stereo')]
    specs += [('english-token-limit','english-16k',3,10),('english-frame-limit','english-16k',4096,1),
              ('silence','silence-32k',4096,10),('english-repeat','english-16k',4096,10)]
    for name,input_name,maximum,per_frame in specs:
        original=next(c for c in speech['cases'] if c['name']==input_name)
        source=args.speech.parent/original['pcm']
        assert sha(source)==original['pcm_sha256']
        pcm=np.load(source,allow_pickle=False).reshape(-1)
        assert pcm.dtype==np.float32 and np.isfinite(pcm).all() and len(pcm)<=480000
        case=dict(name=name,pcm=save(name+'-pcm',pcm),max_tokens=maximum,max_tokens_per_frame=per_frame,
                  source=dict(speech_manifest_sha256=sha(args.speech),original=original),stages=[],steps=[])
        if not np.any(pcm):
            case['expected']=dict(text='',token_ids=[],frame_indices=[],duration_frames=[],stop_reason='SilentInput',encoded_frames=0,decoder_calls=0)
        else:
            assert len(pcm)>=257
            prepared=run('frontend',dict(waveforms=pcm[None,:],waveforms_lens=np.array([len(pcm)],np.int64)))
            case['stages'].append(dict(model='frontend',outputs={k:save(name+'-frontend-'+k,v) for k,v in prepared.items()}))
            encoded=run('encoder',dict(audio_signal=prepared['features'],length=prepared['features_lens']))
            case['stages'].append(dict(model='encoder',outputs={k:save(name+'-encoder-'+k,v) for k,v in encoded.items()}))
            hidden=encoded['outputs'];frames=int(encoded['encoded_lengths'][0])
            assert hidden.shape==(1,1024,frames) and frames==(len(pcm)//160+1+7)//8
            zero=lambda:(np.zeros((2,1,640),np.float32),np.zeros((2,1,640),np.float32))
            state=zero();tokens=[];positions=[];durations=[];frame=0;emitted=0
            def decode(previous,states,vector):
                return run('decoder',dict(encoder_outputs=np.ascontiguousarray(vector[None,:,None]),
                    targets=np.array([[previous[-1] if previous else 8192]],np.int32),target_length=np.array([1],np.int32),
                    input_states_1=states[0],input_states_2=states[1]))
            while frame<frames and len(tokens)<maximum:
                outputs=decode(tokens,state,hidden[0,:,frame])
                assert outputs['outputs'].shape==(1,1,1,8198) and np.array_equal(outputs['prednet_lengths'],[1])
                logits=outputs['outputs'].reshape(-1)
                token=int(np.argmax(logits[:8193]));duration=int(np.argmax(logits[8193:]))
                index=len(case['steps'])
                case['steps'].append(dict(frame=frame,target=tokens[-1] if tokens else 8192,token=token,duration=duration,
                    state_input_sha256=[hashlib.sha256(v.tobytes()).hexdigest() for v in state],
                    outputs={k:save(name+'-step-'+str(index)+'-'+k,v) for k,v in outputs.items()}))
                if token!=8192:
                    tokens.append(token);positions.append(frame);durations.append(duration)
                    state=(outputs['output_states_1'],outputs['output_states_2']);emitted+=1
                if duration:
                    frame+=duration;emitted=0
                elif token==8192 or emitted==per_frame:
                    frame+=1;emitted=0
            # Independently execute the pinned upstream loop with its own states.
            def reference_decode(prev_tokens,prev_state,vector):
                values=decode(prev_tokens,prev_state,vector);logits=values['outputs'].reshape(-1)
                return logits[:8193],int(logits[8193:].argmax()),(values['output_states_1'],values['output_states_2'])
            shim=types.SimpleNamespace(use_low_precision=False,_blank_idx=8192,_vocab_size=8193,_max_tokens_per_step=per_frame,
                _create_state=zero,_decode=reference_decode,_vocab=vocab,
                DECODE_SPACE_PATTERN=re.compile(r'\A\s|\s\B|(\s)\b'),window_step=.01,_subsampling_factor=8)
            upstream_tokens,upstream_frames,_=next(upstream_loop(shim,hidden.transpose(0,2,1),np.array([frames],np.int64)))
            assert tokens==upstream_tokens[:len(tokens)] and positions==upstream_frames[:len(tokens)]
            if frame>=frames:assert tokens==upstream_tokens
            text=upstream_text(shim,tokens,positions,None).text
            case['expected']=dict(text=text,token_ids=tokens,frame_indices=positions,duration_frames=durations,
                stop_reason='EndOfAudio' if frame>=frames else 'TokenLimit',encoded_frames=frames,decoder_calls=len(case['steps']))
            case['upstream_crosscheck']=dict(tokens=len(upstream_tokens),matched=True)
        cases.append(case)
        print(name,json.dumps(case['expected'],ensure_ascii=True),flush=True)
        (args.output/'progress.json').write_text(json.dumps(dict(files=files,cases=cases),indent=2)+'\n',encoding='utf-8')
    assert cases[0]['expected']==cases[-1]['expected'],'Native repeated request differs'
    manifest=dict(schema=1,scope='parakeet-transcription',scaled_absolute_tolerance=1e-4,assets=assets,
        assets_lf_sha256=source_sha(assets_path),generator_lf_sha256=source_sha(Path(__file__)),
        numpy=np.__version__,onnxruntime=ort.__version__,reference_raw_sha256=sha(upstream),native_settings=dict(provider='CPUExecutionProvider',threads=1,
        execution='sequential',optimization='all',spinning=False),files=files,cases=cases)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
