"""Reuse the qualified native decoder and original seek rules without array files.

The two nested functions are compiled directly from the frozen generator's AST,
without editing their bodies. Only their array observer differs: finite arrays
are checked and counted rather than written. This is application evidence, not
a replacement for complete intermediate numerical qualification.
"""
from pathlib import Path
import ast
import hashlib
import sys
from common import load,read


def functions(generator,namespace):
    tree=ast.parse(Path(generator).read_text(encoding='utf-8'))
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    names={'decode_window','segments_for'}
    nodes=[n for n in main.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==2 and {n.name for n in nodes}==names
    identity=hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode()).hexdigest()
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(generator),'exec'),namespace)
    return namespace['decode_window'],namespace['segments_for'],identity


def recording(pcm,case,decode,segments_for,tokenizer):
    position=0;windows=[];segments=[];stop='Completed'
    while position<len(pcm):
        if len(windows)==case['max_windows']:stop='WindowLimit';break
        length=min(480000,len(pcm)-position);name=case['name']+f'-w{len(windows):03d}'
        result=decode(pcm[position:position+length],case,name)
        committed,advance=segments_for(result,position,length);segments.extend(committed)
        windows.append(dict(start_seconds=position/16000,audio_seconds=length/16000,advanced_seconds=advance/16000,decoding=result))
        position+=advance
        if result['stop_reason']=='TokenLimit':stop='TokenLimit';break
        if advance==0:stop='NoProgress';break
    return dict(text=tokenizer.decode([t for s in segments for t in s['token_ids']],skip_special_tokens=True),
        segments=segments,windows=windows,stop_reason=stop,duration_seconds=len(pcm)/16000,processed_seconds=position/16000)


class WhisperRecording:
    def __init__(self,root,base,manifest):
        generator=base/'reference-source/tests/whisper/recording/generate_reference.py'
        sys.path.insert(0,str(generator.parent))
        helper=load('qualified_whisper_recording',generator)
        np,torch,ort=helper.np,helper.torch,helper.ort
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        models=root/manifest['families']['whisper']['model_directory']
        source=base/'upstream/whisper'
        config=read(models/'generation_config.json')
        tokenizer=helper.Tokenizer.from_file(str(models/'tokenizer.json'))
        extractor=helper.WhisperFeatureExtractor(feature_size=128,sampling_rate=16000,hop_length=160,chunk_length=30,n_fft=400,dither=0.0)
        settings=ort.SessionOptions();settings.intra_op_num_threads=settings.inter_op_num_threads=1
        settings.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;settings.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        for name in ['session.intra_op.allow_spinning','session.inter_op.allow_spinning']:settings.add_session_config_entry(name,'0')
        sessions={name:ort.InferenceSession(str(models/'onnx'/filename),settings,providers=['CPUExecutionProvider'])
            for name,filename in [('encoder','encoder_model.onnx'),('first','decoder_model.onnx'),('past','decoder_with_past_model.onnx')]}
        assert all(s.get_providers()==['CPUExecutionProvider'] for s in sessions.values())
        self.array_count=0;self.value_count=0
        def observe(name,array):
            assert array.dtype==np.float32 and np.isfinite(array).all(),name
            self.array_count+=1;self.value_count+=int(array.size)
        scope=dict(vars(helper),config=config,tokenizer=tokenizer,extractor=extractor,sessions=sessions,
            timestamp=helper.upstream(source/'decoding.py'),seek=helper.seek_program(source/'transcribe.py'),save_array=observe)
        self.decode,self.segments_for,self.body_sha256=functions(generator,scope)
        self.tokenizer=tokenizer

    def recording(self,pcm,case):
        return recording(pcm,case,self.decode,self.segments_for,self.tokenizer)
