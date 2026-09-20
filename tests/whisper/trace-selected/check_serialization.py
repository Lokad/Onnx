"""Verify large-model external serialization options on the actual ORT with a tiny model; no inference."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from common import ROOT,pin,read,write
sys.path.append(str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True);args=parser.parse_args();base=args.artifact.resolve()
    assert not (base/'serialization-check.json').exists();process=psutil.Process();old=process.cpu_affinity();process.cpu_affinity([2])
    try:
        import onnx,onnxruntime as ort
        assert ort.__version__=='1.29.0' and np.__version__=='2.2.4'
        folder=base/'serialization-check';folder.mkdir()
        h=onnx.helper;t=onnx.TensorProto
        graph=h.make_graph([h.make_node('MatMul',['x','w'],['y'])],'serialization-only',
            [h.make_tensor_value_info('x',t.FLOAT,[1,128])],[h.make_tensor_value_info('y',t.FLOAT,[1,128])],
            [onnx.numpy_helper.from_array(np.arange(128*128,dtype=np.float32).reshape(128,128),name='w')])
        model=h.make_model(graph,opset_imports=[h.make_opsetid('',14)],ir_version=8);onnx.save(model,folder/'input.onnx')
        options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
        options.optimized_model_filepath=str(folder/'optimized.onnx')
        options.add_session_config_entry('session.optimized_model_external_initializers_file_name','optimized.weights')
        options.add_session_config_entry('session.optimized_model_external_initializers_min_size_in_bytes','1024')
        session=ort.InferenceSession(str(folder/'input.onnx'),options,providers=['CPUExecutionProvider'])
        optimized=onnx.load(folder/'optimized.onnx',load_external_data=False)
        locations=[{e.key:e.value for e in value.external_data}.get('location') for value in optimized.graph.initializer if value.external_data]
        assert locations==['optimized.weights'] and (folder/'optimized.weights').stat().st_size==128*128*4
        assert np.array_equal(np.fromfile(folder/'optimized.weights',dtype='<f4'),np.arange(128*128,dtype=np.float32))
        del session
        write(base/'serialization-check.json',dict(passed=True,pid=process.pid,birth=process.create_time(),onnxruntime=ort.__version__,
            sources=pin(Path(__file__)),files={p.name:pin(p) for p in folder.iterdir()},scope='Load/serialize exact tiny initializer only; no model Run call',
            documentation='https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h'))
        print('Actual ORT external serialization options pass; no inference performed.')
    finally:process.cpu_affinity(old)

if __name__=='__main__':main()
