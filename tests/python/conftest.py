import os
import urllib

def pytest_runtest_setup(item):
    print('foo')
    file_dir = os.path.dirname(os.path.realpath(__file__))
    onnx_model_file = os.path.join(file_dir, 'me5s.model.onnx')
    if not os.path.isfile(onnx_model_file):
        print (f'Downloading multilingual embedded-small 5 ONNX model file to {onnx_model_file}...')
        dl,_ = urllib.request.urlretrieve("https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model.onnx?download=true", onnx_model_file)
        if dl != onnx_model_file: 
            raise RuntimeError(f'Could not download model file from https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model.onnx.')
        else:
            print('download complete.')