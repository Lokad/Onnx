"""Independent direct-DFT mathematical fixtures; Python is not needed by ordinary tests."""
from pathlib import Path
import hashlib,json
import numpy as np

assert np.__version__=='2.2.4'
destination=Path(__file__).resolve().parents[2]/'Lokad.Onnx.Backend.Tests/fixtures/wespeaker-frontend.json'
if destination.exists():raise FileExistsError(destination)
cases=[]
indices=np.arange(512,dtype=np.float64)
dft=np.exp(-2j*np.pi*np.arange(257)[:,None]*indices[None,:]/512)
window=.54-.46*np.cos(2*np.pi*np.arange(400,dtype=np.float64)/399)
edges=np.linspace(1127*np.log1p(20/700),1127*np.log1p(8000/700),82)
mel=1127*np.log1p(np.arange(257)*16000/512/700)
weights=np.maximum(0,np.minimum((mel[None,:]-edges[:-2,None])/(edges[1:-1,None]-edges[:-2,None]),
    (edges[2:,None]-mel[None,:])/(edges[2:,None]-edges[1:-1,None])))
weights[:,-1]=0
for kind,length in [('noise',881),('quiet',881),('impulse',1040),('tones',1040),('dc',880)]:
    values=np.empty(length,np.float32);state=20260919
    for i in range(length):
        state=(1664525*state+1013904223)&0xffffffff
        values[i]=(((state>>8)&65535)-32768)/131072
    if kind=='quiet':values*=np.float32(1e-6)
    if kind=='impulse':values[:]=0;values[[0,199,400,800,length-1]]=[.5,-.25,.75,-.5,.25]
    if kind=='tones':
        t=np.arange(length)/16000
        values=(.2*np.sin(2*np.pi*440*t)+.1*np.sin(2*np.pi*1000*t)+.05*np.sin(2*np.pi*3100*t)).astype(np.float32)
    if kind=='dc':values[:]=.25
    frames=np.lib.stride_tricks.sliding_window_view(values.astype(np.float64)*32768,400)[::160].copy()
    frames-=frames.mean(axis=1,keepdims=True)
    frames=frames-.97*np.concatenate([frames[:,:1],frames[:,:-1]],axis=1)
    padded=np.pad(frames*window,((0,0),(0,112)))
    spectrum=padded@dft.T
    powers=np.abs(spectrum)**2
    features=np.log(np.maximum(powers@weights.T,np.finfo(np.float32).eps));features-=features.mean(axis=0,keepdims=True)
    cases.append(dict(kind=kind,length=length,input=values.tolist(),input_sha256=hashlib.sha256(values.tobytes()).hexdigest(),shape=[1,len(frames),80],values=features.astype(np.float32).ravel().tolist()))
record=dict(scope='Independent float64 direct Fourier sum and mathematical WeSpeaker preprocessing, not a replacement for native conformance',
    numpy=np.__version__,generator_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),tolerance=1e-4,cases=cases)
destination.write_text(json.dumps(record,separators=(',',':'))+'\n',encoding='utf-8')
print(destination)
