"""Prepare pinned local Community-1 PLDA parameters for the managed clusterer."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import scipy
from scipy.linalg import eigh

def prepare(directory: Path, output: Path):
    if output.exists(): raise FileExistsError('Choose a new prepared model output.')
    if np.__version__ != '2.2.4' or scipy.__version__ != '1.16.3':
        raise ValueError('Preparation requires NumPy 2.2.4 and SciPy 1.16.3.')
    pins = {'plda.npz':'9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255',
        'xvec_transform.npz':'325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f'}
    for name,digest in pins.items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest: raise ValueError('Pinned asset mismatch: '+name)
    with np.load(directory/'xvec_transform.npz',allow_pickle=False) as x, np.load(directory/'plda.npz',allow_pickle=False) as p:
        mean1=x['mean1'];mean2=x['mean2'];lda=x['lda'];mean=p['mu'];tr=p['tr'];psi=p['psi']
    for value,shape,dtype in [(mean1,(256,),'float64'),(mean2,(128,),'float32'),(lda,(256,128),'float32'),
                              (mean,(128,),'float64'),(tr,(128,128),'float64'),(psi,(128,),'float64')]:
        if value.shape!=shape or value.dtype!=np.dtype(dtype) or not np.isfinite(value).all(): raise ValueError('Invalid learned tensor.')
    if not (psi>0).all(): raise ValueError('Covariances must be positive.')
    within=np.linalg.inv(tr.T.dot(tr));between=np.linalg.inv((tr.T/psi).dot(tr))
    eigenvalues,eigenvectors=eigh(between,within)
    residual=between@eigenvectors-(within@eigenvectors)*eigenvalues
    scale=max(1.,float(np.linalg.norm(between)*np.linalg.norm(eigenvectors)))
    if np.linalg.norm(residual)/scale>1e-10: raise ValueError('Whitening decomposition residual.')
    phi=eigenvalues[::-1];transform=eigenvectors.T[::-1]
    if not (phi>0).all() or not np.isfinite(transform).all(): raise ValueError('Invalid prepared transform.')
    record=dict(schema=1,input_dimensions=256,output_dimensions=128,mean1=mean1.tolist(),mean2=mean2.tolist(),lda=lda.tolist(),
        mean=mean.tolist(),transform=transform.tolist(),phi=phi.tolist(),assets=pins,numpy=np.__version__,scipy=scipy.__version__,
        source_repository='pyannote/speaker-diarization-community-1',source_revision='3533c8cf8e369892e6b79ff1bf80f7b0286a54ee',
        pyannote_audio_revision='a1ed3bb0440d33d18622e1cb6b138431cfaf4f7a',
        vbx_source_lf_sha256='812c8c4ba276ba0521689693f84690f6fc9af3d67dec232aac45c1fc2d41906f')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f: json.dump(record,f,indent=2)
    return dict(output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),bytes=output.stat().st_size,
                decomposition_relative_residual=float(np.linalg.norm(residual)/scale))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model-directory',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.model_directory,args.output),indent=2))
