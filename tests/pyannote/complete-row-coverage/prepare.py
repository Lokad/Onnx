"""Correct only workload coverage, preserving guarded kernels and conditioned protocol."""
from common import *
from release import released

def main():
    release_state=released()
    census=ROOT/'artifacts/pyannote-convolution-epilogue-20260921'
    assert pin(census/'closed.json')['sha256']=='aaf2a6457d49f087b536b1966427ce1ab580554a568652f79c23a40b85ea4b84'
    proof=read(census/'closed.json');verify(proof['files'])
    analysis=read(census/'analysis.json');assert analysis['passed'] and len(analysis['all_tile_shapes'])==22
    path=ROOT/'tests/pyannote/portable-row-groups/prepare.py'
    source=path.read_text(encoding='utf8')
    changes=[
        ("if (kh,kw) == (1,1): continue", """attrs=row['attributes']
        if ((kh,kw)==(1,1) and attrs['strides']==[1,1] and attrs['dilations']==[1,1]
            and attrs['pads']==[0,0,0,0] and row['input_shape'][2:]==row['output_shape'][2:]): continue"""),
        ("shutil.copy2(TOOLS/'Probe.cs',BASE/'consumer/Probe.cs')", "shutil.copy2(ROOT/'tests/pyannote/portable-row-groups/ProbeV3.cs',BASE/'consumer/Probe.cs')"),
        ("    BASE.mkdir();", "    assert set(shapes)=={tuple(r[k] for k in ['m','n','k']) for r in COVERAGE['all_tile_shapes']}\n    BASE.mkdir();"),
        ("    pins[rel(ort)] = pin(ort)", """    pins[rel(ort)] = pin(ort)
    pins.update(PROOF['files'])
    for p in [CENSUS/'closed.json',RELEASE_STATE,*[ROOT/'tests/pyannote/portable-row-groups'/n for n in ['prepare.py','run.py','audit.py','ProbeV3.cs']]]:pins[rel(p)]=pin(p)"""),
        ('All full/final tiled convolution dimensions; excludes pointwise1x1 path.',
         'Complete22shape coverage including strided1x1; excludes only the actual stride1 pointwise fast path.')]
    for old,new in changes:
        assert source.count(old)==1,old;source=source.replace(old,new)
    namespace=dict(__name__='corrected_complete_shape_preparation',__file__=str(path),COVERAGE=analysis,PROOF=proof,CENSUS=census,RELEASE_STATE=release_state)
    exec(compile(source,str(path),'exec'),namespace)
    namespace['main']()

if __name__=='__main__':main()
